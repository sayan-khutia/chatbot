from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import getpass
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing import Optional, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.store.mongodb import MongoDBStore
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId
from pymongo import MongoClient

load_dotenv()

# Only enable tracing if LANGCHAIN_API_KEY is set
if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system", 
        """
        You are a helpful assistant. Answer user's query simple and easy words in {language} and ask one followup question.

        You are given:
        - A summary of the user's preferences, background, or goals: {user_summary}
        - A summary of your own past suggestions, explanations, or conclusions: {agent_summary}

        Use these summaries to:
        - Give more accurate, personalized, or context-aware answers
        - Avoid repeating information already shared
        - Build on prior advice or decisions when relevant

        You are a helpful, intelligent, and systematic travel planner agent.
        You have access to a set of travel-related tools that allow you to:
        - Search for flights
        - Find hotels
        - Suggest local activities
        - Get location weather

        Your responsibilities span across three key roles:
        1. **Planner** - Break down the user's travel query into actionable steps needed to fulfill it.
        2. **Tool Caller** - Make appropriate tool calls for each step in the plan.
        3. **Verifier** - After each tool call and at the end of the plan, verify that the retrieved data is complete, correct, and sufficient.

        ### 🧩 Step-by-Step Workflow:

        1. **Planning Phase**:
        - When a user asks for a travel plan (e.g., "Plan my trip from Mumbai to Bali"), begin by analyzing the request.
        - Generate a structured set of steps required to build the itinerary (e.g., find flights, check hotels, get local activities, etc.).
        - Do not make any assumptions or fetch data at this stage.

        2. **Execution Phase**:
        - For each step in the plan:
            a. Decide which tool to use.
            b. Call the appropriate tool.
            c. Act as a verifier: Check if the tool result fulfills the step. If incomplete, call again with refined inputs.
        - Continue until all steps have valid data.

        3. **Final Verification Phase**:
        - After all steps are completed, review the full plan.
        - Confirm all necessary data is present and consistent.
        - If anything is missing or ambiguous, repeat tool calls as needed to complete the plan.

        4. **Explanation Phase**:
        - Once verification is done, summarize the final travel plan in a natural, helpful, and friendly way for the user.
        - Ensure the explanation should be told from the perspective of travel planner and not an tool caller

        ### 📌 Rules to Follow:

        - ✅ Only use the tools you are explicitly given access to.
        - 🚫 Do **not** fabricate or hallucinate information.
        - ⛔ If a user asks for something outside tool capabilities, politely say you cannot fulfill it.
        - 💬 Ask follow-up questions **only** if you have a tool to answer the response.
        - 🔍 Be precise, transparent, and accurate in all steps — like a professional travel concierge.

        Your goal is to **plan**, **execute**, and **validate** travel itineraries in a step-by-step, tool-grounded, and trustworthy manner.

        ### 📦 Structured Data (available for verification and explanation):

        Use the following structured data to verify or fulfill user queries **before** making tool calls. 
        If the data already answers the user's request, use it. 
        Do not duplicate tool calls unnecessarily.

        ✈️ **Flights**:
        {flights}   

        🏨 **Hotels**:
        {hotels}

        🎯 **Activities**:
        {activities}

        🌦️ **Weather**:
        {weather}

        This data is meant to:
        - Help fulfill requests using already available info.
        - Allow you to verify correctness before or after tool usage.
        - Save unnecessary calls by reusing available data.
        """
    ),
    ("user", "Hello!"),
    ("ai", "Hello!"),
    MessagesPlaceholder("messages"),
])

user_summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a memory assistant for a user/human. 
            Your job is to keep a compact summary of important details explicitly expressed by the user.
            
            You are given:
            - The current user summary.
            - A full list of user messages (what user said) exchanged with assistant since the last update. List doesn't include any of assistant messages.

            Here is the current user summary:
            {summary}

            Note:                 
            - If you see question in the messages, you should not answer it.
            - Your job is to analyse all the messages and based on your understanding of messages update summary if required.
            - You should not add any new information that is not explicitly stated by the user in messages.
            - You don't need to update summary if you don't find anything to update as such.

            Update the above user summary as follows (if needed). Carefully follow these rules when updating the summary:
            - Extract only facts, preferences, goals, or experiences that the user explicitly stated in their own messages.
            - If the user expresses any physical or emotional experiences, include a brief one-liner.
            - Keep all previously relevant summary points — do not remove good info unless it's outdated or corrected by user.
            - IMPORTANT: **Do not remove or modify unrelated parts of the summary unless the new messages explicitly relate to or update them.**
            - If you add something new to the summary, return the full updated summary by preserving existing relevant entries and appending new ones.
            - Avoid restating literal facts (e.g., "User's name is Varun") if richer insights exist (e.g., "Varun loves exploring cities and their culture").
            - Do not add information based on tool calls, system messages, or any content not clearly from the user.
            - Be concise. Write as if another LLM will use this summary — keep it compact, factual, and clear.
            - Use consistent phrasing for all new entries: e.g., "User mentioned...", "User likes...", "User asked about...", "User prefers...", "User shared...", etc.
            - Do not end the summary with a question, suggestion, or incomplete sentence.

            DO NOT add:
            - Anything the user did not clearly say in their message.
            - Assistant-generated ideas, guesses, or completions.
            - Paraphrases of the assistant's replies.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

agent_summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a memory assistant for an AI agent/assistant.
            Your job is to maintain a concise, high-signal summary of the assistant's useful contributions across a session.

            You are given:
            - The current agent summary
            - A full list of messages exchanged between the user and the AI assistant since the last update

            Here is the current agent summary:
            {agent_summary}

            Update the summary using the following principles:

            - Include only the assistant's most **helpful**, **actionable**, or **user-specific** suggestions, assumptions, instructions, or conclusions
            - Use phrasing like **"Agent suggested..."**, **"Agent recommended..."**, or **"Agent explained..."** when adding new entries
            - **Do not include general world knowledge or common facts** unless the assistant applied it in a uniquely helpful or contextual way
            - If the assistant revises or contradicts an earlier response, replace the outdated part
            - Do not repeat already summarized points
            - Remove obsolete, incorrect, or low-value information
            - Focus on reusable insights or decisions the assistant is building on
            - IMPORTANT: **Do not remove or modify unrelated existing summary points unless clearly revised or invalidated by the assistant in the new messages.**
            - If you add something new to the summary, return the full updated summary by preserving existing relevant entries and appending new ones.
            - Be concise and structured for quick reference by another LLM
            - Read user's messages only for context. Do not include anything that comes solely from the user. Only assistant-authored content should appear in the summary.
            - Ignore tool calls or system messages
            - It's okay if there's nothing new to summarize — do not add filler or low-value content
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


@tool
def search_flights(from_city: str, to_city: str, date: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """
    Tool: search_flights
    Description: Search for flights based on departure, destination, date

    Parameters:
        from_city (str): Departure city.
        to_city (str): Destination city.
        date (str): Date of travel in YYYY-MM-DD format.

    Returns:
        str: A response of available flight options.

    """
    content = f"Flights from {from_city} to {to_city} on {date}:\n"
    content += "- IndiGo 6E-123, 10:00 AM to 4:00 PM, $250\n"
    return Command(update={
        "flights": content,
        "messages": [
            ToolMessage(
                f"Here are some flights options for your trip. {content}", 
                tool_call_id=tool_call_id
            )
        ]
    })


@tool
def find_hotels(location: str, checkin: str, checkout: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """
    Tool: find_hotels
    Description: Find hotel options based on location, check-in/out dates, and optional budget.

    Parameters:
        location (str): City or region where hotel is needed.
        checkin (str): Check-in date in YYYY-MM-DD format.
        checkout (str): Check-out date in YYYY-MM-DD format.

    Returns:
        str: A summary of hotel options.

    """
    content = f"{location}: Bali Beach Resort, ₹90/night, 4.3⭐ (Check-in: {checkin}, Check-out: {checkout})"
    return Command(update={
        "hotels": content,
        "messages": [
            ToolMessage(f"Here are some hotel options for your trip. {content}", tool_call_id=tool_call_id)
        ]
    })


@tool
def suggest_activities(location: str, interests: list, tool_call_id: Annotated[str, InjectedToolCallId]):
    """
    Tool: suggest_activities
    Description: Suggests activities in a location based on user interests.

    Parameters:
        location (str): Travel destination (e.g., "Bali").
        interests (list): List of interest categories (e.g., ["beach", "culture"]).

    Returns:
        str: A list of activities relevant to the location and interests.

    """
    all_activities = {
        "beach": f"{location}: Beach Day at Seminyak – Relax on white sands.",
        "culture": f"{location}: Visit Uluwatu Temple – Explore Balinese culture.",
        "nature": f"{location}: Tegallalang Rice Terrace – Scenic rice fields walk.",
        "adventure": f"{location}: Mount Batur Sunrise Hike – Early morning volcano trek."
    }

    suggestions = [all_activities[i] for i in interests if i in all_activities]
    content = "\n".join(suggestions) or "No activities matched the given interests."
    return Command(update={
        "activities": content,
        "messages": [
            ToolMessage(f"Here are some activities options for your trip. {content}", tool_call_id=tool_call_id)
        ]
    })


@tool
def get_location_weather(location: str, date: str, tool_call_id: Annotated[str, InjectedToolCallId]):
    """
    Tool: get_location_weather
    Description: Returns test weather info for a location on a given date.

    Parameters:
        location (str): Name of the city or place (e.g., "Bali").
        date (str): Target date in YYYY-MM-DD format.

    Returns:
        str: A short forecast.
    """
    content1 = f"Weather in {location} on {date}: 29°C, partly cloudy, 70% humidity, 20% chance of rain."
    content2 = f"Weather in {location} on {date}: 18°C, light rain, 85% humidity, 60% chance of rain."
    content3 = f"Weather in {location} on {date}: 25°C, mostly sunny, 60% humidity, 10% chance of rain."
    if location.lower() == "bali":
        return Command(update={
            "weather": content1,
            "messages": [
                ToolMessage(f"Here is the weather for your given location. {content1}", tool_call_id=tool_call_id)
            ]
        })
    elif location.lower() == "london":
        return Command(update={
            "weather": content2,
            "messages": [
                ToolMessage(f"Here is the weather for your given location. {content2}", tool_call_id=tool_call_id)
            ]
        })
    else:
        return Command(update={
            "weather": content3,
            "messages": [
                ToolMessage(f"Here is the weather for your given location. {content3}", tool_call_id=tool_call_id)
            ]
        })


tools = [search_flights, find_hotels, suggest_activities, get_location_weather]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

chain = prompt_template | llm_with_tools
user_summary_chain = user_summary_prompt | llm
agent_summary_chain = agent_summary_prompt | llm


def merge_str(existing: Optional[str], new: str | list[str]) -> str:
    """Reducer that merges string values, handling multiple tool updates."""
    if isinstance(new, list):
        # Multiple updates in same step - join them
        new_value = "\n".join(new)
    else:
        new_value = new
    
    if existing:
        return f"{existing}\n{new_value}"
    return new_value


class State(BaseModel):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    language: Optional[str] = None
    flights: Annotated[Optional[str], merge_str] = None
    hotels: Annotated[Optional[str], merge_str] = None
    activities: Annotated[Optional[str], merge_str] = None
    weather: Annotated[Optional[str], merge_str] = None


graph_builder = StateGraph(State)
user_summary_namespace = "User's Summary"
agent_summary_namespace = "Agent's Summary"


def concatenate_memories(items):
    return "\n".join(
        item.dict()["value"]["memory"]
        for item in items
        if "memory" in item.dict().get("value", {})
    )


def llm_call(state: State, config: RunnableConfig, *, store: BaseStore):
    # Get the user id from the config
    user_id = config["configurable"]["user_id"]
    # Namespace the memory
    user_namespace = (user_id, user_summary_namespace)
    # Get the summary from the store
    user_summary = store.search(user_namespace)
    # Concatenate the memories
    user_summary_str = concatenate_memories(user_summary)
    # Get the agent summary from the store
    agent_summary = store.search((user_id, agent_summary_namespace))
    # Concatenate the memories
    agent_summary_str = concatenate_memories(agent_summary)
    print("\n")
    print(state.messages)
    new_messages = chain.invoke(
        {
            "messages": state.messages, 
            "language": state.language, 
            "user_summary": user_summary_str, 
            "agent_summary": agent_summary_str,
            "flights": state.flights,
            "hotels": state.hotels,
            "activities": state.activities,
            "weather": state.weather
        })
    return {"messages": [new_messages]}


def update_user_memory(state: State, config: RunnableConfig, *, store: BaseStore):
    # Get the user id from the config
    user_id = config["configurable"]["user_id"]
    # Namespace the memory
    namespace = (user_id, user_summary_namespace)
    # Create a new memory ID
    memory_id = f"{user_id}-summary"
    # fetch previous summary
    prev_summary = store.search(namespace)
    # concatenate previous summary
    prev_summary_str = concatenate_memories(prev_summary)
    # get human messages
    human_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    # generate new summary
    memory = user_summary_chain.invoke({"messages": human_messages, "summary": prev_summary_str})
    memory_text = memory.content if hasattr(memory, "content") else memory
    print("\n")
    print("User Summary:")
    print(memory_text)
    # store new summary inside namespace and InMemoryStore.
    store.put(namespace, memory_id, {"memory": memory_text})


def update_agent_memory(state: State, config: RunnableConfig, *, store: BaseStore):
    # Get the user id from the config
    user_id = config["configurable"]["user_id"]
    # Namespace the memory
    namespace = (user_id, agent_summary_namespace)
    # Summarize the conversation
    # Create a new memory ID
    memory_id = f"{user_id}-agent-summary"
    # We create a new memory
    prev_summary = store.search(namespace)
    prev_summary_str = concatenate_memories(prev_summary)
    memory = agent_summary_chain.invoke({"messages": state.messages, "agent_summary": prev_summary_str})
    memory_text = memory.content if hasattr(memory, "content") else memory
    print("\n")
    print("Agent Summary:")
    print(memory_text)
    store.put(namespace, memory_id, {"memory": memory_text})


def go_to(state: State):
    messages = state.messages
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return ["update_user_memory", "update_agent_memory"]


# Add nodes
graph_builder.add_node("llm_call", llm_call)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("update_user_memory", update_user_memory)
graph_builder.add_node("update_agent_memory", update_agent_memory)

# Add edges
graph_builder.add_edge(START, "llm_call")
graph_builder.add_conditional_edges(
    "llm_call", 
    go_to, 
    ["tools", "update_user_memory", "update_agent_memory"]
)
graph_builder.add_edge("tools", "llm_call")
graph_builder.add_edge("update_user_memory", END)
graph_builder.add_edge("update_agent_memory", END)

# MongoDB connection - set MONGODB_URI in your .env file
# Default to local MongoDB instance if not set
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("MONGODB_DB_NAME", "chatbot_db")

# Initialize MongoDB client
mongo_client = MongoClient(MONGODB_URI)

# MongoDB Checkpointer for conversation state (short-term memory)
# This persists the graph state after each step, allowing conversations to resume
checkpointer = MongoDBSaver(mongo_client, db_name=DB_NAME)

# MongoDB Store for user/agent summaries (long-term memory)
# This stores the user preferences and agent insights across sessions
mongodb_store = MongoDBStore(mongo_client[DB_NAME]["long_term_memory"])

graph = graph_builder.compile(checkpointer=checkpointer, store=mongodb_store)

config = {"configurable": {"thread_id": "1", "user_id": "1"}}
config_2 = {"configurable": {"thread_id": "2", "user_id": "1"}}


def main():
    """Main function to run the chatbot with example queries."""
    print("=" * 60)
    print("🌍 Travel Planner Chatbot - Starting...")
    print(f"📦 Connected to MongoDB: {MONGODB_URI}")
    print(f"📁 Database: {DB_NAME}")
    print("=" * 60)
    
    try:
        # Example: Weather query - First conversation
        print("\n🗣️ First message in conversation thread 1...")
        response = graph.invoke({
            "messages": [
                HumanMessage(content="Hi can you tell me about the weather in bali from 25th July 2026 to 30th July 2026?"),
            ], 
            "language": "English"
        }, config=config)

        print("\n" + "=" * 60)
        print("📨 Response Messages:")
        print("=" * 60)
        for message in response["messages"]:
            print(message.content)
            print("-" * 40)

        # Example: Continue the same conversation - demonstrating persistence
        print("\n🗣️ Continuing conversation thread 1 (agent remembers context)...")
        response2 = graph.invoke({
            "messages": [
                HumanMessage(content="What hotels are available there?"),
            ], 
            "language": "English"
        }, config=config)

        print("\n" + "=" * 60)
        print("📨 Follow-up Response (same thread - has context):")
        print("=" * 60)
        for message in response2["messages"]:
            print(message.content)
            print("-" * 40)

    finally:
        # Clean up MongoDB connection
        mongo_client.close()
        print("\n✅ MongoDB connection closed.")


def get_conversation_history(thread_id: str, user_id: str):
    """
    Retrieve past conversation history from MongoDB.
    
    Args:
        thread_id: The conversation thread identifier
        user_id: The user identifier
    
    Returns:
        List of messages from the conversation
    """
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    state = graph.get_state(config)
    if state and state.values:
        return state.values.get("messages", [])
    return []


def list_user_threads(user_id: str):
    """
    List all conversation threads for a user from MongoDB.
    
    Note: This queries the checkpoints collection directly.
    
    Args:
        user_id: The user identifier
    
    Returns:
        List of thread IDs
    """
    db = mongo_client[DB_NAME]
    checkpoints = db["checkpoints"]
    
    # Find distinct thread_ids for the user
    threads = checkpoints.distinct("thread_id", {"metadata.user_id": user_id})
    return threads


if __name__ == "__main__":
    main()
