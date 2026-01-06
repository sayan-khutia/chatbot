# 🌍 Travel Planner Chatbot

A sophisticated travel planning chatbot built with **LangChain** and **LangGraph**. This agent helps users plan trips by searching flights, finding hotels, suggesting activities, and providing weather information.

## ✨ Features

- **🛫 Flight Search** - Find flights between cities on specific dates
- **🏨 Hotel Finder** - Discover accommodation options with ratings and prices
- **🎯 Activity Suggestions** - Get personalized activity recommendations based on interests
- **🌦️ Weather Information** - Check weather forecasts for your destination
- **🧠 Memory System** - Maintains user preferences and conversation history across sessions
- **📋 Multi-step Planning** - Breaks down complex travel requests into actionable steps

## 🏗️ Architecture

The chatbot uses a **LangGraph** state machine with the following components:

```
START → llm_call → [tools] → llm_call → [update_user_memory, update_agent_memory] → END
```

- **llm_call**: Main LLM node that processes user queries and decides on tool usage
- **tools**: Executes travel-related tools (flights, hotels, activities, weather)
- **update_user_memory**: Summarizes and stores user preferences
- **update_agent_memory**: Summarizes and stores agent's recommendations

## 📋 Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A Groq API key (free tier available)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
cd /home/sayan-khutia/workspace/chatbot
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Required: Groq API Key
# Get your free API key from: https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key_here

# Optional: LangChain Tracing (for debugging)
LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your_langchain_api_key_here

# Optional: OpenAI API Key (if you want to use GPT-4o instead)
# OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Install Dependencies

**Using uv (Recommended):**

```bash
uv sync
```

**Using pip:**

```bash
pip install -r requirements.txt
```

### 4. Run the Chatbot

**Using uv:**

```bash
uv run python chatbot.py
```

**Using Python directly (after pip install):**

```bash
python chatbot.py
```

## 📖 Usage Examples

The chatbot can handle various travel-related queries:

### Weather Query
```python
response = graph.invoke({
    "messages": [
        HumanMessage(content="What's the weather like in Bali from July 25-30, 2025?"),
    ], 
    "language": "English"
}, config=config)
```

### Full Trip Planning
```python
response = graph.invoke({
    "messages": [
        HumanMessage(content="Plan my trip from Mumbai to Bali for July 25-30, 2025. I'm interested in beaches and culture."),
    ], 
    "language": "English"
}, config=config)
```

### Hotel Search
```python
response = graph.invoke({
    "messages": [
        HumanMessage(content="Find me hotels in Bali from July 25 to July 30, 2025"),
    ], 
    "language": "English"
}, config=config)
```

## 🛠️ Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `search_flights` | Search for available flights | `from_city`, `to_city`, `date` |
| `find_hotels` | Find hotel options | `location`, `checkin`, `checkout` |
| `suggest_activities` | Get activity recommendations | `location`, `interests` (list) |
| `get_location_weather` | Get weather forecast | `location`, `date` |

## 🔧 Configuration

### Using Different LLMs

The default LLM is **Groq's Gemma2-9b-it**. To use **OpenAI GPT-4o** instead:

1. Set your `OPENAI_API_KEY` in `.env`
2. In `chatbot.py`, comment out the Groq LLM and uncomment the OpenAI LLM:

```python
# llm = ChatGroq(
#     model="gemma2-9b-it",
#     ...
# )

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
```

### Thread Configuration

Each conversation uses a unique thread ID for memory management:

```python
config = {"configurable": {"thread_id": "1", "user_id": "1"}}
```

- `thread_id`: Unique identifier for conversation continuity
- `user_id`: Unique identifier for user-specific memory

## 📁 Project Structure

```
chatbot/
├── chatbot.py        # Main chatbot implementation
├── main.py           # Simple entry point (optional)
├── pyproject.toml    # Project configuration and dependencies
├── requirements.txt  # Pip-compatible dependencies (auto-generated)
├── uv.lock          # UV lock file for reproducible builds
├── .env             # Environment variables (create this)
└── README.md        # This file
```

## 🔍 Debugging

Enable LangSmith tracing for detailed debugging:

1. Create an account at [smith.langchain.com](https://smith.langchain.com/)
2. Get your API key
3. Add to `.env`:
   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_key_here
   ```

## ⚠️ Notes

- The tools currently return mock data for demonstration purposes
- In-memory storage is used (data doesn't persist between restarts)
- For production use, consider implementing persistent storage with a database

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

