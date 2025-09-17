from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import asyncio
import os
import shelve

from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# ✅ Initialize model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

# ✅ Chat history functions
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

def delete_chat_history():
    with shelve.open("chat_history") as db:
        if "messages" in db:
            del db["messages"]

# ✅ System prompt setup
messages = load_chat_history()
sys_prompt = "You are an expert in Long term Settlement cases. From the context answer all questions."
if not messages:
    messages.append({"role": "system_prompt", "content": sys_prompt})

# ✅ Main async function
async def main(user_input):
    messages = load_chat_history()

    # Ensure system prompt is always first
    system_message = messages[0] if messages and messages[0]["role"] == "system_prompt" else None
    user_messages = [m for m in messages if m["role"] != "system_prompt"]

    # Append new user input
    user_messages.append({"role": "user", "content": user_input})

    # Format messages
    formatted_messages = f"System: {system_message['content']}\n" if system_message else ""
    formatted_messages += "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in user_messages])

    # ✅ Initialize MCP client
    client = MultiServerMCPClient({
        "rag_agent": {
            "command": "/home/codespace/.python/current/bin/python",
            "args": ["server/rag_agent.py"],
            "transport": "stdio",
        }
    })

    tools = await client.get_tools()
    agent = create_react_agent(model, tools)
    agent_response = await agent.ainvoke({"messages": formatted_messages})

    response_text = agent_response['messages'][-1].content

    # Save response
    messages.append({"role": "assistant", "content": response_text})
    save_chat_history(messages)

    return response_text

# ✅ Safe event loop handler
def run_async(func, *args):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(func(*args))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(func(*args))

# ✅ CLI loop
if __name__ == "__main__":
    print("💬 Welcome to the AI Chat! Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("🗑️ Deleting chat history...")
            delete_chat_history()
            print("👋 Exiting chat. Goodbye!")
            break

        response = run_async(main, user_input)
        print("\n🤖 AI:", response, "\n")
