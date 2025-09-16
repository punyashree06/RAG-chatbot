from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import asyncio
import os
import shelve  # ✅ Added shelve for memory

from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",google_api_key=os.environ.get('GOOGLE_API_KEY'))

# ✅ Load chat history from shelve
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

# ✅ Save chat history to shelve
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

# ✅ Delete chat history when exiting
def delete_chat_history():
    with shelve.open("chat_history") as db:
        if "messages" in db:
            del db["messages"]

messages = load_chat_history()
sys_prompt = ("You are a immersive story teller, whatever the user gives as input you will generate each scene per time with the output "
              "along with the  voices in the character of the scene and one image relating to the scene, "
              "and the text of the scene. You may add games to the scene which can add more to "
              "the storyline.")
if not messages:
    messages.append({"role": "user", "content": sys_prompt})

async def main(user_input):
    """Process user input while storing chat history."""
    messages = load_chat_history()  # ✅ Load past chat history

    # Ensure system prompt is always first
    system_message = messages[0] if messages and messages[0]["role"] == "system_prompt" else None
    user_messages = [m for m in messages if m["role"] != "system_prompt"]

    # Append new user input
    user_messages.append({"role": "user", "content": user_input})

    # Reconstruct formatted messages string
    formatted_messages = f"System: {system_message['content']}\n" if system_message else ""
    formatted_messages += "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in user_messages])

    async with MultiServerMCPClient(
        {
            "Image_generator": {
                "command": r"D:\mcp_02\.venv\Scripts\python.exe",
                "args": [r"server\visualizer.py"],
                "transport": "stdio",
            }  
        }
    ) as client:
        agent = create_react_agent(model, client.get_tools())
        agent_response = await agent.ainvoke({"messages": formatted_messages})  # ✅ Pass as string
        
        response_text = agent_response['messages'][-1].content  # Extract response
        
        # Save the response
        messages.append({"role": "assistant", "content": response_text})
        save_chat_history(messages)  # ✅ Update history
        
        return response_text
    

if __name__ == "__main__":
    print("💬 Welcome to the AI Chat! Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("🗑️ Deleting chat history...")
            delete_chat_history()  # ✅ Clear history on exit
            print("👋 Exiting chat. Goodbye!")
            break

        response = asyncio.run(main(user_input))
        print("\n🤖 AI:", response, "\n")  