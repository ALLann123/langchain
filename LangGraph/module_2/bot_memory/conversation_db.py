#!/usr/bin/python3
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)
    # Create table if it doesn't exist
    conn.execute("""
    CREATE TABLE IF NOT EXISTS kv_store (
        namespace TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        PRIMARY KEY (namespace, key)
    )
    """)
    conn.commit()
    return SqliteSaver(conn)

memory = init_db()

# Set up LLM
load_dotenv()
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com"
)

# Message state class
class State(MessagesState):
    summary: str

# Nodes
def call_model(state: State):
    summary = state.get("summary", "")
    messages = [SystemMessage(content=f"Summary: {summary}")] + state["messages"] if summary else state["messages"]
    response = model.invoke(messages)
    return {"messages": response}

def summarize_conversation(state: State):
    summary = state.get("summary", "")
    summary_message = f"Extend this summary: {summary}" if summary else "Create a summary of this conversation:"
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    return {"summary": response.content, "messages": [RemoveMessage(id=m.id) for m in state["messages"][:-2]]}

# Graph workflow
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node("summarize", summarize_conversation)
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", lambda s: "summarize" if len(s["messages"]) > 6 else END)
workflow.add_edge("summarize", END)
graph = workflow.compile(checkpointer=memory)

# Database inspection
def show_db_contents():
    conn = sqlite3.connect("checkpoint.sqlite")
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM kv_store WHERE namespace='checkpoints'")
    print("\n=== DATABASE CONTENTS ===")
    for row in cursor.fetchall():
        print(f"Key: {row[0]}\nValue: {row[1][:200]}...\n")
    conn.close()

# Interactive chat
def interactive_chat():
    config = {"configurable": {"thread_id": "user1"}}
    print("Starting chat session (type 'quit' to exit)")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        output = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        ai_response = output['messages'][-1].content
        print(f"\nAI: {ai_response}")
        
        # Show current state and DB
        state = graph.get_state(config)
        print(f"\nCurrent summary: {state.values.get('summary', 'No summary yet')}")
        show_db_contents()

if __name__ == "__main__":
    interactive_chat()