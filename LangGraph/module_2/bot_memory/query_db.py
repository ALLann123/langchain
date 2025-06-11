#!/usr/bin/python3
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

#create our sqlite db to use as llm memory locally
conn=sqlite_conn=sqlite3.connect("checkpoint.sqlite", check_same_thread=False)
memory=SqliteSaver(conn)


#lets set our llm
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)


#lets set the message state
class State(MessagesState):
    summary: str

#-----Nodes
#define the logic to call the model
def call_model(state:State):
    #get summary if it exists
    summary=state.get("summary", "")

    #if there is summary we add it
    if summary:

        #add summary to system message
        system_message=f"Summary of conversation earlier: {summary}"

        #append the summary to any new message
        messages=[SystemMessage(content=system_message)] + state["messages"]

    else:
        messages=state["messages"]

    #now lets invoke the llm
    response=model.invoke(messages)
    return {"messages":response}

#define a node to produce a summary
#N/B RemoveMessage is used to filter out our state after we've produced the summary

def summarize_conversation(state:State):
    #First, we get any existing summary
    summary=state.get("summary", "")

    #Create our summarization prompt
    if summary:
        #A summary already exists
        summary_message=(
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above"
        )

    else:
        summary_message="Creating a summary of the conversation above: "

    
    #add prompt to our history
    messages=state["messages"] + [HumanMessage(content=summary_message)]
    #prompt the model to make a summary
    response=model.invoke(messages)

    #delete all but the 2 most recent messages
    delete_messages=[RemoveMessage(id=m.id) for m in state["messages"][:2]]

    return {"summary": response.content, "messages": delete_messages}


#Lets build a  conditional edge node to determnine wheteher to end or summarize the conversation
def should_continue(state:State):
    """Return the next node to execute."""

    messages=state["messages"]

    #if there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    #otherwise we can just end
    return END


#define a new graph
workflow=StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

#set the entry point
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

#compile the graph
graph=workflow.compile(checkpointer=memory)

#display(Image(graph.get_graph().draw_mermaid_png()))

#lets test the Chatbot
#create a thread
config={"configurable":{"thread_id":1}}

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == 'quit':
        break
            
    output = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
    ai_response = output['messages'][-1].content
    print(f"\nAI: {ai_response}")


    """
    python .\query_db.py

You: Hey whats my name and favorite team?

AI: Your name is Allan, and your favorite team is Manchester United! ⚽🔴 Let me know if there's anything else you'd like to chat about, Allan! 😊

You: I love skating, playing roll ball. And I want to study computer science one day

AI: That's awesome, Allan! Skating and playing roll ball sound like so much fun—it's great that you're into such active and exciting hobbies! 🛼🏀 And wanting to study computer scienc 
e is a fantastic goal. It's such a versatile and in-demand field, with opportunities to work on cutting-edge technologies like artificial intelligence, software development, cybersecurity, and more.

Do you already have some experience with coding or computers, or are you just starting to explore the idea? I'd be happy to share tips or resources to help you get started on your computer science journey! 😊

You: quit



You: whats my name?

AI: Your name is Allan! 😊

You: whats my favorite team?

AI: Your favorite team is Manchester United! 🔴⚽

You: Who is the player I love the most in manchester united?

AI: The player you love the most in Manchester United is **Bruno Fernandes**! 🏟️✨ You admire his le adership, vision, and ability to control the game. He's definitely a key player for the team! 😊

You: what sport do I play?

AI: You play **roll ball**! 🛼🏀 It's such a unique and exciting sport that combines skating and bal l skills. You also enjoy **skating**, which shows your adventurous and active personality! 😊

You: what course will I study?

AI: You aspire to study **computer science**! 💻✨ It's an amazing field with endless opportunities, and I’m sure your curiosity and ambition will help you excel in it. Let me know if 
you'd like tips or resources to get started on your journey! 😊

You: quit
    """