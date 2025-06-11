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

"""
#start conversation
input_message=HumanMessage(content="Hi, I am Allan")
output=graph.invoke({"messages":[input_message]}, config)
for m in output['messages'][-1:]:
    m.pretty_print()



input_message=HumanMessage(content="I like manchester united")
output=graph.invoke({"messages": [input_message]}, config)
for m in output['messages'][-1:]:
    m.pretty_print()

input_message=HumanMessage(content="Bruno Fernandez is my favorite player, is he the highest paid at the club")
output=graph.invoke({"messages":[input_message]}, config)
for m in output['messages'][-1:]:
    m.pretty_print()
"""
input_message=HumanMessage(content="What's my name?")
output=graph.invoke({"messages":[input_message]}, config)
for m in output['messages'][-1:]:
    m.pretty_print()

#Now lets display the summary of the 6 messages from the Human and AI
print()
print("***************Printing Chat Summary**************")
chat_history=graph.get_state(config).values.get("summary", "")

print(chat_history)


"""
python external_memory.py
================================== Ai Message ==================================

Your name is Allan! 😊

***************Printing Chat Summary**************
Sure! Here's an updated summary of our conversation so far:

- You introduced yourself as Allan.
- You shared that you're a fan of Manchester United and mentioned Bruno Fernandes as your favorite player.
- I explained that while Bruno Fernandes is a key player and earns a significant salary (around £240,000 per week after his contract extension in 2022), he is not the highest-paid player at the club. Players like Casemiro (reportedly earning £350,000 per week) are paid more.
- We discussed Bruno's importance to the team, his leadership, and his impressive performances on the pitch.
- You later asked me again what your name is, and I confirmed that your name is Allan.

Let me know if you'd like to explore anything else about Manchester United, Bruno Fernandes, or anything else on your mind! 😊⚽
"""