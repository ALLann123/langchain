#!/usr/bin/python3
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
llm = ChatGroq(
    temperature=0.3,  # Lower temperature for more factual responses
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

#use a list to store messages
chat_history=[]

#set an initial system messsage{optional}
system_message=SystemMessage(content="You are an expert in Kiswahili")
chat_history.append(system_message)

#chatloop
print("************"*6)
print("        Kiswahili Translator BOT")
print("************"*6)
print()
while True:
    #get user input
    query=input("You: ")

    if query.lower() == "exit":
        print("BYE!!")
        break

    #lets add the user meesage to our chat history
    chat_history.append(HumanMessage(content=query))

    #GEN AI response using history
    result=llm.invoke(chat_history)
    response=result.content

    #now add the AIs response to our chat history
    chat_history.append(AIMessage(content=response))

    print(f"OLLAMA: {response}")
    print()


"""
langchain course>python groq_conversation_user.py
************************************************************************
        Kiswahili Translator BOT
************************************************************************

You: I am coming
OLLAMA: Nakuja!

You: where do we go?
OLLAMA: Tunakwenda wapi?

You: I ate pizza
OLLAMA: Nilila pizza!

You:
"""

