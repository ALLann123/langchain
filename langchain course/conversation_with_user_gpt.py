#!/usr/bin/python3
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load environment variables
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

#use a list to store messages
chat_history=[]

#set an initial system messsage{optional}
system_message=SystemMessage(content="You are an expert mathematician")
chat_history.append(system_message)

#chatloop
print("************"*6)
print("        AI MATHEMATICIAN BOT")
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

    print(f"GPT: {response}")
    print()


    """
    langchain course>python conversation_with_user_gpt.py
************************************************************************
        AI MATHEMATICIAN BOT
************************************************************************

You: what is the square root of 49
GPT: The square root of **49** is **7**.

You: what is the square of 9999
GPT: To calculate the square of **9999**, we compute:

\[
9999^2 = 9999 \times 9999
\]

Using either manual computation or a calculator:

\[
9999^2 = 99,980,001
\]

So, the square of **9999** is **99,980,001**.

You: how about 10
GPT: The square of **10** is:

\[
10^2 = 10 \times 10 = 100
\]

So, the square of **10** is **100**. 😊

You: exit
BYE!!
    """