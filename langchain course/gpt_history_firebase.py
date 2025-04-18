#!/usr/bin/python3
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from firebase_admin import credentials, firestore, initialize_app
from langchain_google_firestore import FirestoreChatMessageHistory

# Load environment variables
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

# Firestore setup
project_id = "my-memory-e4baf"
SESSION_ID = "chat_history"
COLLECTION_NAME = "langchain_tutorial"

#initialize the firebase with service account
print("[+]Initializing firebase app with credentials...")
#use the json file
cred=credentials.Certificate("serviceAccountkey.json")
initialize_app(cred)

#initialize firestore client
print()
print("[+]Initializing firestore client...")
client=firestore.client()

#initialize firestore chat history
print()
print("[+]Initializing firestore chat history....")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)
print()
print("[+]Chat History initialized...")
print("Current History: ", chat_history.messages)

print("*********************************************************")
print("Start chatting with AI. Type 'exit' to quit")

system_message=SystemMessage(content="You are an expert AI assistant for legal firms in Kenya and you understand Kenyan law. ")

#chatloop
print("************"*6)
print("        LEGAL ASSISTANT")
print("************"*6)
print()
while True:
    #get user input
    query=input("You: ")

    if query.lower() == "exit":
        print("BYE!!")
        break

    #lets add the user meesage to our chat history
    chat_history.add_user_message(query)

     # Combine the system message + chat history
    full_message_history = [system_message] + chat_history.messages

    #GEN AI response using history
    ai_result=llm.invoke(full_message_history)

    ai_response=ai_result.content
    #now add the AIs response to our chat history
    chat_history.add_ai_message(ai_response)

    print(f"Legal Assistant: {ai_response}")
    print()



















"""

J:\code\AI\AI\langchain course>python gpt_history_firebase.py
[+]Initializing firebase app with credentials...

[+]Initializing firestore client...

[+]Initializing firestore chat history....

[+]Chat History initialized...
Current History:  []
*********************************************************
Start chatting with AI. Type 'exit' to quit
************************************************************************
        LEGAL ASSISTANT
************************************************************************

You: Hello, who are you?
Legal Assistant: Hello! I am a legal assistant specialized in Kenyan law. I am here to help you better understand the legal system, answer your questions related to Kenyan laws, regulations, and procedures, and provide guidance on various legal matters. How can I assist you today?

You: state three types of law
Legal Assistant: In Kenya, as in many legal systems, different types of law govern various aspects of society. Here are three key types of law:

1. **Criminal Law**
   - This type of law deals with offenses against the state, public order, and justice. Criminal law defines crimes (e.g., theft, assault, or murder), establishes punishments, and outlines procedures for prosecuting and punishing offenders.
   - In Kenya, criminal law is primarily governed by statutes such as the *Penal Code* (Cap 63) and the *Criminal Procedure Code* (Cap 75).

2. **Civil Law**
   - Civil law addresses disputes between individuals or organizations where one party alleges that the other has caused harm or violated their rights. Civil law includes areas like contract disputes, property disputes, torts (such as negligence), family law, and succession law.
   - In Kenya, civil claims are governed by various laws, including the *Civil Procedure Act* (Cap 21) and specific statutes covering areas like family law, land law, and contractual rights.

3. **Constitutional Law**
   - Constitutional law deals with matters related to the interpretation and application of Kenya's Constitution. It defines the structure of government, the powers and responsibilities of different branches of government, and the rights and freedoms of individuals.
   - In Kenya, constitutional law is grounded in the *Constitution of Kenya, 2010*, which is the supreme law of the land and overrides any other conflicting law.

If you'd like further details or examples of any of these types of law, let me know!

You: quit
Legal Assistant: Understood. If you need any assistance in the future, don't hesitate to reach out. Take care! 🙂

You: exit
BYE!!

J:\code\AI\AI\langchain course>python gpt_history_firebase.py
[+]Initializing firebase app with credentials...

[+]Initializing firestore client...

[+]Initializing firestore chat history....

[+]Chat History initialized...
Current History:  [HumanMessage(content='Hello, who are you?', additional_kwargs={}, response_metadata={}), AIMessage(content='Hello! I am a legal assistant specialized in Kenyan law. I am here to help you better understand the legal system, answer your questions related to Kenyan laws, regulations, and procedures, and provide guidance on various legal matters. How can I assist you today?', additional_kwargs={}, response_metadata={}), HumanMessage(content='state three types of law', additional_kwargs={}, response_metadata={}), AIMessage(content="In Kenya, as in many legal systems, different types of law govern various aspects of society. Here are three key types of law:\n\n1. **Criminal Law**  \n   - This type of law deals with offenses against the state, public order, and justice. Criminal law defines crimes (e.g., theft, assault, or murder), establishes punishments, and outlines procedures for prosecuting and punishing offenders.\n   - In Kenya, criminal law is primarily governed by statutes such as the *Penal Code* (Cap 63) and the *Criminal Procedure Code* (Cap 75).\n\n2. **Civil Law**  \n   - Civil law addresses disputes between individuals or organizations where one party alleges that the other has caused harm or violated their rights. Civil law includes areas like contract disputes, property disputes, torts (such as negligence), family law, and succession law.\n   - In Kenya, civil claims are governed by various laws, including the *Civil Procedure Act* (Cap 21) and specific statutes covering areas like family law, land law, and contractual rights.\n\n3. **Constitutional Law**  \n   - Constitutional law deals with matters related to the interpretation and application of Kenya's Constitution. It defines the structure of government, the powers and responsibilities of different branches of government, and the rights and freedoms of individuals.\n   - In Kenya, constitutional law is grounded in the *Constitution of Kenya, 2010*, which is the supreme law of the land and overrides any other conflicting law.\n\nIf you'd like further details or examples of any of these types of law, let me know!", additional_kwargs={}, response_metadata={}), HumanMessage(content='quit', additional_kwargs={}, response_metadata={}), AIMessage(content="Understood. If you need any assistance in the future, don't hesitate to reach out. Take care! 🙂", additional_kwargs={}, response_metadata={})]
*********************************************************
Start chatting with AI. Type 'exit' to quit
************************************************************************
        LEGAL ASSISTANT
************************************************************************

You: hello, what did we talk about previously?
Legal Assistant: Hello! In our previous conversation, we discussed general legal concepts, including three main types of law in Kenya: **Criminal Law**, **Civil Law**, and **Constitutional Law**. You then mentioned "quit," which ended the discussion.

If you'd like to continue or revisit any of these topics, or if you have a new legal question, feel free to ask! 😊

You:


"""
