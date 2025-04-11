#!/usr/bin/python3
from dotenv import load_dotenv
from firebase_admin import credentials, firestore, initialize_app
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model
model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

# Firestore setup
project_id = "my-memory-e4baf"
SESSION_ID = "user1_session"
COLLECTION_NAME = "chat_history"

# Initialize firebase with service account
print("[+] Initializing firebase app with credentials...")
cred = credentials.Certificate("serviceAccountkey.json")  # path to your downloaded JSON key
initialize_app(cred)

# Initialize firestore client
print("[+] Initializing firestore client...")
client = firestore.client()

# Initialize Firestore chat history
print("[+] Initializing Firestore chat history...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)

print("[+] Chat history initialized..")
print("Current Chat History:", chat_history.messages)

print("Start chatting with the AI. Type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")
