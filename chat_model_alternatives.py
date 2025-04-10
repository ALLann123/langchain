#!/usr/bin/python3
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the Gemini 2.5 model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=api_key
)

# Loop for user input
while True:
    message = input("Allan: ")
    print("------------------------------------------------------------------")
    
    # Invoke the model
    result = model.invoke(message)
    
    # Print the AI response
    print(f"AI answer >> {result.content}")
