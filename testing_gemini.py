#!/usr/bin/python3
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure the API key
genai.configure(api_key=api_key)

# Initialize the Gemini 2.5 model
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# Start a chat session (optional for multi-turn)
chat = model.start_chat()

# Send user message
user_input = input("YOU>> ")
print("--------------------------------------------")
response = chat.send_message(user_input)

# Print response
print(f"\n🧠 Gemini Says:\n{response.text}")
