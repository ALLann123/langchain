#!/usr/bin/python3
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

result = llm.invoke("What is the square root of 49?")
print(result.content)


"""
langchain course>python testing_gemini_lang.py
The square root of 49 is 7.  Because 7 * 7 = 49.
"""
