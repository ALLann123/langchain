#!/usr/bin/python3
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatGroq(
    temperature=0.3,  # Lower temperature for more factual responses
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

result=llm.invoke("What is the square root of 49?")
print(result.content)


"""
langchain course>python chat_groq_starter.py
The square root of 49 is 7.
"""

