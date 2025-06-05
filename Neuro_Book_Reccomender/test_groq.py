#!/usr/bin/python3
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    temperature=0.3,  # Lower temperature for more factual responses
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)


response=llm.invoke("Capital city of Kenya")

print(f"AI: {response.content}")