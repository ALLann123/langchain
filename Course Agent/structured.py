#!/usr/bin/python3
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel
import json

# Load environment variables
load_dotenv()

# Define the response format using pydantic
class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

# Define your prompt
messages = [
    SystemMessage(content="Extract the event information and return it strictly as JSON with fields: name, date, participants"),
    HumanMessage(content="Alice and Bob are going to a science fair on Friday."),
]

# Initialize the Groq model
model = ChatGroq(
    temperature=0.3,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

# Get the model response
response = model.invoke(messages)

# Print the response to inspect it
print("Raw model response:\n", response.content)


event_data = json.loads(response.content)
event = CalendarEvent(**event_data)

print("\nExtracted Event:")
print("Name:", event.name)
print("Date:", event.date)
print("Participants:", event.participants)


"""
 python .\structured.py
Raw model response:
 {
"name": "Science Fair",
"date": "Friday",
"participants": ["Alice", "Bob"]
}

Extracted Event:
Name: Science Fair
Date: Friday
Participants: ['Alice', 'Bob']
"""