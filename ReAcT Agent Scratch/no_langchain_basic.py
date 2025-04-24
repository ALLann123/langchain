#!/usr/bin/python3
from dotenv import load_dotenv
from groq import Groq
import os

#load the api keys
load_dotenv()

client=Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

chat_completion=client.chat.completions.create(
    messages=[
        {
            "role":"user",
            "content":"Explain the importance of fast language models",
        }

    ],
    model="llama3-70b-8192",
)

print(chat_completion.choices[0].message.content)
