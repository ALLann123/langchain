#!/usr/bin/python3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

#load environment variables
load_dotenv()

#the llm we are using google gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

#now lets set the system message
message=[
    SystemMessage("You are an expert comedian with a background in programming"),
    HumanMessage("Tell me three jokes about hackers and an extra one about python")
]

#lets send our message to the llm
result=llm.invoke(message)
#get the results 
print(f"AI: {result.content}")



"""
langchain course>python gemini_model_conversation.py
AI: Alright, buckle up for some hacker humor, expertly crafted for your amusement:

1. **The Social Engineer:**  Why was the hacker so good at getting passwords?  Because they were a master of *password phishing*!  They could convince anyone to hand over their credentials just by *complimenting their profile picture*.  It's a real *catfishing* expedition out there.  (Get it? Phishing... like fishing... but for passwords... I'll be here all week.)


2. **The Script Kiddie:** A script kiddie walks into a bar and orders a drink.  The bartender asks for ID.  The kiddie pulls out a USB drive and says, "I have my whole identity on here." The bartender replies, "That's not what I meant.  Do you have something that verifies *you're* the one who *owns* that identity?" The kiddie shrugs and says, "Hold on..." *plugs the USB drive into the bar's POS system*.


3. **The Penetration Tester:** Two penetration testers walk into a bar.  The first one says, "I'm going to hack the jukebox." The second one says, "Don't bother, I already flipped the breaker."  (It's funny because... sometimes the simplest solution is the most effective... and also a bit anticlimactic.  Like a good hack, I guess?)


And now, as promised, your Python treat:

4. **The Zen of Python:**  Why do Python programmers prefer snakes over other pets? Because they understand the importance of *whitespace*!  (This is a deep cut referencing Python's significant whitespace syntax...  If you don't get it, just nod and chuckle politely. It's funnier that way.)
"""
