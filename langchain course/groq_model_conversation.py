#!/usr/bin/python3
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
llm = ChatGroq(
    temperature=0.3,  # Lower temperature for more factual responses
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

message=[
    SystemMessage("You are an expert penetration tester with over 10 years experience"),
    HumanMessage("Give a short tip on how to learn binary exploitation")
]

#pass our query into the llm
result=llm.invoke(message)

#get the result content from the llm
print(result.content)


"""
langchain course>python groq_model_conversation.py
Binary exploitation! It's a fascinating and challenging field, but I'm happy to share a tip to help you get started.

**Tip:** Start with the basics of x86 assembly language and focus on understanding the stack.

Here's why:

1. **Assembly language is the foundation**: Binary exploitation involves manipulating machine code, which is written in assembly language. Understanding x86 assembly will help you comprehend how the program flows, how data is stored, and how instructions are executed.
2. **The stack is key**: The stack is a crucial component in binary exploitation. It's where function calls, local variables, and return addresses are stored. Understanding how the stack works will help you identify vulnerabilities, such as buffer overflows, and develop exploits.
3. **Practice with simple exercises**: Begin with simple exercises, such as writing assembly code to perform basic operations like adding two numbers or printing a string. This will help you develop muscle memory and improve your understanding of assembly language.
4. **Use online resources**: There are many excellent online resources available, including tutorials, videos, and blogs. Some popular resources include:
        * The x86 Assembly Language and C Fundamentals course on Udemy
        * The Binary Exploitation tutorials on Hack The Box
        * The Exploit Education tutorials on YouTube
5. **Work on real-world challenges**: Once you have a solid grasp of assembly language and the stack, move on to real-world challenges, such as exploiting vulnerable binaries or participating in capture the flag (CTF) challenges.

Remember, learning binary exploitation takes time, patience, and practice. Focus on building a strong foundation, and you'll be well on your way to becoming a skilled binary exploitation expert.

**Additional resources:**

* "The Art of Exploitation" by Jon Erickson
* "Hacking: The Art of Exploitation" by Jon Erickson (online course)
* "Binary Exploitation" by RPISEC (online course)

Happy learning!
"""