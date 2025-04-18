#!/usr/bin/python3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate

#load environment variables
load_dotenv()

#the llm we are using google gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

#introduction to prompt template
template="Write an {tone} email to {company} expressing interest in the {position} position, mention {skill} as a key strength. Keep it to 4 lines maximum."


#step 1: COnvert template to something LLM understands
prompt_template = ChatPromptTemplate.from_template(template)

#display the above 
#print(prompt_template)

#Create a string with place holders. 
prompt=prompt_template.invoke({
    "tone":"energetic",
    "company":"Ask AI",
    "position":"AI Engineer",
    "skill":"Langchain"
})

#print(prompt)

#now lets pass the prompt to our LLM
result=llm.invoke(prompt)
print(f"AI Response: {result.content}")


"""
langchain course>python prompt_templates.py
AI Response: Subject: AI Engineer - Ready to Build the Future!

Fired up about the AI Engineer role!  My LangChain expertise and passion for innovative AI solutions make me a perfect fit.  Let's talk!

"""