#!/usr/bin python3
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

#load our api keys
load_dotenv()

#create an instance of our model
model=ChatGroq(
    temperature=0.3,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)


template="""
You are an expert in answering questions about a pizza restaurant
Here are some relevant reviews: {reviews}
Here is the question to answer: {question}
"""

prompt=ChatPromptTemplate.from_template(template)
chain=prompt | model | StrOutputParser()

while True:
    question=input("Ask your question?(q to quit):")
    if question.lower() == "q":
        break
    results=chain.invoke({"reviews":[], "question":question})
    print(results)



