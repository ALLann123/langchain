#!/usr/bin/python3
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.schema.output_parser import StrOutputParser

# Load environment variables
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
model= ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

#define prompt templates 

prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a zoologiest who knows facts about {animal}"),
        ("human", "Tell me {fact_count} facts."),
    ]
)
#create the combined chain using langchain expression language
chain=prompt_template | model | StrOutputParser()
#the above is a sequntial chain

#lets run the chain using the invoke function
result=chain.invoke({"animal":"python", "fact_count":2})

print(result)



"""
langchain course> python .\chains_basics.py
Sure! Here are two fascinating facts about pythons:

1. **Non-Venomous Constrictors**: Pythons are non-venomous snakes and instead rely on constriction to subdue their prey. They wrap their powerful bodies around their 
target and squeeze tightly, cutting off blood flow and oxygen until the prey is immobilized.

2. **Impressive Size Range**: Pythons include some of the largest snake species in the world! For example, the reticulated python (*Malayopython reticulatus*) is not 
only the longest python but also holds the title for being the longest snake in the world, capable of reaching lengths over 30 feet (9 meters). Conversely, the smallest python, the pygmy python (*Antaresia perthensis*), grows to just about 2 feet (60 cm) long.
"""