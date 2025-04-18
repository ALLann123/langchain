#!/usr/bin/python3
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence

# Load environment variables
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a zoologist who knows facts about {animal}"),
        ("human", "Tell me {fact_count} facts."),
    ]
)

# Create individual runnables
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Create chain
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Run the chain with correct keys
response = chain.invoke({"animal": "cat", "fact_count": 2})

# Output
print(response)



"""
langchain course> python .\inner_working_chains.py
Sure! Here are two fascinating facts about cats:

1. **Cats have a unique "vocabulary" of meows just for humans**: Cats don't typically meow to communicate with other cats, but they use a wide variety of meows, chirps, and trills specifically to interact with humans. Each cat can develop a unique "language" of sounds to communicate 
its needs and emotions to its owner.

2. **Cats can rotate their ears 180 degrees**: Cats have 32 muscles in each ear, allowing them to rotate their ears up to 180 degrees to pinpoint the source of a sound. This ability not only helps them locate prey in the wild but also gives them their famously keen sense of hearing! 
PS J:\code\AI\AI\langchain course>
"""