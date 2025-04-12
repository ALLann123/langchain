#!/usr/bin/python3
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

#load environment variables
load_dotenv()
api_key=os.getenv("GITHUB_TOKEN")

#lets create our chat Model i.e chatgpt
# Create the LangChain chat model using the GitHub Marketplace endpoint
model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

#define the prompt templates
prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer"),
        ("human", "List the main features of the product {product_name}.")
    ]
)

def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            (
                "human",
                "Given these features: {features}, list the pros of these features.",
            ),
        ]
    )
    return pros_template.format_prompt(features=features)


# Define cons analysis step
def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            (
                "human",
                "Given these features: {features}, list the cons of these features.",
            ),
        ]
    )
    return cons_template.format_prompt(features=features)

# Combine pros and cons into a final review
def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"

# Simplify branches with LCEL
pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

# Run the chain
result = chain.invoke({"product_name": "MacBook Pro"})

# Output
print(result)

