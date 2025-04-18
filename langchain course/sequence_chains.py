#!/usr/bin/python3
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda

#load environment variables
load_dotenv()

#the llm we are using google gemini
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

animal_facts_template=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a zoologiest who knows facts about {animal}"),
        ("human", "Tell me {fact_count} facts."),
    ]
)


#define a prompt template for translation to French
translation_template=ChatPromptTemplate.from_messages(
    [
        ("system","You are a translator and convert the provided text into {language}."),
        ("human","Translate the following text to {language}: {text}"),
    ]
)

#Define additional processing steps using RunnableLambda
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "french"})

#create the combined chain using LCEL
chain=animal_facts_template | model | StrOutputParser() | prepare_for_translation | translation_template | model | StrOutputParser()

#run the chain
result=chain.invoke({"animal":"Sharks", "fact_count":2})

#output
print(result)



"""
langchain course> python .\sequence_chains.py
1. **Les requins n'ont pas d'os :** Leurs squelettes sont faits de cartilage, un tissu flexible et léger. Cela leur confère une plus grande maniabilité dans l'eau.   

2. **Certains requins peuvent "marcher" :** Des espèces comme le requin-chabot ocellé peuvent utiliser leurs nageoires pour "marcher" sur le fond marin, ce qui leur permet d'explorer les récifs peu profonds et les mares résiduelles à marée basse.
"""