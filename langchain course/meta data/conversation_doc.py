#!/usr/bin/python3
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(
    temperature=0.3,  # Lower temperature for more factual responses
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

#define persistent directory
current_dir=os.path.dirname(os.path.abspath(__file__))
db_dir=os.path.join(current_dir,"db")
persistent_directory=os.path.join(db_dir, "chroma_db_with_metadata")

#define the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Specify device
    encode_kwargs={'normalize_embeddings': True}  # Helps with similarity
)

#load the existing vector store
db=Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

#define the users question
query="Who controls FSociety?"

#Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity",  # Changed to basic similarity search
    search_kwargs={"k": 3}     # Removed score_threshold parameter
)

relevant_docs=retriever.invoke(query)

"""
#display the relevant results with metadata
print("\n---Relevant Documents----")
for i, doc in enumerate(relevant_docs,1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source:{doc.metadata['source']}\n")
"""

#combine the query and the relevant document 
combined_input=(
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide a rough answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure. Lastly,  Keep the response to three statements"

)

model=llm

#define the message prompt for the model
messages=[
    SystemMessage(content="You are a Movie Story Teller."),
    HumanMessage(content=combined_input)
]

#invoke the model with the combined input
result=model.invoke(messages)

#display the full result with the combined input
print()
print("\n---Generated Response---")
print("content only:")
print(result.content)



"""
python .\conversation_doc.py


---Generated Response---
content only:
Based on the provided documents, here is a rough answer to the question "Who controls FSociety?":

FSociety is an underground hacking collective recruited by Mr. Robot, who is a figment of Elliot's imagination. Therefore, it can be inferred 
that Elliot has some level of control or influence over FSociety. However, the documents do not explicitly state who ultimately controls FSociety, suggesting that there may be other forces at play.
PS J:\code\AI\AI\langchain course\meta data> 
"""