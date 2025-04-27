#!/usr/bin/python3
from dotenv import load_dotenv
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

# Define persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(
    current_dir, "db", "chroma_db_kilimall")

# Define the embedding model i.e. text numerical representation
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Load the existing vector store with the embedding function
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}  # Add cosine distance metric
)

# Create a retriever for querying the vector store
# search_type -specifies the type of search e.g similarity
# search_kwargs -contains additional arguments for the search(e.g number of results to return)

# Retrieve relevant documents based on query
retriever = db.as_retriever(
    search_type="similarity",  # Use similarity search
    search_kwargs={
        "k": 3,  #  Return top 3 results
    }
)

# Create the LangChain chat model using the GitHub Marketplace endpoint
model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

# contextualize question prompt
# this system prompt helps the AI understand that it should reformulate the question
# based on the chat hisory to make it a standalone question

contextualize_q_system_prompt = (
    "Given a chat and the latest user question "
    "which might influence context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# create a history-aware retriever
# This uses the LLM to reformulate the question based on chat history

history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
)

# Answer question prompt
# This system prompt helps AI understand that it should provide concise answers
# based on retrieved context and indicates what to do if the answer is unknown

qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, respond with "
    "I do not know. Use three sentences maximum and keep the answer "
    "concise. "
    "\n\n"
    "{context}"
)

# create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all the retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

# create a retrieval chain that combines the history-aware retriver and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# function to simulate a continual chat
def continual_chat():
    print("\n****** SCRAPER BOT AI ******")
    print("Start chatting with AI! Type 'exit' to end the conversation.\n")

    chat_history = []

    while True:
        query = input("own_the_net: ")
        print("-----------------------------------------------------------------------")
        if query.lower() == "exit":
            break

        result = rag_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })

        print(f"AI: {result['answer']}\n")

        # Update chat history
        chat_history.extend([
            HumanMessage(content=query),
            SystemMessage(content=result["answer"])
        ])

# Start chat
continual_chat()


'''
python .\conversation.py

****** SCRAPER BOT AI ******
Start chatting with AI! Type 'exit' to end the conversation.

own_the_net: name of the online store?
-----------------------------------------------------------------------
Number of requested results 3 is greater than number of elements in index 2, updating n_results = 2
AI: The name of the online store is Kilimall.

own_the_net: who are they?
-----------------------------------------------------------------------
Number of requested results 3 is greater than number of elements in index 2, updating n_results = 2
AI: Kilimall is Kenya's leading online shopping mall, launched in June 2014 with the mission of "Enriching Lives for Africa." It offers a wide range of products, including electronics, clothing, home appliances, beauty items, and more, while focusing on customer satisfaction and convenience.

own_the_net: name there products?
-----------------------------------------------------------------------
Number of requested results 3 is greater than number of elements in index 2, updating n_results = 2
AI: Kilimall offers products across various categories, including electronics (smartphones, laptops, subwoofers), clothing, home appliances (refrigerators), beauty items (creams, makeup), shoes, fashion bags, baby products, and accessories.

own_the_net: Do they have T-shirts?
-----------------------------------------------------------------------
Number of requested results 3 is greater than number of elements in index 2, updating n_results = 2
AI: Yes, Kilimall offers clothing, which likely includes T-shirts.

own_the_net: what are the prices?
-----------------------------------------------------------------------
Number of requested results 3 is greater than number of elements in index 2, updating n_results = 2
AI: The prices on Kilimall vary depending on the product. For clothing items like T-shirts, specific prices are not provided in the context, so you would need to check their website or app for exact pricing.

own_the_net: What is the most expensive product on there online store?
-----------------------------------------------------------------------
Number of requested results 3 is greater than number of elements in index 2, updating n_results = 2
AI: The context does not specify the most expensive product on Kilimall's online store. You would need to browse their website or app to determine their 
highest-priced item.

own_the_net: do they offer delivery?
-----------------------------------------------------------------------
Number of requested results 3 is greater than number of elements in index 2, updating n_results = 2
AI: Yes, Kilimall offers delivery services and provides order delivery tracking for customer convenience.

own_the_net: country wide?
-----------------------------------------------------------------------
Number of requested results 3 is greater than number of elements in index 2, updating n_results = 2
AI: The context does not specify whether Kilimall offers delivery across all of Kenya. However, as Kenya's leading online shopping mall, it likely provides delivery to most areas within the country.
'''


'''
The error above exists because:This message comes from the Chroma vector store. It means that when your code tries to retrieve the top 3 most relevant documents (k=3), 
Chroma finds that one of the retrieved documents has fewer than 3 relevant chunks of information.
'''