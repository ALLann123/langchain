#!/usr/bin/python3from dotenv import load_dotenv
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
    current_dir, "db", "chroma_db_with_metadata")

# Define the embedding model i.e. text numerical representation
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

#create a retriever for querying the vector store
#search_type -specifies the type of search e.g similarity
#search_kwargs -contains additional arguments for the search(e.g number of results to return)

# Retrieve relevant documents based on query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.1
    }
)

# Create the LangChain chat model using the GitHub Marketplace endpoint
model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

#contextualize question prompt
#this system prompt helps the AI understand that it should reformulate the question
#based on the chat hisory to make it a standalone question

contextualize_q_system_prompt = (
    "Given a chat and the latest user question "
    "which might influence context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

#create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

#create a history-aware retriever
#This uses the LLM to reformulate the question based on chat history

history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
)

#Answer question prompt
#This system prompt helps AI understand that it should provide concise answers
#based on retrieved context and indicates what to do if the answer is unknown

qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, respond with "
    "I do not know. Use three sentences maximum and keep the answer "
    "concise. "
    "\n\n"
    "{context}"
)

#create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


#create a chain to combine documents for question answering
#`create_stuff_documents_chain` feeds all the retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

#create a retrieval chain that combines the history-aware retriver and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#function to simulate a continual chat

def continual_chat():
    print("\n****** WELCOME TO THE WORLD OF MR. ROBOT ******")
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
 python .\conversation_rag.py

****** WELCOME TO THE WORLD OF MR. ROBOT ******
Start chatting with AI! Type 'exit' to end the conversation.

own_the_net: Who is Eliot?
-----------------------------------------------------------------------
AI: Elliot Alderson is the main protagonist of the television series *Mr. Robot*. He is a brilliant but troubled cybersecurity engineer and hacker with dissociative identity disorder, who battles his own fractured psyche while taking on global power structures. Throughout the series, his journey revolves 
around unmasking truths about himself and the world.

own_the_net: Where does Elliot Work?
-----------------------------------------------------------------------
AI: Elliot works at Allsafe Cybersecurity, where he protects networks during the day.

own_the_net: Who is Mr.Robot?
-----------------------------------------------------------------------
AI: Mr. Robot is a figment of Elliot Alderson's imagination, a projection of his deceased father. He represents a dominant personality within Elliot's dissociative identity disorder and serves as the leader of the hacking group fsociety.

own_the_net: Do they have a cyber crime group?
-----------------------------------------------------------------------
AI: Yes, Elliot is part of a hacking collective called fsociety, which aims to bring down powerful corporations like E Corp.

own_the_net: give me a small summary of the season four?
-----------------------------------------------------------------------
AI: In Season 4 of *Mr. Robot*, Elliot and Mr. Robot work together to take down the Deus Group, the architects of global control, with help from Darlene. They succeed in a massive digital heist, but Whiterose's mysterious machine poses a major threat. Elliot confronts buried childhood trauma and discovers an alternate reality, questioning the nature of his psyche and the world around him.

own_the_net: exit
'''