#!/usr/bin/python3
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from sentence_transformers import SentenceTransformer
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")
model=ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)


# Define the directory containing files and persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db")


def get_vectorstore_from_url(url):

    # Load and split website content
    loader = WebBaseLoader(url)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)  # Added overlap
    document_chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name = "all-MiniLM-L6-v2",
        model_kwargs = {'device': 'cpu'},
        encode_kwargs = {'normalize_embeddings': False}  # Changed to False
    )


    # Create vector store
    vector_store = Chroma.from_documents(document_chunks, embeddings, persist_directory = persistent_directory)
    return vector_store




def get_context_retriever_chain(vector_store):
    # Create the LangChain chat model using the GitHub Marketplace endpoint
    llm = model

    retriever = vector_store.as_retriever(
        search_type = "similarity",  # Changed from similarity_score_threshold
        search_kwargs = {
            "k": 3  # Removed score_threshold
        }
    )


    prompt=ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),       
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain=create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):

    llm=model

    prompt=ChatPromptTemplate.from_messages([
        ("system", "You Are a Penetration tesing Bot. Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain= create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


#handle chat history
def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain=get_conversational_rag_chain(retriever_chain)
    response=conversation_rag_chain.invoke({
        "chat_history":st.session_state.chat_history,
        "input":user_query
    })
    return response["answer"]


#-----App COnfig
#create frontend
#lets set the page title and icon inthe bar
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")


#-----Side Bar
#add sidebar
#the with key word is used to add everything we want to the side bar
with st.sidebar:
    st.header("Settings")
    website_url=st.text_input("Web Site URL")

if website_url is None or website_url =="":
    st.info("Please enter website URL")

else:
    #session state is an object that does not change
    if "chat_history" not in st.session_state:
        #handles chat history. We will use the message schema from langchain core
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a Bot How can I help you?"),
        ]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store=get_vectorstore_from_url(website_url)


    
    #----User Input
    #lets add chat component for our frontend
    #then save the user input to a variable
    user_query=st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response=get_response(user_query)

        #lets add what the human wrote to our chathistory
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        #now add the response from the AI 
        st.session_state.chat_history.append(AIMessage(content=response))



    #----Conversation
    for message in st.session_state.chat_history:
        #lets write the AI message on our chat history
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)

        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

