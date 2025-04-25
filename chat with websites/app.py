#!/usr/bin/python3
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from sentence_transformers import SentenceTransformer

load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")
# Define the directory containing files and persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

#handle chat history
def get_response(user_input):
    return "I don't know"

def get_vectorstore_from_url(url):
    #---Gets the text from the url in document form
    #scrape the side and store to the loader variable
    loader=WebBaseLoader(url)
    #load the variable into memory
    documents=loader.load()

    #now lets split the text into chunks
    text_splitter=RecursiveCharacterTextSplitter()
    document_chunks=text_splitter.split_documents(documents)

    #create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={'normalize_embeddings': True}
    )

     #create vector database
    vector_store = Chroma.from_documents(
        documents=document_chunks,
        embedding=embeddings,
        persist_directory=persistent_directory,
        collection_metadata={"hnsw:num_threads": 1}  # For better stability
    )
    return vector_store



def get_context_retriever_chain(vector_store):
    # Create the LangChain chat model using the GitHub Marketplace endpoint
    llm = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=api_key,
        base_url="https://models.inference.ai.azure.com"
    )


    retriever=vector_store.as_retriever()

    prompt=ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),       
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain=create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


#-----App COnfig
#create frontend
#lets set the page title and icon inthe bar
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

#session state is an object that does not change
if "chat_history" not in st.session_state:
    #handles chat history. We will use the message schema from langchain core
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a Bot How can I help you?"),
    ]



#-----Side Bar
#add sidebar
#the with key word is used to add everything we want to the side bar
with st.sidebar:
    st.header("Settings")
    website_url=st.text_input("Web Site URL")

if website_url is None or website_url =="":
    st.info("Please enter website URL")

else:
    #call our function to create vector store and pass in the website url
    documents=get_vectorstore_from_url(website_url)
    retriever_chain=get_context_retriever_chain(documents)

    #----User Input
    #lets add chat component for our frontend
    #then save the user input to a variable
    user_qeury=st.chat_input("Type your message here...")
    if user_qeury is not None and user_qeury != "":
        response=get_response(user_qeury)
        #lets add what the human wrote to our chathistory
        st.session_state.chat_history.append(HumanMessage(content=user_qeury))
        #now add the response from the AI 
        st.session_state.chat_history.append(AIMessage(content=response))

        retrived_documents=retriever_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input":user_qeury
        })

        st.write(retrived_documents)

    #----Conversation
    for message in st.session_state.chat_history:
        #lets write the AI message on our chat history
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)

        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

