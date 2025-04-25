#!/usr/bin/python3
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

#handle chat history
def get_response(user_input):
    return "I don't know"

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


#----Conversation
for message in st.session_state.chat_history:
    #lets write the AI message on our chat history
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)

    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

