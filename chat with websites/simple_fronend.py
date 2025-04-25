#!/usr/bin/python3
import streamlit as st

#create frontend
#lets set the page title and icon inthe bar
st.set_page_config(page_title="Chat with websites", page_icon="🤖")

st.title("Chat with websites")

#add sidebar
#the with key word is used to add everything we want to the side bar
with st.sidebar:
    st.header("Settings")
    website_url=st.text_input("Web Site URL")


#lets add chat component for our frontend
st.chat_input("Type your message here...")

#now lets add a place where our chat messages will be
#Add the AI chat messages
with st.chat_message("AI"):
    st.write("Hello, how can I help you?")

#add human history chats
with st.chat_message("human"):
    st.write("I want to know about LangChain")

with st.chat_message("AI"):
    st.write("No")

#the above we have build the front end now lets add some functionality with open AI