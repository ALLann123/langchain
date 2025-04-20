#!/usr/bin/python3
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import streamlit as st

st.sidebar.title("ChatGPT LLM")
sidebar_input=st.sidebar.text_input("Whats on your mind?")


for key in ["step" , "query", "result"]:
    if key not in st.session_state:
        st.session_state[key]= ""

def clear_all():
    st.session_state.query=""


def run_model():
    load_dotenv()
    llm = ChatGroq(
        temperature=0.3,  # Lower temperature for more factual responses
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-70b-8192"
    )
    user_input=st.session_state.query
    llm_output=llm.invoke(user_input)
    st.session_state.result=llm_output.content
    clear_all()


st.title("My Local Ollama")

st.header("Enter Question:")
st.text_input("Whats on your mind?", key="query")

st.button("Send", on_click=run_model)

with st.container(border=True):
    st.subheader("AI Response:")
    st.write(st.session_state.result)
