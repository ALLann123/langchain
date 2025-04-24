#!/usr/bin/python3
import streamlit as st
import os
from chatgpt_with_streamlit.app.streamlit_utils import stdout_streaming, load_chat_history, save_chat_history, remove_file_extension
from chatgpt_with_streamlit.GetLanguageModel.call_llm import get_chat_llm_chain

#streamlit application
def run():
    #sidebar components
    ##handle message history on the sidebar
    message_history_files=[file for file in os.listdir() if file.endswith("_message.json")] #handle message storing
    sidebar=st.sidebar

    with sidebar:
        #select LLM option[OpenAI, Ollama, Gemini]
        save_chat_btn=st.button("Save Chat", use_container_width=True)
        st.markdown("""___""")
        for file in message_history_files:
            if st.button(file.rstrip("_message.json"), use_container_width=True):
                pass
                #st.session_state.messages=load_chat_history

    if save_chat_btn:
        if len(st.session_state.messages)!= 0:
            #save_chat_history(st.session_state.messages)
            pass
        st.session_state.messages=[]
    #image icon similar to openAI
    img_path="./static/logo.jpg"
    cols=st.columns(10)
    with cols[4]:
        st.image(img_path, width=100)
    
    st.markdown("<h1> style='text-align: center;'>How Can I help you today? </h2>")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_input := st.chat_inpuy("Message ChatGPT..."):
        #user message handling
        st.session_state.messages.append(
            {"role":"user", "content":user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        #AI message handling
        with st.chat_message("ai"):
            with st.spinner("Thinking...."):
                output=st.empty()
                #function to pass stdout to streamlit
                #LLMs may stream on the stdout
                #call streamlist utils function
                with stdout_streaming(output.info):
                    #call LLM Chain and invoke the LLM Model to generate response
                    response=...
                    



run()