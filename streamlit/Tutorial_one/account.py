#!/usr/bin/python3
import streamlit as st
import base64

def app(uploaded_file=None):
    # Custom CSS for beautified, spaced tabs
    st.markdown("""
        <style>
            .stTabs [data-baseweb="tab-list"] {
                gap: 40px;
                justify-content: center;
            }
            .stTabs [data-baseweb="tab"] {
                font-size: 18px;
                padding: 10px 24px;
                border-radius: 12px;
            }
            .stTabs [aria-selected="true"] {
                background-color: #02ab21;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("📁 RESUME LAB")

    ai_model, cv, cover_letter = st.tabs(["🤖 Chat CV", "📄 CV", "✉️ Cover Letter"])

    with ai_model:
        st.header("🤖 Chat with AI about your CV")
        st.subheader("💬 CV EXPERT LLM")
        st.text_input("What's on your mind?")

    with cv:
        st.subheader("📄 View or Edit Your CV")
        st.write("You are in the CV Tab")
        if uploaded_file:
            st.success(f"Displaying: {uploaded_file.name}")
            if uploaded_file.type == "application/pdf":
            # Display PDF using an iframe
                base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)


    with cover_letter:
        st.subheader("✉️ Generate or Review Cover Letter")
        st.write("You are in the Cover Letter Tab")
