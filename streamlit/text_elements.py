#!/usr/bin/python3
import streamlit as st
import os

st.set_page_config(page_title="Dark Mode App", layout="wide")
st.title("This is a Title")
st.header("This is a header")
st.subheader("This is a subheader")

st.markdown("This is **Bold**")
st.markdown("This is _italic_")

st.caption("Small text")

code_example="""
def greetings(name):
    print("Hello", name)
"""
st.code(code_example, language='python')

st.divider()

st.image(os.path.join(os.getcwd(), "static", "zeus.jpg"), width=700)




