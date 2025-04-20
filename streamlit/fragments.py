#!/usr/bin/python3
import streamlit as st

st.title("My Awesome APP")

@st.fragment()
def toggle_and_text():
    cols=st.columns(2)
    cols[0].toggle("Toggle")
    cols[1].text_area("Enter text")

@st.fragment()
def filter_and_file():
    new_cols=st.columns(5)
    new_cols[0].checkbox("Filter")
    new_cols[1].file_uploader("Upload image")
    new_cols[2].selectbox("Choose option", ["Python", "C++", "Rust", "Java", "Go"])
    new_cols[3].slider("Select Value", 0, 100, 50)
    new_cols[4].text_input("Enter text:")

toggle_and_text()
cols=st.columns(2)
cols[0].selectbox("Select", [1,2,3], None)
cols[1].button("Update")
filter_and_file()

