#!/usr/bin/python3
import streamlit as st
import pandas as pd


#Title
st.title("Form Demo")

#Form the hold the interactive elements
with st.form(key="sample_form"):

    #text input
    st.subheader("Text Inputs")
    name=st.text_input("Enter your name")
    feedback=st.text_area("Provide your feedback")

    #date and time inputs
    st.subheader("Date and Time Inputs")
    dob=st.date_input("Select your date of birth")
    time=st.time_input("Choose a preferred time")

    #selectors
    st.subheader("Selectors")
    choice=st.radio("Favorite language", ['C', 'C++', 'Python', 'Java'])
    gender=st.selectbox("Select your gender", ['Male', 'Female', 'Other'])
    slider_value=st.select_slider("Select a range", options=[1,2,3,4,5])

    #Toggle and checkboxes
    st.subheader("Toggles & Checkboxes")
    notifications=st.checkbox("Receive notifications?")
    toggle_value=st.checkbox("Enable dark mode?", value=True)

    #submit button for the form
    submit_button=st.form_submit_button(label="Submit")

#outside of the form
st.subheader("Buttons")
if st.button("Click me"):
    st.write(f"Hello, {name}")
    