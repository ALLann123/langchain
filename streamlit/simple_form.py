#!/usr/bin/python3
import streamlit as st
import pandas as pd
from datetime import datetime

#title
st.title("User Information Form")

#lest create a form with the key

form_values={
    "name":None,
    "height":None,
    "gender":None,
    "dob":None
}

min_date=datetime(1990,1,1)
max_date=datetime.now()

with st.form(key="user_info_form"):
    form_values["name"]=st.text_input("Enter your name: ")
    form_values["height"]=st.number_input("Enter your height(cm): ")
    form_values["gender"]=st.selectbox("Gender", ["Male", "Female", "other"])
    form_values["dob"]=st.date_input("Enter your date of birth", max_value=max_date, min_value=min_date)
    submit_button=st.form_submit_button()

    #ensure all fields in a form have been field
    if submit_button:
        if not all(form_values.values()):
            st.warning("Please fill in all of the fields")

        else:
            st.balloons()
            st.write("###Info")
            #we are looping through the dictionary
            for (key, value) in form_values.items():
                st.write(f"{key}: {value}")





    