#!/usr/bin/python3
import streamlit as st
import time

@st.cache_data(ttl=60) #cache this data for 60 seconds

def fetch_data():
    #simulate a slow data fetching process
    time.sleep(3) #delays the mimic an API call
    return {"data": "This is cached data!"}

st.write("Fetching data...")
data=fetch_data()
st.write(data)
