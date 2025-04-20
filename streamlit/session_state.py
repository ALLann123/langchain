#!/usr/bin/python3
import streamlit as st
import pandas as pd
from datetime import datetime

if "counter" not in st.session_state:
    st.session_state.counter=0

if st.button("Increment Counter"):
    st.session_state.counter+=1
    st.write(f"Counter incremented to {st.session_state.counter}")


if st.button("Reset"):
    st.session_state.counter=0

else:
    st.write(f"Counter did not reset ")

st.write(f"Counter Value: {st.session_state.counter}") 
