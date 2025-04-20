#!/usr/bin/python3
import streamlit as st

st.title("ADDITION PAGE")


for key in ["step" , "num1", "num2", "result"]:
    if key not in st.session_state:
        st.session_state[key]= ""

def addition():
    num1=float(st.session_state.num1)
    num2=float(st.session_state.num2)
    st.session_state.result=num1+num2

def clear_all():
    st.session_state.num1=""
    st.session_state.num2=""
    st.session_state.result=""


st.header("Add two Numbers")
st.text_input("First Number: ", key="num1")
st.text_input("Second Number: ", key="num2")

st.button("Calculate", on_click=addition)
st.button("Clear", on_click=clear_all)

#show the result
if st.session_state.result != "":
    st.success(f"The Sum: {st.session_state.result}")

