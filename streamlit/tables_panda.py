#!/usr/bin/python3
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Pandas Begginer", layout="wide")
st.title("Elements Demo")

# dataframe section
st.subheader("Dataframe")

df=pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25,32,37,45],
    'Occupatiion': ['Engineer', 'Doctor', 'Artist','Chef']    

})

#now display the above pandas data frame
st.dataframe(df)

#data editor section(editable dataframe)
st.subheader("Data Editor")
editable_df=st.data_editor(df)

#static viewing of a table
st.subheader("Static Table")
st.table(df)

#metrics section
st.subheader("Metrics")
st.metric(label="Total Rows", value=len(df))
st.metric(label="Average Age", value=round(df['Age'].mean(), 1))

#json and dict section
st.subheader("JSON and Dictionary")
sample_dict={
    "name":"Alice",
    "age":25,
    "skills":["Python", "Data Science", "Machine Learning"]
}

st.json(sample_dict)

#Also show it as dictionary
st.write("Dictionary:", sample_dict)