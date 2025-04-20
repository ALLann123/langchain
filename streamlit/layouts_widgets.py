#!/usr/bin/python3
import streamlit as st

#sidebar layout
st.sidebar.title("This is the sidebar")
st.sidebar.write("You can place elements like sliders, button, and text here.")
sidebar_input=st.sidebar.text_input("Enter something in the sidebar")

#tabs layout
tab1,tab2,tab3=st.tabs(["Tab 1", "Tab 2", "Tab 3"])

with tab1:
    st.write("You are in Tab 1")

with tab2:
    st.write("You are in Tab 2")

with tab3:
    st.write("You are in Tab 3")

#columns layout
col1, col2, col3=st.columns(3)

with col1:
    st.header("Column 1")
    st.write("Content for column 1")

with col2:
    st.header("Column 2")
    st.write("Content for column 2")

with col3:
    st.header("Column 3")
    st.write("Content for column 3")


#containers example
with st.container(border=True):
    st.write("This is inside the container.")
    st.write("You can think of containers as a grouping for elements")
    st.write("Containers help manage section of a page. ")

#Empty placeholder
placeholder=st.empty()
placeholder.write("This an empty placeholder, useful for dynamic content")

if st.button("Update Placeholder"):
    placeholder.write("The content of this placeholder has been updated.")

st.divider()
#expander 
with st.expander("Expand for more details"):
    st.write("This is additional information that is hidden by default")
    st.write("You can use expanders to keep your interface cleaner. ")


#popover(Tooltip)
st.write("Hove over this button for a tooltip")
st.button("Button with Tooltip", help="This is a tool tip or popover on hover")

#sidebar input handling
if sidebar_input:
    st.write(f"You entered in the sidebar: {sidebar_input}")


