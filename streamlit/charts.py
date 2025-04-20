#!/usr/bin/python3
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plit

st.set_page_config(page_title="Chart")

#Title
st.title("Charts Demo")

#generate sample data
chart_data=pd.DataFrame(
    np.random.randn(20,3),
    columns=['A', 'B', 'C'] 
)


#Area CHart section
st.subheader("Area Chart")
st.area_chart(chart_data)


#Bar chart section
st.subheader("Bar Chart")
st.bar_chart(chart_data)

#line chart section
st.subheader("Line Chart")
st.line_chart(chart_data)

#scatter chart section
st.subheader("Scatter Chart")
scatter_data=pd.DataFrame({
    'x':np.random.randn(100),
    'y':np.random.randn(100)
})

st.scatter_chart(scatter_data)

#Map Section("Map")
map_data=pd.DataFrame(
    np.random.randn(100, 2)/[50, 50] + [37.76, -122.4], #cordinates around SF
    columns=['lat', 'lon']
)

st.map(map_data)

st.header("Nairobi Area Map")

# Generate random coordinates around Nairobi (approx. center: -1.2864, 36.8172)
nairobi_coords = [-1.2864, 36.8172]  # Latitude, Longitude of Nairobi
map_data = pd.DataFrame(
    np.random.randn(100, 2) / [50, 50] + nairobi_coords,  # Spread points around Nairobi
    columns=['lat', 'lon']
)

# Display the map
st.map(map_data, zoom=12)  # Adjust zoom level (higher = more zoomed in)

