import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px 
#st.text("Text")
#st.write("Super Function")
#st.header("Header")
#st.subheader("sub-Header")
#st.title("title")
#st.markdown("***Markdown***")
#st.code("print('Hello,world')",language='Python')
#st.latex(r''' e^{i\pi} + 1 = 0 ''')
#displaying interactive widgets
btn=st.button("Submit")
if btn:
    st.info("Info")
option=st.radio("Select",['A','B','C'])
if option=='A':
    st.warning("Warning!")
elif option=='B':
    st.error("Error!")
elif option=='C':   
    st.success("Success ^_^")
chk=st.checkbox("I agree")
if chk:
    st.exception("Agreement") 
box=st.selectbox("Select",['A','B','C'])
if box=='A':
    st.warning("Warning!")
elif box=='B':
    st.error("Error!")
elif box=='C':   
    st.success("Success ^_^")
Age=st.slider("Select Your age",0,100)
st.select_slider("Select",['D','E','F'])
st.text_input("Enter a text")
st.text_area("Enter a paragraph")
st.file_uploader("Upload")
#st.camera_input("Take a photo")
st.date_input("Today")
st.time_input("Now")
st.number_input("Num")
st.multiselect("Select",['Mohamed','Ahmed','Ali'])
st.color_picker("Colors")
####Date Frame 
st.header("Data Frame")
df=sns.load_dataset('taxis')
st.write(df)
st.header("Data Frame Header")
st.dataframe(df.head())
st.header("Data Frame Sample")
st.dataframe(df.sample())

btn=st.button("Show Data")
if btn:
    st.dataframe(df.sample(5))

st.header("Table")
st.table(df.head())

