import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

st.title("Data Analysis Application")
st.write("This is a simple data analysis application created by @Sumit")


datasets = ["anagrams","anscombe","attention","brain_networks","car_crashes","diamonds","dots","dowjones","exercise","flights","fmri","geyser","glue","healthexp","iris","mpg","penguins","planets","seaice","taxis","tips","titanic"]

selected_dataset = st.selectbox("Select a dataset",(datasets))

# button to upload your own dataset
uploaded_data = st.file_uploader("Upload your custom dataset", type=["csv","xlsx"])


if selected_dataset !=None:
    df = sns.load_dataset(f"{selected_dataset}")

if uploaded_data is not None:
    df = pd.read_csv(uploaded_data)
  
        
# display the data
st.write(df) 
    
# displaying the number of rows and numbers of columns of data
st.write("Number of rows ",df.shape[0])
st.write("Number of columns ",df.shape[1])


# display the column names of selected data with their datatypes
st.write("Columns names with data type: ", df.dtypes)

# display summary statistics
st.write(df.describe())

# pairplot
# set the hue column
color = st.selectbox("Select column for hue", df.columns)
st.subheader("pairplot")
st.pyplot(sns.pairplot(df,hue=color))
