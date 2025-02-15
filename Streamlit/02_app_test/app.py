import time
import streamlit as st
import pandas as pd 
import numpy as np

# Adding the title
st.title("Upload and analyze") 

# # write on page
# st.write("here is a simple text")

# # getting user input from slider
# number = st.slider("Pick a number",0,100,12)

# # write on page
# st.write(number)

# # adding button
# if st.button("say hello"):
#     st.write("hi, hello sir")
# else:
#     st.write("bye")
    
# # adding radio button
# genere = st.radio("What's your favorite movie genera",("horror",'comedy',"adventure"))
# st.write(genere)

# # add a drop down list
# option = st.selectbox('How would you like to be contacted?', ('Email', 'Home phone', 'Mobile phone'))
# st.write(option)


# # add a drop down list to sidebar
# option = st.sidebar.selectbox('How would you like to be contacted?', ('Email', 'Home phone', 'Mobile phone'))
# st.write(option)

# # user input
# userinput = st.sidebar.text_input("Enter your phone number")
# st.write(userinput)


# file uploader
file = st.sidebar.file_uploader("Upload your csv file and get the insights from it",type="csv")

if file!=None:
    df = pd.read_csv(file)
    st.write(df.head(2))

if st.button("Click to Analyze"):
    if file!=None:
        
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1, text=progress_text)

        # Display a message once the progress bar is complete
        my_bar.empty()  # Clear the progress bar
        st.success("Operation completed successfully!")
        st.balloons()

        st.image("https://i.etsystatic.com/5347321/r/il/1d0755/2228983980/il_600x600.2228983980_ev29.jpg")
    else:
        st.write("Upload file first")

# line chart
# data = pd.DataFrame({
#     "first column": list(range(1,11)),
#     "second column": np.arange(number,number+10)
# })
# st.line_chart(data)