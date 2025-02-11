import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the title of the app
st.title("CSV File Plotter")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display the DataFrame
    st.write("### Preview of the uploaded data:")
    st.write(df.head())

    # Get column names
    columns = df.columns.tolist()

    # Select columns for plotting
    st.write("### Select columns for plotting:")
    x_axis = st.selectbox("Select X-axis column:", columns)
    y_axis = st.selectbox("Select Y-axis column:", columns)

    # Select plot type
    st.write("### Select plot type:")
    plot_type = st.selectbox("Choose a plot type:", ["Line Plot", "Bar Plot", "Scatter Plot", "Histogram", "Box Plot"])

    # Generate the plot
    if st.button("Generate Plot"):
        st.write(f"### {plot_type} for {x_axis} vs {y_axis}")

        plt.figure(figsize=(10, 6))

        if plot_type == "Line Plot":
            sns.lineplot(x=df[x_axis], y=df[y_axis])
        elif plot_type == "Bar Plot":
            sns.barplot(x=df[x_axis], y=df[y_axis])
        elif plot_type == "Scatter Plot":
            sns.scatterplot(x=df[x_axis], y=df[y_axis])
        elif plot_type == "Histogram":
            sns.histplot(df[x_axis], kde=True)
        elif plot_type == "Box Plot":
            sns.boxplot(x=df[x_axis], y=df[y_axis])

        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(f"{plot_type} of {x_axis} vs {y_axis}")
        st.pyplot(plt)

else:
    st.write("Please upload a CSV file to get started.")