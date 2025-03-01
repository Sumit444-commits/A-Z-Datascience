import os
import streamlit as st
from langchain_deepseek import ChatDeepSeek

# Set the API Key securely using environment variable or Streamlit input
api_key_name = "DEEPSEEK_API_KEY"
if not os.getenv(api_key_name):
    api_key = st.sidebar.text_input("Enter your DeepSeek API key", type="password")
    if api_key:
        os.environ[api_key_name] = api_key

# Initialize DeepSeek LLM
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=1000,  # You can adjust this based on your needs
    timeout=60,
    max_retries=2,
)

# Function to handle chatbot interaction
def chat_with_bot(user_input):
    messages = [
        ("system", "You are a helpful assistant that answers questions."),
        ("human", user_input)
    ]
    response = llm.invoke(messages)
    return response.content

# Streamlit UI for Chatbot
st.title("DeepSeek Chatbot")
st.write("Ask me anything!")

# Collect user input
user_input = st.text_input("Your question:")

if user_input:
    # Get the bot's response
    response = chat_with_bot(user_input)
    
    # Display the response
    st.write(f"Bot: {response}")

# Optional: To keep track of the conversation history, use a session state (good for context in conversation).
if 'history' not in st.session_state:
    st.session_state.history = []

if user_input:
    # Append user input and bot response to the conversation history
    st.session_state.history.append(f"You: {user_input}")
    st.session_state.history.append(f"Bot: {response}")

# Display conversation history
for message in st.session_state.history:
    st.write(message)
