import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot With GROQ"

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, api_key, engine, temperature, max_tokens):
    os.environ["GROQ_API_KEY"] = api_key
    
    try:
        llm = ChatGroq(model=engine)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({'question': question})
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

## Title of the app with Banner Image
if os.path.exists("img.png"):
    st.image("img.png", use_container_width=True)
st.title("Enhanced Q&A Chatbot With Groq")

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

## Select the Groq model
available_models = [
    "llama-3.3-70b", 
    "llama-3.1-8b-instant", 
    "mixtral-8x7b-32768", 
    "gemma2-9b-it", 
    "whisper-large-v3"
]
engine = st.sidebar.selectbox("Select Groq model", available_models)

## Adjust response parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## Main interface for user input
st.write("Ask Me Any Questions")
user_input = st.text_input("You:")

if user_input and api_key:
    response = generate_response(user_input, api_key, engine, temperature, max_tokens)
    st.write(response)
elif user_input:
    st.warning("Please enter the Groq API Key in the sidebar.")
else:
    st.write("Please provide user input.")
