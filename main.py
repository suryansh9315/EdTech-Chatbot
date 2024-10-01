import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title("EdTech QA/Chatbot")
question = st.text_input("Question: ")
btn = st.button("Get Answer")

if btn:
    if question:
        chain = get_qa_chain()
        response = chain.invoke(question)

        st.header("Answer: ")
        st.write(response["result"])



