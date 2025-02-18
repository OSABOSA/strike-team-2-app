from fastapi import HTTPException
import streamlit as st
from main import analyze_text

if __name__ == "__main__":

    st.title("Sentiment Analysis App")
    st.write("Enter text below to analyze its sentiment:")

    # Text input
    user_input = st.text_area("Enter your text here:")

    if st.button("Analyze"):
        if user_input:
            try:
                result = analyze_text(user_input)
                st.write("Sentiment Analysis Result:", result)
            except HTTPException as e:
                st.error(f"Error: {e.detail}")
        else:
            st.warning("Please enter text before analyzing.")
