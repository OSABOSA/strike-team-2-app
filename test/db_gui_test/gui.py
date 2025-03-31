from fastapi import HTTPException
import streamlit as st
import fitz
import pdf_extract as pdf
import chroma_manager as cm
import io
import requests
from pdf_extract import fetching_from_url
from llm_tackler import ask_local_llm

if __name__ == "__main__":
    st.title("Database Test App")
    st.write("Choose a method to provide a PDF file:")

    # Toggle switch for file upload or URL input
    use_url = st.toggle("Use URL instead of file upload")

    file = None
    pdf_url = ""

    if use_url:
        pdf_url = st.text_input("Enter PDF URL:")
    else:
        file = st.file_uploader("Pick a file", type=("pdf", "docx"), accept_multiple_files=False)

    # Store in Chroma
    if st.button("Store in Chroma"):
        if file or pdf_url:
            try:
                if file:
                    st.success("PDF uploaded successfully!")
                    pdf_bytes = file.getvalue()
                    text_chunks = pdf.extract_text_from_pdf(io.BytesIO(pdf_bytes))
                elif pdf_url:
                    st.success("Fetching PDF from URL...")
                    text_chunks, is_status_ok = fetching_from_url(pdf_url)
                    if not is_status_ok:
                        st.error("Failed to fetch PDF from the URL.")

                cm.store_in_chroma(text_chunks)
                st.write("Stored in Chroma successfully!")
            except HTTPException as e:
                st.error(f"Error: {e.detail}")
        else:
            st.warning("Please provide a PDF file or enter a URL before storing in Chroma.")
    user_input = st.text_area("Enter your question here:")

    # Add after user_input in your Streamlit app
    if st.button("Ask AI"):
        # if there was no pdf provided leave a message
        if not file and not pdf_url:
            st.warning("Please provide a PDF file or enter a URL before asking the AI, so it can give you a proper answer.")

        if user_input:
            response = ask_local_llm(user_input)
            st.write("AI Response:", response)
        else:
            st.warning("Please enter a question.")
