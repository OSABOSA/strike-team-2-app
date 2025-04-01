import streamlit as st

from src.models import LocalModels


def page_text_to_image(client):
    st.title("Text to Image")
    st.write("This page is for converting text to images.")
    text = st.text_area("Enter text to convert to an image")
    if st.button("Convert Text to Image"):
        image = client.text_to_image(text)
        st.success("Text converted to image successfully.")
        st.image(image, caption="Generated Image", use_column_width=True)


def page_sentiment_analysis(client):
    st.title("Sentiment Analysis")
    st.write("This page is for performing sentiment analysis.")
    text = st.text_area("Enter text for sentiment analysis")
    if st.button("Analyze Sentiment"):
        sentiment = client.sentiment_analysis(text)
        st.success(f"Sentiment: {sentiment}")


def page_image_to_text(client):
    st.title("Image to Text")
    st.write("This page is for converting images to text.")
    st.write("Upload an image and click the button to convert it to text.")
    is_url = st.toggle("Pass URL instead of uploading image", False)
    if is_url:
        url = st.text_input("Enter image URL")
        if st.button("Convert Image to Text"):
            text = client.image_to_text(url)
            st.success("Image converted to text successfully.")
            st.write("Text:")
            st.write(text)
        return
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        if st.button("Convert Image to Text"):
            text = client.image_to_text(uploaded_file)
            st.success("Image converted to text successfully.")
            st.write("Text:")
            st.write(text)


def page_llm(client):
    st.title("Language Model")
    st.write("This page is for querying the language model.")
    question = st.text_input("Enter a question")
    if st.button("Query Model"):
        # response = client.llm(question)
        response = LocalModels.ask_local_llm(question)
        st.success("Model queried successfully.")
        st.write("Response:")
        st.write(response)
