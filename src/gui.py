import streamlit as st
from streamlit_option_menu import option_menu
from pages import page_image_to_text, page_llm, page_sentiment_analysis, page_text_to_image
from models import HuggingFaceModels


def main():
    client = HuggingFaceModels()
    st.set_page_config(page_title="Multi-Page Streamlit App", page_icon="📘", layout="wide")

    # Sidebar navigation
    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",  # Required
            options=["Image to text", "Text to image", "Chat", "Sentiment Analysis"],  # Required
            icons=["house", "info-circle", "envelope", "phone"],  # Optional
            menu_icon="cast",  # Optional
            default_index=0,  # Optional
        )

    # Display content based on selected page
    if selected == "Image to text":
        page_image_to_text(client)
    elif selected == "Text to image":
        page_text_to_image(client)
    elif selected == "Chat":
        page_llm(client)
    elif selected == "Sentiment Analysis":
        page_sentiment_analysis(client)


if __name__ == "__main__":
    main()
