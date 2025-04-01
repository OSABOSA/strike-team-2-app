import streamlit as st
from src.chain import LlmModule
from src.data.database_ops import PineconeVectorDatabase
from dotenv import load_dotenv, find_dotenv
from src.data.callback_handler import callback_llm_response, pass_placeholder


def main():
    load_dotenv(find_dotenv())

    database = PineconeVectorDatabase()

    llm = LlmModule(progress_callback=callback_llm_response, db_query_callback=database.query_data)  # TODO change nonsense arguments

    st.title("Car review finder")
    st.write("This is a simple app that finds car reviews.")

    query = st.text_input("Ask a question")

    if st.button("Find reviews"):
        llm.chat(query)

    output_placeholder = st.empty()
    pass_placeholder(output_placeholder)


if __name__ == "__main__":
    main()
