import streamlit as st
from chain import LlmModule
from enum import Enum
from data.database_ops import PineconeVectorDatabase


class CallbackType(Enum):
    INIT = 1
    QUERY = 2
    DELTA = 3
    RESPONSE = 4


def callback_llm_response(response_type, response):
    if response_type == CallbackType.DELTA:
        st.write(response)


def main():
    llm = LlmModule(progress_callback=callback_llm_response, db_query_callback=PineconeVectorDatabase.query_data)  # TODO change nonsense arguments
    st.title("Car review finder")
    st.write("This is a simple app that finds car reviews.")
    st.text_input("Ask a question")

    if st.button("Find reviews"):
        llm.chat(st.text_input)

