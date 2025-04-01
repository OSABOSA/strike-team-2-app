from enum import Enum
import streamlit as st


class CallbackType(Enum):
    INIT = 1
    STATUS = 2
    DELTA = 3
    RESPONSE = 4


def callback_llm_response(response_type, response):
    if response_type == CallbackType.RESPONSE or response_type == CallbackType.STATUS:
        st.write(response)
    elif response_type == CallbackType.INIT:
        st.write(f"Searching in database for {response}")


