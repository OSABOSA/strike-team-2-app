from enum import Enum
import streamlit as st

full_llm_response = ""
output_placeholder: st.empty = None


class CallbackType(Enum):
    INIT = 1
    STATUS = 2
    DELTA = 3
    RESPONSE = 4
    DELETE = 5


def pass_placeholder(placeholder):
    global output_placeholder
    output_placeholder = placeholder


def callback_llm_response(response_type, response):
    if response_type == CallbackType.INIT:
        st.write(response)

    if response_type == CallbackType.STATUS:
        st.write(f'Searching in database for "{response}"...\n')

    if response_type == CallbackType.DELTA:
        global full_llm_response
        full_llm_response += response
        output_placeholder.write(full_llm_response)

    if response_type == CallbackType.DELETE:
        full_llm_response = ""

