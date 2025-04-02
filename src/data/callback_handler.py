from enum import Enum
import streamlit as st


class CallbackType(Enum):
    INIT = 1
    STATUS = 2
    DELTA = 3
    RESPONSE = 4


def callback_llm_response(response_type, response):
    if response_type == CallbackType.STATUS:
        st.session_state.response_container.write(f'Searching in database for "{response}"...\n')

    if response_type == CallbackType.DELTA:
        if "current_response" not in st.session_state:
            st.session_state.current_response = ""
        st.session_state.current_response += response  # Append new delta to stored response
        st.session_state.response_container.markdown(st.session_state.current_response, unsafe_allow_html=True)

