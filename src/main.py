import streamlit as st
from src.chain import LlmModule
from src.data.database_ops import PineconeVectorDatabase
from dotenv import load_dotenv, find_dotenv
from src.data.callback_handler import callback_llm_response

st.set_page_config(page_title="Car review finder", page_icon="🚗")

if "llm" not in st.session_state:
    database = PineconeVectorDatabase()
    st.session_state.llm = LlmModule(
        progress_callback=callback_llm_response, db_query_callback=database.query_data
    )

col1, col2 = st.columns([4, 1])  # Adjust ratio for spacing

with col1:
    st.title("Car review finder")
    st.write("This is a simple app that finds car reviews.")

with col2:
    if st.button("Clear conversation"):
        st.session_state.llm.reset_messages()  # Clears chat history
        st.rerun()  # Refresh to reflect changes

for message in st.session_state.llm.get_messages():
    try:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    except KeyError:
        continue

user_input = st.chat_input("Type your message...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response with streaming
    with st.chat_message("assistant"):
        st.session_state.response_container = st.empty()
        st.session_state.current_response = ""  # Reset response storage
        st.session_state.llm.chat(user_input)  # Streaming via callback
        #print(st.session_state.llm.get_messages())
        #print("\n\n")
