import streamlit as st
import inspect
import asyncio

def initialize_chat_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

def display_chat_interface(hybrid_search_callback):
    """Chat UI for document Q&A."""
    initialize_chat_state()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask me anything about the documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if inspect.iscoroutinefunction(hybrid_search_callback):
                        response_obj = asyncio.run(hybrid_search_callback(prompt))
                    else:
                        response_obj = hybrid_search_callback(prompt)

                    response_text = getattr(getattr(response_obj, 'generative', None), 'text', str(response_obj))

                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.session_state.conversation_history.append({"role": "assistant", "content": response_text})
                    st.markdown(response_text)
                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    st.markdown(error_message)
                    