# streamlit_app.py

import streamlit as st

from lmkr_core import lmkr_answer, history_to_str


st.set_page_config(page_title="LMKR Assistant", page_icon="💬")


def init_state():
    if "history_pairs" not in st.session_state:
        # list of (user, assistant) tuples
        st.session_state.history_pairs = []
    if "messages" not in st.session_state:
        # list of {"role": "user"/"assistant", "content": "..."}
        st.session_state.messages = []


def render_chat():
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


def main():
    init_state()

    st.title("💬LMKR Assistant")
    st.caption(
        "Ask anything about LMKR: company, products, services, projects, contacts, etc."
    )

    # Show previous conversation
    render_chat()

    # Chat input at the bottom
    user_input = st.chat_input("Type your question about LMKR...")
    if user_input:
        # Display user message
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build chat history text from existing pairs
        chat_history_str = history_to_str(st.session_state.history_pairs)

        # Get answer from RAG + HF API
        answer = lmkr_answer(user_input, chat_history_str)

        # Store the pair for future context
        st.session_state.history_pairs.append((user_input, answer))

        # Display assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        with st.chat_message("assistant"):
            st.markdown(answer)


if __name__ == "__main__":
    main()