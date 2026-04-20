import streamlit as st
from datetime import datetime
import pandas as pd
import base64

def load_chat_css():
    """Load CSS for chat messages once to avoid duplication."""
    return """
    <style>
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
        }
        .chat-message.user {
            background-color: #2b313e;
        }
        .chat-message.bot {
            background-color: #475063;
        }
        .chat-message .avatar {
            width: 20%;
        }
        .chat-message .avatar img {
            max-width: 78px;
            max-height: 78px;
            border-radius: 50%;
            object-fit: cover;
        }
        .chat-message .message {
            width: 80%;
            padding: 0 1.5rem;
            color: #fff;
        }
    </style>
    """

def display_chat_message(user_question, response):
    """Display a single chat message pair (user and bot)."""
    html = f"""
    <div class="chat-message user">
        <div class="avatar">
            <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
        </div>
        <div class="message">{user_question}</div>
    </div>
    <div class="chat-message bot">
        <div class="avatar">
            <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
        </div>
        <div class="message">{response}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def display_conversation_history():
    """Display the conversation history."""
    for question, answer, _, timestamp, pdf_name in reversed(st.session_state.conversation_history):
        html = f"""
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
            </div>
            <div class="message">{question}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
            </div>
            <div class="message">{answer}</div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

def download_conversation_history():
    """Provide download link for conversation history."""
    if st.session_state.conversation_history:
        df = pd.DataFrame(
            st.session_state.conversation_history,
            columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"],
        )
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = (
            f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv">'
            f'<button>Download conversation history as CSV file</button></a>'
        )
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.markdown("To download the conversation, click the Download button on the left side at the bottom of the conversation.")

def show_snow():
    """Show snow effect."""
    st.snow()