import streamlit as st
from datetime import datetime
import os

from config.settings import settings
from services.pdf_processor import get_pdf_text
from services.rag_service import get_text_chunks, get_vectorstore, get_conversation_chain
from ui.chat_ui import load_chat_css, display_chat_message, display_conversation_history, download_conversation_history, show_snow
from utils.helpers import validate_api_key, validate_pdf_docs

# Cache the vectorstore to avoid recomputing
@st.cache_data
def process_pdfs(pdf_docs, model_name, api_key):
    """Process PDFs and create vectorstore."""
    validate_api_key(api_key)
    validate_pdf_docs(pdf_docs)
    text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(text, model_name)
    vectorstore = get_vectorstore(text_chunks, model_name, api_key)
    return vectorstore

def handle_user_input(user_question, model_name, api_key, pdf_docs):
    """Handle user question and generate response."""
    try:
        vectorstore = process_pdfs(pdf_docs, model_name, api_key)
        chain = get_conversation_chain(model_name, vectorstore, api_key)
        response = chain.invoke(user_question)

        # Display the message first
        display_chat_message(user_question, response)

        # Add to history after display
        pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
        st.session_state.conversation_history.append(
            (
                user_question,
                response,
                model_name,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ", ".join(pdf_names),
            )
        )

        # Display conversation history (previous messages)
        display_conversation_history()
        download_conversation_history()
        show_snow()

        # Clear the input
        st.session_state.user_question = ""

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    # Load CSS once for chat messages
    st.markdown(load_chat_css(), unsafe_allow_html=True)
    st.header("Chat with multiple PDFs (v1) :books:")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Sidebar links
    linkedin_profile_link = "https://www.linkedin.com/in/vigneshwarar13/"
    portfolio_link = "https://vigneshwarar-portfolio.netlify.app/"
    github_profile_link = "https://github.com/Vigneshwarar13"

    st.sidebar.markdown(
        f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_profile_link}) "
        f"[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=react&logoColor=white)]({portfolio_link}) "
        f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github_profile_link})"
    )

    # Model selection
    model_name = st.sidebar.radio("Select the Model:", ("Google AI",))

    # API key input
    api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")
    if not api_key:
        st.sidebar.warning("Please enter your Google API Key to proceed.")
        return

    # Menu
    with st.sidebar:
        st.title("Menu:")

        col1, col2 = st.columns(2)

        reset_button = col2.button("Reset")
        clear_button = col1.button("Rerun")

        if reset_button:
            st.session_state.conversation_history = []
            st.rerun()

        if clear_button:
            if 'user_question' in st.session_state:
                st.warning("The previous query will be discarded.")
                st.session_state.user_question = ""
                if st.session_state.conversation_history:
                    st.session_state.conversation_history.pop()
            else:
                st.warning("The question in the input will be queried again.")

        # PDF uploader
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    st.success("Done")
            else:
                st.warning("Please upload PDF files before processing.")

    # User question input
    user_question = st.text_input("Ask a Question from the PDF Files", value=st.session_state.get('user_question', ''))

    if user_question and pdf_docs and api_key:
        handle_user_input(user_question, model_name, api_key, pdf_docs)

if __name__ == "__main__":
    main()