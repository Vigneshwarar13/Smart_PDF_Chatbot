# import packages
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64

# import for langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from datetime import datetime 

# format docs for context
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# to get text from pdf :
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# to get chunks from text :
def get_text_chunks(text, modal_name):
    if modal_name == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    else:
        raise ValueError(f"Unsupported model: {modal_name}")

    chunks = text_splitter.create_documents([text])
    return chunks

import google.generativeai as genai

# embedding this chunks and storing in vector database :
def get_vectorstore(text_chunks, modal_name, api_key=None):
    if api_key is None or not api_key.strip():
        raise ValueError("API key is missing or empty. Please provide a valid Google API key.")
    
    if not text_chunks:
        raise ValueError("No text chunks to embed.")
    
    if modal_name == "Google AI":
        # Verified working models from diagnostics
        model_options = [
            "models/gemini-embedding-001",
            "gemini-embedding-001",
            "text-embedding-004",         
            "models/text-embedding-004",  
            "embedding-001",              
            "models/embedding-001"        
        ]
        
        last_exception = None
        for model_id in model_options:
            try:
                print(f"DEBUG: Attempting LangChain embedding with model: {model_id}")
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=model_id, 
                    google_api_key=api_key
                )
                # Test with a dummy embed to confirm it works
                embeddings.embed_query("health check")
                
                print(f"DEBUG: Successfully initialized LangChain model: {model_id}")
                vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)
                return vectorstore
            except Exception as e:
                print(f"DEBUG: LangChain failed for {model_id}: {str(e)}")
                last_exception = e
                continue
        
        # FINAL FALLBACK: Use the official google-generativeai SDK directly
        print("DEBUG: All LangChain models failed. Attempting direct SDK embedding fallback...")
        try:
            genai.configure(api_key=api_key)
            working_model = None
            for m in genai.list_models():
                if 'embedContent' in m.supported_generation_methods:
                    try:
                        genai.embed_content(model=m.name, content="test", task_type="retrieval_document")
                        working_model = m.name
                        print(f"DEBUG: Found working model via direct SDK: {working_model}")
                        break
                    except:
                        continue
            
            if working_model:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=working_model, 
                    google_api_key=api_key
                )
                vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)
                return vectorstore
        except Exception as sdk_e:
            print(f"DEBUG: Direct SDK fallback failed: {str(sdk_e)}")

        error_msg = str(last_exception) if last_exception else "Unknown error"
        print(f"Vectorstore creation error: {error_msg}")
        raise ValueError(f"Failed to create vectorstore with any available model: {error_msg}")
    else:
        raise ValueError(f"Unsupported model: {modal_name}")

#create a conversational chain using langchain :
def get_conversation_chain(modal_name, vectorstore=None, api_key=None):
    if modal_name == "Google AI":
        prompt_template = (
            "Use the following pieces of context to answer the question at the end. "
            "If you don't know the answer, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    else:
        raise ValueError(f"Unsupported model: {modal_name}")

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

# take user question as input :
def user_input(user_question, model_name, api_key, pdf_docs):
    if api_key is None or not api_key.strip() or not pdf_docs:
        st.error("❌ Please upload a PDF document and enter a valid Google API key.")
        return None

    try:
        pdf_text = get_pdf_text(pdf_docs)
        if not pdf_text.strip():
            st.error("❌ No text could be extracted from the PDF files. Please check your PDFs.")
            return None
        
        text_chunks = get_text_chunks(pdf_text, model_name)
        if not text_chunks:
            st.error("❌ Could not split text into chunks. Please check your PDF content.")
            return None
        
        vectorstore = get_vectorstore(text_chunks, model_name, api_key)
    except ValueError as e:
        st.error(f"❌ Validation Error: {str(e)}")
        return None
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Processing Error: {error_msg}")
        print(f"Full error traceback: {error_msg}")
        return None

    if model_name == "Google AI":
        try:
            chain = get_conversation_chain(model_name, vectorstore=vectorstore, api_key=api_key)
            response = chain.invoke(user_question)
            user_question_output = user_question
            response_output = response
            pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
            st.session_state.conversation_history.append(
                (
                    user_question_output,
                    response_output,
                    model_name,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ", ".join(pdf_names),
                )
            )
        except Exception as e:
            error_msg = str(e)
            st.error(f"❌ Chat Error: {error_msg}")
            print(f"Chain invocation error: {error_msg}")
            return None
    else:
        st.error("❌ Selected model is not supported.")
        return None

    st.markdown(
        f"""
        <style>
            .chat-message {{
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
            }}
            .chat-message.user {{
                background-color: #2b313e;
            }}
            .chat-message.bot {{
                background-color: #475063;
            }}
            .chat-message .avatar {{
                width: 20%;
            }}
            .chat-message .avatar img {{
                max-width: 78px;
                max-height: 78px;
                border-radius: 50%;
                object-fit: cover;
            }}
            .chat-message .message {{
                width: 80%;
                padding: 0 1.5rem;
                color: #fff;
            }}
            .chat-message .info {{
                font-size: 0.8rem;
                margin-top: 0.5rem;
                color: #ccc;
            }}
        </style>
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
            </div>    
            <div class="message">{user_question_output}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
            </div>
            <div class="message">{response_output}</div>
            </div>
            
        """,
        unsafe_allow_html=True
    )

    for question, answer, _, timestamp, pdf_name in reversed(st.session_state.conversation_history):
        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>    
                <div class="message">{question}</div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
                </div>
                <div class="message">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if st.session_state.conversation_history:
        df = pd.DataFrame(
            st.session_state.conversation_history,
            columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"],
        )
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
        href = (
            f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv">'
            f'<button>Download conversation history as CSV file</button></a>'
        )
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.markdown("To download the conversation, click the Download button on the left side at the bottom of the conversation.")

    st.snow()
    
# main function to run the app :
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs (v1) :books:")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    linkedin_profile_link = "https://www.linkedin.com/in/vigneshwarar13/"
    portfolio_link = "https://vigneshwarar-portfolio.netlify.app/"
    github_profile_link = "https://pictureurselfphotography.netlify.app/"

    st.sidebar.markdown(
        f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_profile_link}) "
        f"[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=react&logoColor=white)]({portfolio_link}) "
        f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github_profile_link})"
    )
    model_name = st.sidebar.radio("Select the Model:", ( "Google AI"))

    api_key = None

    if model_name == "Google AI":
        api_key = st.sidebar.text_input("Enter your Google API Key:")
        st.sidebar.markdown("Click [here](https://ai.google.dev/) to get an API key.")
        
        if not api_key:
            st.sidebar.warning("Please enter your Google API Key to proceed.")
            return

   
    with st.sidebar:
        st.title("Menu:")
        
        col1, col2 = st.columns(2)
        
        reset_button = col2.button("Reset")
        clear_button = col1.button("Rerun")

        if reset_button:
            st.session_state.conversation_history = []  # Clear conversation history
            st.session_state.user_question = None  # Clear user question input 
            
            
            api_key = None  # Reset Google API key
            pdf_docs = None  # Reset PDF document
            
        else:
            if clear_button:
                if 'user_question' in st.session_state:
                    st.warning("The previous query will be discarded.")
                    st.session_state.user_question = ""  # Temizle
                    if len(st.session_state.conversation_history) > 0:
                        st.session_state.conversation_history.pop()  # Son sorguyu kaldır
                else:
                    st.warning("The question in the input will be queried again.")




        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    st.success("Done")
            else:
                st.warning("Please upload PDF files before processing.")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question, model_name, api_key, pdf_docs)
        st.session_state.user_question = ""  # Clear user question input 

if __name__ == "__main__":
    main()