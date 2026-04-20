from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    """Format retrieved documents for context."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_text_chunks(text, model_name):
    """Split text into chunks."""
    if model_name == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    chunks = text_splitter.create_documents([text])
    return chunks

def get_vectorstore(text_chunks, model_name, api_key):
    """Create vectorstore from text chunks."""
    if model_name == "Google AI":
        print("Using embedding model: embedding-001")
        embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001", google_api_key=api_key)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(model_name, vectorstore, api_key):
    """Create the RAG conversation chain."""
    if model_name == "Google AI":
        prompt_template = (
            "Use the following pieces of context to answer the question at the end. "
            "If you don't know the answer, say you don't know. {context} Question: {question} Answer:"
        )
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain