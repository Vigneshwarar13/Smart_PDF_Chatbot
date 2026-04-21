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

import google.generativeai as genai
from langchain_core.documents import Document

def get_vectorstore(text_chunks, model_name, api_key):
    """Create vectorstore from text chunks."""
    if model_name == "Google AI":
        # CRITICAL FIX: The error 404 models/embedding-001 not found for v1beta 
        # suggests we need to be very flexible with model names and SDK versions.
        
        model_options = [
            "models/gemini-embedding-001",  # Verified by user diagnostics
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
            # Find any working embedding model
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
                # Wrap the direct SDK in a compatible LangChain interface
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=working_model, 
                    google_api_key=api_key
                )
                vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)
                return vectorstore
        except Exception as sdk_e:
            print(f"DEBUG: Direct SDK fallback failed: {str(sdk_e)}")

        raise last_exception if last_exception else ValueError("Failed to initialize any embedding model")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def get_conversation_chain(model_name, vectorstore, api_key):
    """Create the RAG conversation chain."""
    if model_name == "Google AI":
        prompt_template = (
            "Use the following pieces of context to answer the question at the end. "
            "If you don't know the answer, say you don't know. {context} Question: {question} Answer:"
        )
        
        chat_model_options = [
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-pro",
            "models/gemini-1.5-flash",
            "models/gemini-pro"
        ]
        
        last_exception = None
        model = None
        
        # 1. Try known stable identifiers
        for chat_model_id in chat_model_options:
            try:
                print(f"DEBUG: Attempting Chat model: {chat_model_id}")
                model = ChatGoogleGenerativeAI(model=chat_model_id, temperature=0.3, google_api_key=api_key)
                # Quick test to see if model is valid
                model.invoke("test")
                print(f"DEBUG: Successfully initialized Chat model: {chat_model_id}")
                break
            except Exception as e:
                print(f"DEBUG: Chat model {chat_model_id} failed: {str(e)}")
                last_exception = e
                model = None
                continue
        
        # 2. FALLBACK: Auto-discover any generation model via direct SDK
        if model is None:
            print("DEBUG: All predefined Chat models failed. Attempting auto-discovery...")
            try:
                genai.configure(api_key=api_key)
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        # Skip experimental or very specific models if possible, or just try them
                        try:
                            print(f"DEBUG: Testing discovered Chat model: {m.name}")
                            temp_model = ChatGoogleGenerativeAI(model=m.name, temperature=0.3, google_api_key=api_key)
                            temp_model.invoke("test")
                            model = temp_model
                            print(f"DEBUG: Found working Chat model via discovery: {m.name}")
                            break
                        except:
                            continue
            except Exception as sdk_e:
                print(f"DEBUG: SDK discovery for Chat failed: {str(sdk_e)}")
        
        if model is None:
            raise last_exception if last_exception else ValueError("Failed to initialize any chat model")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Use LCEL (LangChain Expression Language) for a robust chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain