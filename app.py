import streamlit as st
import ollama
import faiss
import numpy as np
import time
from langchain.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import re
import os
import tempfile

# --- Must be the first Streamlit command ---
st.set_page_config(page_title="AI Document Chatbot", page_icon="📄", layout="wide")

# --- Clean, Minimalist CSS ---
st.markdown("""
    <style>
    /* Clean, minimalist styling */
    .main { 
        padding: 20px; 
        border-radius: 10px; 
        min-height: 100vh; 
    }
    
    /* Button styling */
    .stButton>button { 
        background-color: #3498db; 
        color: white; 
        border-radius: 8px; 
        padding: 10px 20px; 
        font-weight: bold; 
        transition: background-color 0.3s; 
        border: none;
    }
    .stButton>button:hover { 
        background-color: #2980b9; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Input field styling */
    .stTextInput>div>input { 
        border-radius: 8px; 
        border: 1px solid #ddd; 
        padding: 12px; 
        font-size: 16px; 
        transition: border 0.3s;
    }
    .stTextInput>div>input:focus { 
        border: 1px solid #3498db;
        box-shadow: 0 0 0 2px rgba(52,152,219,0.2);
    }
    
    /* Title and headers */
    .title { 
        font-size: 2.5em; 
        text-align: center; 
        margin-bottom: 25px; 
        font-weight: 600;
        color: inherit;
    }
    .subheader { 
        font-size: 1.3em; 
        margin-top: 20px; 
        font-weight: 500;
    }
    
    /* Answer box */
    .answer-box { 
        background-color: #f8f9fa; 
        padding: 15px; 
        border-radius: 8px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
        margin-top: 10px; 
        border-left: 3px solid #3498db;
    }
    
    /* Error box */
    .error-box { 
        background-color: #fdecea; 
        color: #d32f2f; 
        padding: 12px; 
        border-radius: 8px; 
        border-left: 3px solid #d32f2f; 
        margin: 10px 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content { 
        padding: 20px; 
        border-radius: 8px; 
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] div div:first-child { 
        border: 1px dashed #ddd !important;
        border-radius: 8px !important;
    }
    
    /* Hide the sidebar toggle button */
    [data-testid="stSidebarNav"] > button { display: none !important; }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .title { font-size: 2em; }
        .subheader { font-size: 1.1em; }
    }
    </style>
""", unsafe_allow_html=True)

# --- Functions ---
def load_documents(file=None, url=None):
    if url:
        try:
            loader = WebBaseLoader(url)
            return loader.load()
        except Exception as e:
            raise ValueError(f"Error loading document from URL: {e}")
    elif file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        try:
            if file.name.lower().endswith(".pdf"):
                return PyPDFLoader(temp_file_path).load()
            elif file.name.lower().endswith(".txt"):
                try:
                    return TextLoader(temp_file_path, encoding='utf-8').load()
                except UnicodeDecodeError:
                    return TextLoader(temp_file_path, encoding='latin-1').load()
            else:
                raise ValueError("Unsupported file type. Please upload a PDF or TXT file.")
        finally:
            os.unlink(temp_file_path)
    else:
        raise ValueError("Please provide a file or URL.")

def split_documents(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(pages)

def create_vector_store(split_docs):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    document_texts = [doc.page_content for doc in split_docs]
    embeddings = embedder.encode(document_texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    return index, document_texts, embedder

def retrieve_context(query, embedder, index, documents, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding.astype(np.float32), k)
    return [documents[i] for i in indices[0]]

def clean_answer(text, show_think=False):
    if show_think:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def generate_answer_with_ollama(query, context):
    formatted_context = "\n".join(context)
    prompt = f"""You are an expert assistant trained on document information.
    Use this context to answer the question:

    {formatted_context}

    Question: {query}

    Answer in detail using only the provided context:"""
    response = ollama.generate(
        model='deepseek-r1:1.5b',
        prompt=prompt,
        options={'temperature': 0.3, 'max_tokens': 2000}
    )
    return response['response']

def typing_effect(text, output_area, show_think=False, delay=0.02):
    cleaned_text = clean_answer(text, show_think)
    typed_text = ""
    for char in cleaned_text:
        typed_text += char
        output_area.markdown(f'<div class="answer-box">{typed_text}</div>', unsafe_allow_html=True)
        time.sleep(delay)
    output_area.markdown(f'<div class="answer-box">{typed_text}</div>', unsafe_allow_html=True)

# --- Main App ---
st.markdown('<h1 class="title">📄 AI Document Chatbot</h1>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Options")
    input_method = st.radio("Input Method:", ("Upload File", "Enter URL"))
    show_think = st.checkbox("Show reasoning process")

# Document Input Section
if input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload PDF or TXT document", type=["pdf", "txt"])
    doc_url = None
else:
    doc_url = st.text_input("Enter document URL", placeholder="https://example.com/document.pdf")
    uploaded_file = None

# Process document when available
if uploaded_file or doc_url:
    try:
        with st.spinner("Processing document..."):
            pages = load_documents(file=uploaded_file, url=doc_url)
            split_docs = split_documents(pages)
            index, document_texts, embedder = create_vector_store(split_docs)
            st.session_state["index"] = index
            st.session_state["documents"] = document_texts
            st.session_state["embedder"] = embedder
        st.success("Document processed successfully!")
    except Exception as e:
        st.markdown(f'<div class="error-box">Error: {str(e)}</div>', unsafe_allow_html=True)

# Chat Interface
st.markdown('<h2 class="subheader">💬 Ask about the document</h2>', unsafe_allow_html=True)

with st.form(key='query_form'):
    query = st.text_input("Your question:", placeholder="Type your question here...", label_visibility="collapsed")
    submit_button = st.form_submit_button(label="Get Answer")

if submit_button:
    if not query:
        st.markdown('<div class="error-box">Please enter a question.</div>', unsafe_allow_html=True)
    elif "index" not in st.session_state:
        st.markdown('<div class="error-box">Please upload a document or provide a URL first.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Searching for answers..."):
            context = retrieve_context(query, st.session_state["embedder"], st.session_state["index"], st.session_state["documents"])
            full_answer = generate_answer_with_ollama(query, context)
        
        st.markdown('<h3 class="subheader">💡 Answer</h3>', unsafe_allow_html=True)
        answer_output_area = st.empty()
        typing_effect(full_answer, answer_output_area, show_think=show_think)

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 40px; color: #666; font-size: 0.9em;">
        Made with ❤️ using Streamlit & Ollama
    </div>
""", unsafe_allow_html=True)