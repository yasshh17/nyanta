"""Professional RAG Chatbot"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from src.document_loader import load_and_chunk
from src.vector_store import PineconeVectorStore
from src.rag_chain import RAGChain
from src.database import ChatDatabase
import os
import uuid
from datetime import datetime
from pinecone import Pinecone
import time

# Page config
st.set_page_config(
    page_title="Nyanta - AI Knowledge Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Nyanta - Your AI-powered knowledge assistant"
    }
)

# CSS
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #0F0F0F 0%, #1A1A1A 100%);
        padding: 2rem 1rem;
    }
    
    /* Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        color: #9CA3AF;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Statistics cards */
    .stat-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .stat-box:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 8px 16px rgba(99, 102, 241, 0.15);
    }
    
    .stat-box strong {
        color: #E5E7EB;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 0.25rem;
    }
    
    .success-msg {
        color: #34D399;
        font-weight: 600;
    }
    
    .warning-msg {
        color: #FBBF24;
        font-weight: 600;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
        animation: fadeIn 0.3s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Input area */
    .stChatInputContainer {
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding-top: 1.5rem;
        background: rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
        border: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1A1A 0%, #0F0F0F 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* File uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: rgba(99, 102, 241, 0.6);
        background: rgba(99, 102, 241, 0.05);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 100%);
        border-radius: 4px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(99, 102, 241, 0.1);
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Smooth animations */
    * {
        scroll-behavior: smooth;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
@st.cache_resource
def get_database():
    return ChatDatabase()

db = get_database()

# Initialize Pinecone check with loading state
@st.cache_data(ttl=60, show_spinner=False)
def check_pinecone_documents():
    """Check if documents exist in Pinecone."""
    try:
        # Get API key (prioritize .env/os.getenv to avoid Streamlit secrets warning)
        pinecone_key = os.getenv('PINECONE_API_KEY')
        if not pinecone_key:
            try:
                if hasattr(st, 'secrets') and 'PINECONE_API_KEY' in st.secrets:
                    pinecone_key = st.secrets['PINECONE_API_KEY']
            except:
                pass
        
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index("rag-chatbot")
        stats = index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        return vector_count > 0, vector_count
    except Exception as e:
        return False, 0

# Generate or restore session ID
if 'session_id' not in st.session_state:
    recent_session = db.get_most_recent_session()
    if recent_session:
        st.session_state.session_id = recent_session
    else:
        st.session_state.session_id = str(uuid.uuid4())

# Initialize session state
if 'vector_store' not in st.session_state:
    with st.spinner("Initializing AI system..."):
        st.session_state.vector_store = PineconeVectorStore()

if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = RAGChain()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = db.load_messages(st.session_state.session_id)

if 'documents_indexed' not in st.session_state:
    has_docs, _ = check_pinecone_documents()
    st.session_state.documents_indexed = has_docs

if 'processing' not in st.session_state:
    st.session_state.processing = False

# Header with gradient
st.markdown('<h1 class="main-header"> Nyanta </h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your AI-powered knowledge assistant • Ask anything about your documents</p>', 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # Logo/branding
    st.markdown("### 📚 Document Library")
    
    # File upload with better UX
    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        help="Supports PDF, TXT, and DOCX • Max 200MB per file",
        label_visibility="collapsed"
    )
    
    # Process button with loading state
    if uploaded_files:
        process_button = st.button(
            f"📥 Process {len(uploaded_files)} {'file' if len(uploaded_files) == 1 else 'files'}", 
            type="primary",
            disabled=st.session_state.processing,
            use_container_width=True
        )
    else:
        st.button(
            "📥 Upload documents first",
            disabled=True,
            use_container_width=True
        )
        process_button = False
    
    if process_button and uploaded_files:
        st.session_state.processing = True
        
        # Professional progress indicator
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status = st.empty()
            
            try:
                os.makedirs("documents", exist_ok=True)
                all_chunks = []
                total_files = len(uploaded_files)
                
                for idx, file in enumerate(uploaded_files):
                    # Update status with emoji
                    status.markdown(f"⚡ Processing **{file.name}**...")
                    progress_bar.progress((idx) / total_files)
                    
                    file_path = f"documents/{file.name}"
                    with open(file_path, "wb") as f:
                        f.write(file.getvalue())
                    
                    try:
                        chunks = load_and_chunk(file_path)
                        all_chunks.extend(chunks)
                        
                        db.save_document(
                            filename=file.name,
                            file_size=file.size,
                            chunk_count=len(chunks)
                        )
                        
                        st.toast(f"✓ {file.name} processed", icon="✅")
                        
                    except Exception as e:
                        st.toast(f"Error: {file.name} - {str(e)}", icon="❌")
                
                if all_chunks:
                    status.markdown(f"🔮 Creating embeddings for **{len(all_chunks):,} chunks**...")
                    progress_bar.progress(0.9)
                    
                    st.session_state.vector_store.create_index(all_chunks)
                    st.session_state.documents_indexed = True
                    
                    progress_bar.progress(1.0)
                    time.sleep(0.5)
                    
                    status.markdown("")
                    progress_bar.empty()
                    
                    st.success(f"🎉 Indexed {len(all_chunks):,} chunks from {total_files} files!")
                    check_pinecone_documents.clear()
                    
                    st.balloons()
                    
            except Exception as e:
                st.error(f"❌ {str(e)}")
            finally:
                st.session_state.processing = False
    
    st.markdown("---")
    
    # Premium statistics design
    st.markdown("### 📊 Knowledge Base")
    
    has_docs, vector_count = check_pinecone_documents()
    doc_stats = db.get_document_stats()
    
    if has_docs:
        # Status card
        st.markdown(f'''
        <div class="stat-box">
            <strong>System Status</strong>
            <div class="stat-value"><span class="success-msg">● Online</span></div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Metrics grid
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Vectors", f"{vector_count:,}", delta=None)
            st.metric("Documents", doc_stats["total_documents"], delta=None)
        with col2:
            st.metric("Chunks", f"{doc_stats['total_chunks']:,}", delta=None)
            avg_chunks = doc_stats['total_chunks'] // max(doc_stats['total_documents'], 1)
            st.metric("Avg/Doc", avg_chunks, delta=None)
    else:
        st.markdown('''
        <div class="stat-box">
            <strong>System Status</strong>
            <div class="stat-value"><span class="warning-msg">● Waiting</span></div>
        </div>
        ''', unsafe_allow_html=True)
        st.info("🎯 Upload documents to activate AI search")
    
    st.markdown("---")
    
    # Session controls with icons
    st.markdown("### 💭 Conversation")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✨ New Chat", use_container_width=True, help="Start a fresh conversation"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True, help="Clear current conversation"):
            db.clear_session(st.session_state.session_id)
            st.session_state.chat_history = []
            st.rerun()
    
    # Session metadata
    message_count = len(st.session_state.chat_history)
    if message_count > 0:
        st.caption(f"💬 {message_count} messages in this session")
    st.caption(f"🔑 {st.session_state.session_id[:12]}")
    
    st.markdown("---")
    
    # Tech stack
    with st.expander("⚙️ Tech Stack", expanded=False):
        st.markdown("""
        **AI & ML**
        - LLM: Groq Llama 3.3 70B
        - Embeddings: OpenAI text-embedding-3
        - Vector DB: Pinecone Serverless
        
        **Backend**
        - Framework: LangChain
        - Storage: SQLite
        - Language: Python 3.11
        
        **Features**
        - RAG Pipeline
        - Semantic Search
        - Source Citations
        - Session Persistence
        """)

# Main chat interface
st.markdown("### 💬 Chat")

# Empty state if no messages
if not st.session_state.chat_history:
    st.markdown("""
    <div style='text-align: center; padding: 4rem 2rem; color: #6B7280;'>
        <h2 style='font-size: 1.5rem; margin-bottom: 1rem;'>👋 Welcome to Nyanta!</h2>
        <p style='font-size: 1.1rem;'>Upload documents to get started, then ask me anything.</p>
        <p style='margin-top: 1rem; font-size: 0.9rem;'>I'll search your documents and provide answers with citations.</p>
    </div>
    """, unsafe_allow_html=True)

# Display chat history with animations
for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🧠"):
        st.markdown(message["content"])
        
        if "sources" in message and message["sources"]:
            with st.expander(f"📚 {len(message['sources'])} sources", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**{i}. {source['source']}** • Chunk {source['chunk_id']}")
                    with st.container():
                        st.code(source['content'], language=None)
                    if i < len(message["sources"]):
                        st.markdown("---")

# Chat input with better placeholder
if prompt := st.chat_input(
    "Ask anything about your documents..." if st.session_state.documents_indexed else "Upload documents first to start chatting",
    disabled=not st.session_state.documents_indexed,
    key="chat_input"
):
    # Add user message
    user_message = {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.chat_history.append(user_message)
    db.save_message(st.session_state.session_id, "user", prompt)
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Generate response with typing indicator
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Simulate thinking (brief)
                time.sleep(0.3)
                
                # Search
                results = st.session_state.vector_store.search(prompt, k=3)
                
                if not results:
                    response_text = "I couldn't find relevant information in your documents. Try rephrasing or uploading more content."
                    st.markdown(response_text)
                    
                    assistant_message = {
                        "role": "assistant",
                        "content": response_text,
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.chat_history.append(assistant_message)
                    db.save_message(st.session_state.session_id, "assistant", response_text)
                else:
                    # Generate answer
                    response = st.session_state.rag_chain.query(prompt, results)
                    
                    # Display with smooth animation
                    st.markdown(response['answer'])
                    
                    # Sources in professional format
                    if response['sources']:
                        with st.expander(f"📚 {len(response['sources'])} sources", expanded=False):
                            for i, source in enumerate(response['sources'], 1):
                                st.markdown(f"**{i}. {source['source']}** • Chunk {source['chunk_id']}")
                                with st.container():
                                    st.code(source['content'], language=None)
                                if i < len(response['sources']):
                                    st.markdown("---")
                    
                    # Save to history
                    assistant_message = {
                        "role": "assistant",
                        "content": response['answer'],
                        "sources": response['sources'],
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.chat_history.append(assistant_message)
                    db.save_message(
                        st.session_state.session_id, 
                        "assistant", 
                        response['answer'],
                        response['sources']
                    )
                    
            except Exception as e:
                error_msg = f"⚠️ Something went wrong. Please try again.\n\nError: {str(e)}"
                st.error(error_msg)
                
                assistant_message = {
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.chat_history.append(assistant_message)
                db.save_message(st.session_state.session_id, "assistant", error_msg)

# Professional footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 3rem 0 2rem 0;'>
    <p style='
        font-size: 1.1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.75rem;
        letter-spacing: -0.02em;
    '>
        Nyanta
    </p>
    <p style='color: #9CA3AF; font-size: 0.9rem; line-height: 1.5;'>
        Transform documents into conversations
    </p>
</div>
""", unsafe_allow_html=True)
