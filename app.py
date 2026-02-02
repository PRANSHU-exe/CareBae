import os
import json
import time
import uuid
import tempfile
import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, firestore
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from pathlib import Path
from PyPDF2 import PdfReader
import PyPDF2

from sentence_transformers import SentenceTransformer
import pinecone
import numpy as np
from groq import Groq

# ============================================
# CONFIGURATION & SECURITY VALIDATION
# ============================================

def validate_required_secrets():
    """Validate all required secrets are present"""
    required_secrets = [
        ("FIREBASE_CREDENTIALS_JSON", "Firebase credentials"),
        ("PINECONE_API_KEY", "Pinecone API key"),
        ("GROQ_API_KEY", "Groq API key"),
        ("PINECONE_ENVIRONMENT", "Pinecone environment"),
    ]
    
    missing_secrets = []
    for secret_key, description in required_secrets:
        if secret_key not in st.secrets:
            missing_secrets.append(f"{description} ({secret_key})")
    
    return missing_secrets

# Validate secrets before proceeding
missing_secrets = validate_required_secrets()
if missing_secrets:
    st.error(f"❌ Missing required configuration: {', '.join(missing_secrets)}")
    st.stop()

# Load configuration from secrets
FIREBASE_CREDENTIALS_JSON = json.loads(st.secrets["FIREBASE_CREDENTIALS_JSON"])
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = st.secrets["PINECONE_ENVIRONMENT"]
PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "carebae-docs")
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
PDF_PATH = st.secrets.get("PDF_PATH", "data/pdfs")
MODEL_NAME = st.secrets.get("MODEL_NAME", "llama-3.1-8b-instant")

# Optional admin emails from secrets (empty list if not provided)
ADMIN_EMAILS = st.secrets.get("ADMIN_EMAILS", "").split(",") if "ADMIN_EMAILS" in st.secrets else []

# Initialize Firebase
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS_JSON)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")
        st.stop()

db = firestore.client()

# ============================================
# FILE VALIDATION & RATE LIMITING
# ============================================

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_FILE_TYPES = ["application/pdf"]

def validate_uploaded_file(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded PDF file for size and type"""
    if uploaded_file.type not in ALLOWED_FILE_TYPES:
        return False, "Only PDF files are allowed"
    
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"File size exceeds {MAX_FILE_SIZE/1024/1024}MB limit"
    
    return True, ""

# Simple rate limiting dictionary
if "user_requests" not in st.session_state:
    st.session_state.user_requests = {}

def check_rate_limit(user_id: str, max_requests: int = 10, time_window: int = 60) -> bool:
    """Check if user has exceeded rate limit"""
    now = time.time()
    
    if user_id not in st.session_state.user_requests:
        st.session_state.user_requests[user_id] = []
    
    # Remove old requests
    user_requests = [req_time for req_time in st.session_state.user_requests[user_id] 
                     if now - req_time < time_window]
    st.session_state.user_requests[user_id] = user_requests
    
    if len(user_requests) >= max_requests:
        return False
    
    user_requests.append(now)
    return True

# ============================================
# PINECONE INITIALIZATION
# ============================================

@st.cache_resource
def init_pinecone():
    """Initialize Pinecone client with v3+ syntax"""
    try:
        # Initialize Pinecone
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        existing_indexes = pc.list_indexes()
        index_names = [idx.name for idx in existing_indexes] if hasattr(existing_indexes, '__iter__') else []
        
        if PINECONE_INDEX_NAME not in index_names:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384, 
                metric="cosine",
                spec=pinecone.ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            st.sidebar.info(f"Created new Pinecone index: {PINECONE_INDEX_NAME}")
            time.sleep(30)
        
        # Connect to the index
        index = pc.Index(PINECONE_INDEX_NAME)
        return pc, index
    except Exception as e:
        st.sidebar.error(f"Failed to initialize Pinecone: {e}")
        return None, None

pc, index = init_pinecone()

# ============================================
# EMBEDDING MODEL
# ============================================

@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model for embeddings"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')  
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

embedding_model = load_embedding_model()

def get_embedding(text: str) -> List[float]:
    """Generate embedding for text using sentence transformer or fallback"""
    if embedding_model:
        return embedding_model.encode(text).tolist()
    else:
        return simple_embed_fallback(text)

def simple_embed_fallback(text: str) -> List[float]:
    """Simple deterministic embedding fallback"""
    import hashlib
    hash_int = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    np.random.seed(hash_int)
    vec = np.random.randn(384).tolist()
    norm = np.linalg.norm(vec)
    return [v / norm for v in vec] if norm > 0 else vec

# ============================================
# PDF PROCESSING FUNCTIONS
# ============================================

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF {pdf_path}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """Split text into overlapping chunks with metadata"""
    if not text:
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "text": chunk_text,
            "start_word": i,
            "end_word": min(i + chunk_size, len(words))
        })
        
        if i + chunk_size >= len(words):
            break
    
    return chunks

def process_pdf_file(pdf_path: Path, uploaded_by: str = "admin") -> List[Dict]:
    """Process a single PDF file into chunks with embeddings"""
    text = extract_text_from_pdf(pdf_path)
    if not text:
        st.warning(f"No text extracted from {pdf_path.name}")
        return []
    
    chunks = chunk_text(text)
    
    if not chunks:
        st.warning(f"No chunks created from {pdf_path.name}")
        return []
    
    processed_chunks = []
    for idx, chunk_data in enumerate(chunks):
        chunk_id = f"{pdf_path.stem}_{idx}_{uuid.uuid4().hex[:8]}"
        embedding = get_embedding(chunk_data["text"])
        
        processed_chunks.append({
            "id": chunk_id,
            "text": chunk_data["text"],
            "embedding": embedding,
            "metadata": {
                "source": pdf_path.name,
                "uploaded_by": uploaded_by,
                "uploaded_at": datetime.utcnow().isoformat(),
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "file_size": os.path.getsize(pdf_path),
                "original_filename": pdf_path.name
            }
        })
    
    return processed_chunks

def store_chunks_in_pinecone(chunks: List[Dict], namespace: str = "pdf_docs"):
    """Store chunks in Pinecone vector database"""
    if index is None:
        st.error("Pinecone index not initialized")
        return False
    
    try:
        if not chunks:
            st.warning("No chunks to store")
            return False
        
        vectors = []
        for chunk in chunks:
            vectors.append({
                "id": chunk["id"],
                "values": chunk["embedding"],
                "metadata": chunk["metadata"]
            })
        
        # Upsert in batches
        batch_size = 100
        successful_chunks = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
            successful_chunks += len(batch)
        
        # Also store in Firestore for backup and text retrieval
        for chunk in chunks:
            doc_ref = db.collection("pdf_chunks").document(chunk["id"])
            doc_ref.set({
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "namespace": namespace,
                "stored_at": firestore.SERVER_TIMESTAMP
            })
        
        # Log the upload in Firestore
        if chunks:
            first_chunk = chunks[0]
            upload_log_ref = db.collection("pdf_uploads").document()
            upload_log_ref.set({
                "filename": first_chunk["metadata"]["source"],
                "uploaded_by": first_chunk["metadata"]["uploaded_by"],
                "uploaded_at": firestore.SERVER_TIMESTAMP,
                "chunk_count": len(chunks),
                "namespace": namespace,
                "status": "success"
            })
        
        return True
    except Exception as e:
        st.error(f"Error storing chunks in Pinecone: {e}")
        
        try:
            error_log_ref = db.collection("pdf_uploads").document()
            error_log_ref.set({
                "error": str(e),
                "uploaded_at": firestore.SERVER_TIMESTAMP,
                "status": "failed"
            })
        except:
            pass
        
        return False

# ============================================
# VECTOR SEARCH FUNCTIONS
# ============================================

def query_pinecone(query_text: str, namespace: str = "pdf_docs", top_k: int = 5) -> List[Dict]:
    """Query Pinecone for similar documents"""
    if index is None:
        return []
    
    try:
        query_embedding = get_embedding(query_text)
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        
        retrieved_chunks = []
        for match in results.matches:
            chunk_id = match.id
            doc_ref = db.collection("pdf_chunks").document(chunk_id)
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                retrieved_chunks.append({
                    "text": data["text"],
                    "metadata": match.metadata,
                    "score": match.score
                })
        
        return retrieved_chunks
    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
        return []

def query_user_conversations(user_id: str, query_text: str, top_k: int = 3) -> List[Dict]:
    """Query user's past conversations for relevant context"""
    if index is None:
        return []
    
    try:
        query_embedding = get_embedding(query_text)
        
        namespace = f"user_{user_id}"
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        
        user_contexts = []
        for match in results.matches:
            if match.metadata:
                user_contexts.append({
                    "question": match.metadata.get("question", ""),
                    "answer": match.metadata.get("answer", ""),
                    "timestamp": match.metadata.get("timestamp", ""),
                    "score": match.score
                })
        
        return user_contexts
    except Exception as e:
        st.error(f"Error querying user conversations: {e}")
        return []

def store_user_conversation(user_id: str, question: str, answer: str):
    """Store user conversation in Pinecone for future reference"""
    if index is None:
        return
    
    try:
        conversation_text = f"Q: {question}\nA: {answer}"
        embedding = get_embedding(conversation_text)
        
        conversation_id = f"conv_{uuid.uuid4().hex}"
        metadata = {
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        index.upsert(
            vectors=[{
                "id": conversation_id,
                "values": embedding,
                "metadata": metadata
            }],
            namespace=f"user_{user_id}"
        )
        
        db.collection("user_conversations").document(user_id).collection("chats").document(conversation_id).set({
            "question": question,
            "answer": answer,
            "embedding": embedding,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        
    except Exception as e:
        st.error(f"Error storing user conversation: {e}")

# ============================================
# ADMIN FUNCTIONS
# ============================================

def process_uploaded_pdfs(pdf_dir: str = None, uploaded_files: List = None, user_id: str = "admin"):
    """Process and store PDFs in vector database"""
    total_chunks = 0
    
    if pdf_dir and os.path.exists(pdf_dir):
        pdf_path = Path(pdf_dir)
        pdf_files = list(pdf_path.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            with st.spinner(f"Processing {pdf_file.name}..."):
                chunks = process_pdf_file(pdf_file, uploaded_by=user_id)
                if chunks:
                    success = store_chunks_in_pinecone(chunks)
                    if success:
                        total_chunks += len(chunks)
                        st.success(f"✅ Processed {pdf_file.name}: {len(chunks)} chunks")
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Validate file before processing
            is_valid, error_msg = validate_uploaded_file(uploaded_file)
            if not is_valid:
                st.error(f"Invalid file {uploaded_file.name}: {error_msg}")
                continue
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name
            
            try:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    chunks = process_pdf_file(Path(temp_path), uploaded_by=user_id)
                    if chunks:
                        success = store_chunks_in_pinecone(chunks)
                        if success:
                            total_chunks += len(chunks)
                            st.success(f"✅ Processed {uploaded_file.name}: {len(chunks)} chunks")
                        else:
                            st.error(f"❌ Failed to store chunks for {uploaded_file.name}")
                    else:
                        st.warning(f"⚠️ No chunks created for {uploaded_file.name}")
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    return total_chunks

def get_uploaded_pdfs_stats():
    """Get statistics about uploaded PDFs"""
    try:
        uploads_ref = db.collection("pdf_uploads").order_by("uploaded_at", direction=firestore.Query.DESCENDING).limit(10)
        uploads = list(uploads_ref.stream())
        
        total_vectors = 0
        if index is not None:
            try:
                stats = index.describe_index_stats()
                total_vectors = stats.total_vector_count
            except:
                total_vectors = 0
        
        return {
            "recent_uploads": [upload.to_dict() for upload in uploads],
            "total_vectors": total_vectors
        }
    except Exception as e:
        st.error(f"Error getting stats: {e}")
        return {"recent_uploads": [], "total_vectors": 0}

def display_admin_panel(user_id: str):
    """Display the admin panel for PDF management"""
    st.header("📁 Admin Panel - PDF Management")
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload PDFs", "📊 View Statistics", "⚙️ Settings"])
    
    with tab1:
        st.subheader("Upload PDF Documents")
        st.markdown("""
        Upload PDF documents to add them to the knowledge base. These documents will be:
        - Processed and split into chunks
        - Converted to vector embeddings
        - Stored in Pinecone for semantic search
        - Available to all users
        """)
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Select one or more PDF files to upload (max 10MB each)"
        )
        
        if uploaded_files:
            st.info(f"Selected {len(uploaded_files)} PDF file(s)")
            
            with st.expander("📋 Selected Files Preview"):
                for i, file in enumerate(uploaded_files):
                    st.write(f"{i+1}. **{file.name}** ({file.size / 1024:.1f} KB)")
        
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider("Chunk Size (words)", 300, 1000, 500, 50,
                                 help="Number of words per text chunk")
        with col2:
            overlap = st.slider("Chunk Overlap (words)", 50, 300, 100, 10,
                              help="Overlap between chunks for context preservation")
        
        if st.button("🚀 Process and Upload PDFs", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("Please select PDF files to upload first!")
            else:
                # Validate all files first
                invalid_files = []
                for uploaded_file in uploaded_files:
                    is_valid, error_msg = validate_uploaded_file(uploaded_file)
                    if not is_valid:
                        invalid_files.append(f"{uploaded_file.name}: {error_msg}")
                
                if invalid_files:
                    st.error("Some files are invalid:")
                    for error in invalid_files:
                        st.write(f"- {error}")
                else:
                    with st.spinner("Processing PDFs..."):
                        progress_bar = st.progress(0)
                        total_processed = 0
                        
                        for i, uploaded_file in enumerate(uploaded_files):
                            progress = (i) / len(uploaded_files)
                            progress_bar.progress(progress)
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                tmp_file.write(uploaded_file.getbuffer())
                                temp_path = tmp_file.name
                            
                            try:
                                text = extract_text_from_pdf(Path(temp_path))
                                if not text:
                                    st.warning(f"No text extracted from {uploaded_file.name}")
                                    continue
                                
                                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                                
                                if chunks:
                                    processed_chunks = []
                                    for idx, chunk_data in enumerate(chunks):
                                        chunk_id = f"{uploaded_file.name.replace('.pdf', '')}_{idx}_{uuid.uuid4().hex[:8]}"
                                        embedding = get_embedding(chunk_data["text"])
                                        
                                        processed_chunks.append({
                                            "id": chunk_id,
                                            "text": chunk_data["text"],
                                            "embedding": embedding,
                                            "metadata": {
                                                "source": uploaded_file.name,
                                                "uploaded_by": user_id,
                                                "uploaded_at": datetime.utcnow().isoformat(),
                                                "chunk_index": idx,
                                                "total_chunks": len(chunks),
                                                "file_size": uploaded_file.size,
                                                "original_filename": uploaded_file.name
                                            }
                                        })
                                    
                                    if processed_chunks:
                                        success = store_chunks_in_pinecone(processed_chunks)
                                        if success:
                                            total_processed += len(processed_chunks)
                                            st.success(f"✅ {uploaded_file.name}: {len(processed_chunks)} chunks")
                                        else:
                                            st.error(f"❌ Failed to store {uploaded_file.name}")
                                else:
                                    st.warning(f"No chunks created from {uploaded_file.name}")
                            
                            finally:
                                if os.path.exists(temp_path):
                                    os.unlink(temp_path)
                        
                        progress_bar.progress(1.0)
                        
                        if total_processed > 0:
                            st.balloons()
                            st.success(f"🎉 Successfully processed {len(uploaded_files)} PDF(s) into {total_processed} total chunks!")
                        else:
                            st.error("No chunks were processed. Please check your PDF files.")
    
    with tab2:
        st.subheader("Knowledge Base Statistics")
        
        stats = get_uploaded_pdfs_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Vectors", stats["total_vectors"])
        with col2:
            recent_count = len(stats["recent_uploads"])
            st.metric("Recent Uploads", recent_count)
        with col3:
            if stats["recent_uploads"]:
                latest = stats["recent_uploads"][0]
                st.metric("Latest Upload", latest.get("filename", "N/A"))
        
        st.subheader("Recent Uploads")
        if stats["recent_uploads"]:
            upload_data = []
            for upload in stats["recent_uploads"]:
                upload_time = upload.get("uploaded_at")
                if hasattr(upload_time, 'strftime'):
                    upload_time = upload_time.strftime("%Y-%m-%d %H:%M")
                elif isinstance(upload_time, str):
                    try:
                        dt = datetime.fromisoformat(upload_time.replace('Z', '+00:00'))
                        upload_time = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        pass
                
                upload_data.append({
                    "Filename": upload.get("filename", "Unknown"),
                    "Chunks": upload.get("chunk_count", 0),
                    "Uploaded By": upload.get("uploaded_by", "Unknown"),
                    "Time": upload_time,
                    "Status": upload.get("status", "Unknown")
                })
            
            st.dataframe(upload_data, use_container_width=True)
        else:
            st.info("No uploads yet.")
        
        st.subheader("Test Search")
        test_query = st.text_input("Enter a test query to search the knowledge base:")
        if test_query and st.button("Test Search"):
            with st.spinner("Searching..."):
                results = query_pinecone(test_query, top_k=3)
                if results:
                    st.success(f"Found {len(results)} relevant chunks:")
                    for i, result in enumerate(results):
                        with st.expander(f"Result {i+1} (Score: {result['score']:.3f}) - {result['metadata']['source']}"):
                            st.write(f"**Source:** {result['metadata']['source']}")
                            st.write(f"**Text:** {result['text'][:300]}...")
                else:
                    st.warning("No results found.")
    
    with tab3:
        st.subheader("Settings")
        
        st.write("**Pinecone Configuration**")
        st.code(f"""
        Index Name: {PINECONE_INDEX_NAME}
        API Key: {PINECONE_API_KEY[:10]}...{PINECONE_API_KEY[-10:] if len(PINECONE_API_KEY) > 20 else ''}
        Environment: {PINECONE_ENVIRONMENT}
        """)
        
        st.write("**Embedding Model**")
        if embedding_model:
            st.success("✅ SentenceTransformer loaded (all-MiniLM-L6-v2)")
        else:
            st.error("❌ Embedding model not loaded")
        
        st.write("**System Status**")
        if index is not None:
            try:
                index_stats = index.describe_index_stats()
                st.success("✅ Pinecone is connected and ready")
                st.write(f"Total vectors across all namespaces: {index_stats.total_vector_count}")
            except Exception as e:
                st.error(f"❌ Pinecone error: {e}")
        else:
            st.error("❌ Pinecone not initialized")
        
        st.divider()
        st.write("**Danger Zone**")
        if st.button("Clear All Uploads (Development Only)", type="secondary"):
            st.warning("This will clear all uploaded data. Are you sure?")
            confirm = st.checkbox("I understand this will delete all data")
            if confirm and st.button("Confirm Clear All"):
                st.info("Data clearance would be implemented here")

# ============================================
# GROQ LLM INTEGRATION
# ============================================

def call_groq_with_context(system_prompt: str, user_prompt: str, conversation_history: List[Dict] = None) -> str:
    """Call Groq LLM with provided context and conversation history"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        if conversation_history:
            for msg in conversation_history[-6:]:
                role = "user" if msg["role"] == "user" else "assistant"
                messages.append({"role": role, "content": msg["content"]})
        
        messages.append({"role": "user", "content": user_prompt})
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=800,
            top_p=0.9
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error calling Groq API: {e}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again."

# ============================================
# ADMIN SETUP FUNCTION (FROM SECRETS)
# ============================================

def setup_admin_users_from_secrets():
    """Set up admin users from secrets file (only runs once)"""
    if "admin_setup_done" in st.session_state:
        return
    
    if ADMIN_EMAILS:
        try:
            for email in ADMIN_EMAILS:
                email = email.strip()
                if email:
                    try:
                        user = auth.get_user_by_email(email)
                        db.collection("users").document(user.uid).update({"is_admin": True})
                        print(f"Admin user setup: {email}")
                    except auth.UserNotFoundError:
                        print(f"User not found: {email}")
                    except Exception as e:
                        print(f"Error setting up admin for {email}: {e}")
        except Exception as e:
            print(f"Error in admin setup: {e}")
    
    st.session_state.admin_setup_done = True

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    st.set_page_config(
        page_title="CareBae - Women's Health Assistant",
        page_icon="🌸",
        layout="wide"
    )
    
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user" not in st.session_state:
        st.session_state.user = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_admin" not in st.session_state:
        st.session_state.show_admin = False
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = []
    if "admin_setup_done" not in st.session_state:
        st.session_state.admin_setup_done = False
    
    # Set up admin users from secrets (only once)
    if not st.session_state.admin_setup_done:
        setup_admin_users_from_secrets()
    
    # Main header
    st.title("🌸 CareBae - Women's Health Assistant")
    st.markdown("""
    Your safe, private, and knowledgeable companion for menstrual health education.
    
    **⚠️ Important**: This is for educational purposes only. Always consult with 
    healthcare professionals for medical advice.
    """)
    
    # Sidebar for auth and navigation
    with st.sidebar:
        st.header("Account & Navigation")
        
        if not st.session_state.logged_in:
            auth_tab = st.radio("Choose", ["Login", "Sign Up"], horizontal=True)
            
            if auth_tab == "Login":
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                
                if st.button("Login", use_container_width=True):
                    try:
                        user = auth.get_user_by_email(email)
                        user_doc = db.collection("users").document(user.uid).get()
                        
                        if user_doc.exists:
                            st.session_state.logged_in = True
                            st.session_state.user = {
                                "uid": user.uid,
                                "email": email,
                                **user_doc.to_dict()
                            }
                            st.session_state.messages = []
                            st.session_state.show_admin = False
                            st.session_state.conversation_context = []
                            st.success("Logged in successfully!")
                            st.rerun()
                        else:
                            st.error("User profile not found")
                    except Exception as e:
                        st.error(f"Login failed: {e}")
            
            else:
                username = st.text_input("Username")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                
                if st.button("Create Account", use_container_width=True):
                    try:
                        user = auth.create_user(email=email, password=password)
                        
                        # Check if this email is in admin list
                        is_admin_from_secrets = email in ADMIN_EMAILS if ADMIN_EMAILS else False
                        
                        db.collection("users").document(user.uid).set({
                            "username": username,
                            "email": email,
                            "created_at": firestore.SERVER_TIMESTAMP,
                            "is_admin": is_admin_from_secrets
                        })
                        
                        st.session_state.logged_in = True
                        st.session_state.user = {
                            "uid": user.uid,
                            "username": username,
                            "email": email,
                            "is_admin": is_admin_from_secrets
                        }
                        st.session_state.messages = []
                        st.session_state.show_admin = False
                        st.session_state.conversation_context = []
                        st.success("Account created successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Sign up failed: {e}")
        
        else:
            st.success(f"Welcome, {st.session_state.user.get('username', 'User')}!")
            
            st.divider()
            st.subheader("Navigation")
            
            if st.button("💬 Go to Chat", use_container_width=True):
                st.session_state.show_admin = False
                st.rerun()
            
            if st.session_state.user.get("is_admin", False):
                if st.button("📁 Admin Panel", use_container_width=True):
                    st.session_state.show_admin = True
                    st.rerun()
            
            st.divider()
            st.write("**Account Info**")
            st.write(f"Email: {st.session_state.user.get('email')}")
            if st.session_state.user.get("is_admin"):
                st.write("Admin: ✅ Yes")
            
            st.divider()
            if st.button("Clear Conversation Context", use_container_width=True):
                st.session_state.conversation_context = []
                st.success("Conversation context cleared!")
            
            st.divider()
            if st.button("Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.user = None
                st.session_state.messages = []
                st.session_state.show_admin = False
                st.session_state.conversation_context = []
                st.rerun()
    
    # Main content area
    if not st.session_state.logged_in:
        st.info("Please login or sign up to start chatting with CareBae!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📚 Knowledge Base")
            st.write("Access information from verified health resources and uploaded documents")
        
        with col2:
            st.markdown("### 🔒 Privacy First")
            st.write("Your conversations are private and secure")
        
        with col3:
            st.markdown("### 🩺 Safe Information")
            st.write("Educational content only - we encourage doctor consultations for medical concerns")
    
    elif st.session_state.show_admin and st.session_state.user.get("is_admin", False):
        display_admin_panel(st.session_state.user["uid"])
    
    else:
        # Show chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask about women's health, periods, symptoms, or hygiene..."):
            # Check rate limit
            user_id = st.session_state.user["uid"]
            if not check_rate_limit(user_id):
                st.error("Too many requests. Please wait a minute before trying again.")
                return
            
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.conversation_context.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            db.collection("user_messages").document(user_id).collection("messages").add({
                "role": "user",
                "content": prompt,
                "timestamp": firestore.SERVER_TIMESTAMP
            })
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    pdf_contexts = query_pinecone(prompt, top_k=3)
                    pdf_context_text = "\n\n".join([f"Source: {ctx['metadata']['source']}\n{ctx['text']}" 
                                                   for ctx in pdf_contexts]) if pdf_contexts else "No relevant PDF information found."
                    
                    user_contexts = query_user_conversations(user_id, prompt, top_k=2)
                    user_context_text = "\n\n".join([f"Previous Q: {ctx['question']}\nPrevious A: {ctx['answer']}" 
                                                    for ctx in user_contexts]) if user_contexts else "No previous conversation history."
                    
                    current_context_text = ""
                    if len(st.session_state.conversation_context) > 1:
                        recent_messages = st.session_state.conversation_context[-6:]
                        for msg in recent_messages:
                            role_name = "User" if msg["role"] == "user" else "Assistant"
                            current_context_text += f"{role_name}: {msg['content']}\n"
                    
                    system_prompt = f"""You are CareBae, a compassionate and knowledgeable women's health assistant. 
                    
IMPORTANT GUIDELINES:
1. Provide educational information only - never diagnose, prescribe, or give medical advice
2. Always encourage consulting healthcare professionals for serious concerns
3. Be empathetic, non-judgmental, and supportive
4. Use clear, simple language that's easy to understand
5. If you don't know something, admit it and suggest consulting a doctor
6. REMEMBER THE CURRENT CONVERSATION CONTEXT to provide coherent responses

KNOWLEDGE BASE CONTEXT (from uploaded PDFs):
{pdf_context_text}

USER'S PREVIOUS CONVERSATIONS (from memory):
{user_context_text}

CURRENT CONVERSATION CONTEXT (recent messages):
{current_context_text}

Based on the above information and your general knowledge, provide a helpful response that continues the conversation naturally."""
                    
                    response = call_groq_with_context(system_prompt, prompt, st.session_state.conversation_context)
                    
                    store_user_conversation(user_id, prompt, response)
                    
                    st.session_state.conversation_context.append({"role": "assistant", "content": response})
                    st.markdown(response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    db.collection("user_messages").document(user_id).collection("messages").add({
                        "role": "assistant",
                        "content": response,
                        "timestamp": firestore.SERVER_TIMESTAMP
                    })

if __name__ == "__main__":
    main()