
import streamlit as st
import os
from pathlib import Path
from typing import List, Dict
import tempfile

from utils.recursive_chunker import RecursiveChunker
from weaviatedb.weaviate_connection import get_weaviate_client
from weaviatedb.weaviate_operation import ingest_chunks_to_weaviate
from weaviatedb.weaviate_retrieval import hybrid_generative_search
from weaviatedb.weaviate_schema import create_schema
from UI.chatbot import display_chat_interface

# Constants
COLLECTION_NAME = "documents"
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt'}

# Initialize session state variables
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'vectorization_complete' not in st.session_state:
    st.session_state.vectorization_complete = False
if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False
if 'weaviate_client' not in st.session_state:
    st.session_state.weaviate_client = None

# Initialize Weaviate manager
@st.cache_resource
def init_weaviate():
    try:
        return get_weaviate_client()
    except Exception as e:
        st.error(f"Failed to connect to Weaviate: {str(e)}")
        return None

# Initialize manager if not already done
if not st.session_state.weaviate_client:
    st.session_state.weaviate_client = init_weaviate()
    if not st.session_state.weaviate_client:
        st.stop()




def validate_file(file) -> bool:
    """Check uploaded file type."""
    file_ext = Path(file.name).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        st.error(f"Unsupported file type. Please upload PDF, DOCX, or TXT files only.")
        return False
    return True

def process_uploaded_file(uploaded_file) -> List[Dict]:
    """Process uploaded file and return chunks."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Only RecursiveChunker is implemented, but method is shown for UI
        chunker = RecursiveChunker(
            max_tokens=1000,
            overlap_tokens=100,
            min_chunk_tokens=200,
            max_table_tokens=600,
            target_utilization=0.85,
            encoding='cl100k_base'
        )
        raw_chunks = chunker.process_file(tmp_path)
        # Format chunks for Weaviate - only include required properties
        formatted_chunks = [
            {
                "content": chunk["text"],
                "fileName": uploaded_file.name,
                "sourceFolder": "uploads",
                "chunkIndex": idx
            } for idx, chunk in enumerate(raw_chunks)
        ]
        st.session_state.raw_chunks = raw_chunks
        return formatted_chunks
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return []
    finally:
        os.unlink(tmp_path)

def display_chunk_analysis(chunks: List[Dict]):
    """Show analysis of processed chunks."""
    if not chunks:
        st.warning("No chunks were generated from the file.")
        return
    
    # Get raw chunks for analysis
    raw_chunks = st.session_state.get('raw_chunks', [])
    
    st.write("### Document Processing Results")
    
    # Main metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Chunks", len(chunks))
        st.metric("Average Size", f"{sum(len(c['content']) for c in chunks) / len(chunks):.0f} chars")
    with col2:
        total_size = sum(len(c['content']) for c in chunks)
        st.metric("Total Content Size", f"{total_size:,} chars")
    
    # Processing details
    with st.expander("View Processing Details"):
        st.write("#### Chunk Distribution")
        sizes = [len(c['content']) for c in chunks]
        st.write(f"- Smallest chunk: {min(sizes):,} characters")
        st.write(f"- Largest chunk: {max(sizes):,} characters")
        st.write(f"- Chunks ready for vectorization: {len(chunks)}")
        
        # Source information
        st.write("\n#### Source Information")
        unique_files = set(c['fileName'] for c in chunks)
        st.write(f"Files processed: {', '.join(unique_files)}")
        
        if raw_chunks:
            st.write("\n#### Chunking Methods Used")
            methods = {}
            for chunk in raw_chunks:
                method = chunk.get('metadata', {}).get('chunk_method', 'standard')
                methods[method] = methods.get(method, 0) + 1
            for method, count in methods.items():
                st.write(f"- {method}: {count} chunks")
    
    st.success("✅ Document successfully processed and ready for vectorization!")
    st.info("ℹ️ Click 'Vectorize in Weaviate' below to continue with vectorization.")

def perform_hybrid_search(query: str) -> str:
    """Run hybrid search and return generative text."""
    try:
        # Use the synchronous client and synchronous retrieval function
        with st.session_state.weaviate_client.get_sync_client() as client:
            response = hybrid_generative_search(
                client,
                query,
                COLLECTION_NAME
            )

            # Safely extract generative text if present
            return getattr(getattr(response, "generative", None), "text", str(response))
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return "I encountered an error while searching. Please try again."

# Main UI
st.title("📚 Document Processing and Chatbot")
st.write("Transform your documents into an interactive knowledge base")

# Add detailed instructions
st.info(
    """
    1. Upload your document (PDF, DOCX, or TXT)
    2. Process the document to create optimized chunks
    3. Vectorize the chunks for semantic search
    4. Start chatting with your document!
    """
)

# File upload section
uploaded_file = st.file_uploader("Choose a file (PDF, DOCX, or TXT)", type=['pdf', 'docx', 'txt'])

if uploaded_file:
    if validate_file(uploaded_file):
        if st.button("🔄 Process File"):
            with st.spinner("Processing file into semantic chunks..."):
                st.info("This may take a moment depending on the file size...")
                chunks = process_uploaded_file(uploaded_file)
                if chunks:
                    st.session_state.chunks = chunks
                    st.session_state.processing_complete = True
                    display_chunk_analysis(chunks)

def handle_vectorization():
    progress_bar = st.progress(0)
    status_area = st.empty()
    def update_progress(progress):
        progress_bar.progress(progress)
        chunks_done = int(progress * len(st.session_state.chunks))
        status_area.info(f"Vectorizing chunks: {chunks_done}/{len(st.session_state.chunks)}")
    with st.spinner("Creating schema..."):
        try:
            with st.session_state.weaviate_client.get_sync_client() as client:
                create_schema(client, COLLECTION_NAME)
                st.success("✅ Schema created/verified successfully")
        except Exception as e:
            st.error(f"Schema creation error: {str(e)}")
            return
    with st.spinner("Vectorizing chunks and preparing for semantic search..."):
        try:
            with st.session_state.weaviate_client.get_sync_client() as client:
                log_box = st.empty()
                logs = []
                def log_cb(msg: str):
                    logs.append(msg)
                    log_box.code("\n".join(logs[-200:]))
                result = ingest_chunks_to_weaviate(
                    client,
                    st.session_state.chunks,
                    COLLECTION_NAME,
                    update_progress,
                    log_callback=log_cb
                )
                st.session_state.vectorization_complete = True
                progress_bar.progress(1.0)
                status_area.empty()
                st.success(f"✅ Vectorization complete! {result.get('successful', 0)} chunks successfully vectorized.")
                if result.get('failed', 0) > 0:
                    st.warning(f"⚠️ {result.get('failed')} chunks failed to vectorize.")
                    failed_objs = result.get('failed_objects', [])
                    if failed_objs:
                        with st.expander("Failed objects details"):
                            for i, fo in enumerate(failed_objs, 1):
                                try:
                                    st.write(f"{i}. {getattr(fo, 'message', str(fo))}")
                                except Exception:
                                    st.write(str(fo))
                st.info("🤖 Ready to start chatting with your documents!")
        except Exception as e:
            st.error(f"Vectorization error: {str(e)}")


if st.session_state.processing_complete and not st.session_state.vectorization_complete:
    if st.button("🔮 Vectorize in Weaviate"):
        handle_vectorization()

# Chat interface section
if st.session_state.vectorization_complete:
    st.write("---")
    st.write("### Chat with your documents")
    display_chat_interface(perform_hybrid_search)
