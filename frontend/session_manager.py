import streamlit as st
from typing import Dict, List, Optional, Any


def initialize_session_state():
    """Initialize session state variables"""
    if 'document_structure' not in st.session_state:
        st.session_state.document_structure = []
    
    if 'document_pages' not in st.session_state:
        st.session_state.document_pages = []
    
    if 'document_content' not in st.session_state:
        st.session_state.document_content = ""
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    
    if 'document_info' not in st.session_state:
        st.session_state.document_info = {}
    
    if 'last_file_hash' not in st.session_state:
        st.session_state.last_file_hash = None


def reset_document_state():
    """Reset document-related session state"""
    st.session_state.document_structure = []
    st.session_state.document_pages = []
    st.session_state.document_content = ""
    st.session_state.current_page = 0
    st.session_state.document_info = {}


def is_new_file(uploaded_file) -> bool:
    """Check if the uploaded file is different from the last one"""
    if uploaded_file is None:
        return False
    
    # Create a simple hash from filename and size
    file_hash = f"{uploaded_file.name}_{len(uploaded_file.getvalue())}"
    
    if st.session_state.last_file_hash != file_hash:
        st.session_state.last_file_hash = file_hash
        return True
    
    return False


def navigate_to_section(page: int):
    """Navigate to a specific page/section"""
    if 'document_pages' in st.session_state and st.session_state.document_pages:
        max_page = len(st.session_state.document_pages) - 1
        st.session_state.current_page = max(0, min(page, max_page))


def get_document_info() -> Dict[str, Any]:
    """Get current document information"""
    return st.session_state.get('document_info', {})


def update_document_info(info: Dict[str, Any]):
    """Update document information"""
    st.session_state.document_info.update(info)


def set_document_structure(structure: List[Dict]):
    """Set document structure"""
    st.session_state.document_structure = structure


def set_document_pages(pages: List):
    """Set document pages"""
    st.session_state.document_pages = pages
    if pages and st.session_state.current_page >= len(pages):
        st.session_state.current_page = 0


def set_document_content(content: str):
    """Set document content"""
    st.session_state.document_content = content


def get_current_page() -> int:
    """Get current page number"""
    return st.session_state.get('current_page', 0)


def get_total_pages() -> int:
    """Get total number of pages"""
    pages = st.session_state.get('document_pages', [])
    return len(pages) if pages else 0 