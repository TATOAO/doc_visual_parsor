import streamlit as st


def initialize_session_state():
    """Initialize session state variables"""
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'document_pages' not in st.session_state:
        st.session_state.document_pages = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    if 'chapters' not in st.session_state:
        st.session_state.chapters = []
    if 'document_structure' not in st.session_state:
        st.session_state.document_structure = []
    if 'pdf_document' not in st.session_state:
        st.session_state.pdf_document = None
    if 'target_page' not in st.session_state:
        st.session_state.target_page = 0


def reset_document_state():
    """Reset document-related session state for new file"""
    st.session_state.document_structure = []
    st.session_state.document_pages = []
    st.session_state.current_page = 0
    st.session_state.target_page = 0
    if st.session_state.pdf_document:
        try:
            st.session_state.pdf_document.close()
        except:
            pass
        st.session_state.pdf_document = None


def is_new_file(uploaded_file):
    """Check if the uploaded file is different from the current one"""
    return (st.session_state.uploaded_file is None or 
            st.session_state.uploaded_file.name != uploaded_file.name)


def navigate_to_section(page_num):
    """Navigate to a specific page/section"""
    st.session_state.target_page = page_num
    
    if st.session_state.document_pages:
        # For image-based viewer
        st.session_state.current_page = min(page_num, len(st.session_state.document_pages) - 1)
        st.success(f"📍 Jumped to page {page_num + 1}")
        st.rerun()
    else:
        # For PDF viewer, use target_page to control display
        st.success(f"📍 Navigating to page {page_num + 1}")
        st.rerun()


def get_document_info():
    """Get information about the current document"""
    if not st.session_state.uploaded_file:
        return None
    
    return {
        'name': st.session_state.uploaded_file.name,
        'size': f"{st.session_state.uploaded_file.size / 1024:.1f} KB",
        'type': st.session_state.uploaded_file.type,
        'sections': len(st.session_state.document_structure) if st.session_state.document_structure else 0,
        'pages': len(st.session_state.document_pages) if st.session_state.document_pages else "N/A"
    } 