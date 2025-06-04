import streamlit as st
from pathlib import Path
import sys

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

# Import backend modules
from backend import (
    # Session management
    initialize_session_state,
    is_new_file,
    reset_document_state,
    get_document_info,
    
    # Document processing
    extract_pdf_pages,
    extract_docx_content,
    analyze_document_structure,
    
    # UI components
    check_pdf_viewer_availability,
    display_pdf_viewer,
    display_pdf_with_viewer,
    display_docx_content,
    render_sidebar_structure,
    render_document_info,
    render_control_panel,
    render_upload_area,
)

# Page configuration
st.set_page_config(
    page_title="Document Visual Parser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        border-bottom: 2px solid #f0f2f6;
    }
    
    .sidebar-content {
        padding: 1rem 0;
    }
    
    .chapter-item {
        padding: 0.5rem;
        margin: 0.2rem 0;
        background-color: #f8f9fa;
        border-radius: 5px;
        border-left: 3px solid #0066cc;
    }
    
    .document-viewer {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        background-color: #fafafa;
        min-height: 600px;
    }
    
    .upload-area {
        border: 2px dashed #0066cc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9ff;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def process_pdf_document(uploaded_file):
    """Process PDF document and extract structure"""
    st.info("📄 Processing PDF document...")
    
    # Extract document structure immediately after upload
    if not st.session_state.document_structure:
        with st.spinner("🔍 Analyzing document structure..."):
            structure = analyze_document_structure(uploaded_file)
            st.session_state.document_structure = structure
            if structure:
                st.success(f"✅ Found {len(structure)} headings in document structure")
            else:
                st.info("📝 No clear document structure detected. The document may not have distinct headings or may use non-standard formatting.")
    
    # Try to use the PDF viewer first for better experience
    pdf_viewer_success = display_pdf_with_viewer(uploaded_file)
    
    if not pdf_viewer_success:
        # Fallback to image-based display
        st.info("Using fallback image-based PDF display...")
        
        # Extract pages if not already done
        if not st.session_state.document_pages:
            with st.spinner("Converting PDF pages to images..."):
                pages = extract_pdf_pages(uploaded_file)
                if pages and len(pages) > 0:
                    st.session_state.document_pages = pages
                    st.session_state.current_page = 0
                    st.success(f"✅ PDF loaded successfully! ({len(pages)} pages)")
                else:
                    st.error("❌ Failed to load PDF pages - no pages were extracted")
        
        # Display the image-based viewer
        if st.session_state.document_pages:
            display_pdf_viewer(st.session_state.document_pages)
    
    else:
        st.success("✅ PDF loaded successfully with text selection enabled!")


def process_docx_document(uploaded_file):
    """Process DOCX document"""
    with st.spinner("Processing DOCX document..."):
        content = extract_docx_content(uploaded_file)
        
        # Also extract structure for DOCX
        if not st.session_state.document_structure:
            structure = analyze_document_structure(uploaded_file)
            st.session_state.document_structure = structure
    
    if content:
        st.success("✅ DOCX loaded successfully!")
        display_docx_content(content)
    else:
        st.error("Failed to load DOCX content")


def main():
    """Main application function"""
    initialize_session_state()
    
    # Check PDF viewer availability
    check_pdf_viewer_availability()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📄 Document Visual Parser</h1>
        <p>Upload and analyze your documents with AI-powered structure detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for chapters/sections
    with st.sidebar:
        st.markdown("### 📑 Document Structure")
        st.markdown("---")
        
        # Render document structure
        render_sidebar_structure(st.session_state.document_structure)
        
        st.markdown("---")
        
        # Render document info
        doc_info = get_document_info()
        render_document_info(doc_info)
    
    # Main content area - Fixed layout to prevent PDF displaying at the bottom
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # File upload section
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=['pdf', 'docx'],
            help="Upload PDF or DOCX files to analyze their structure"
        )
        
        if uploaded_file is not None:
            # Check if this is a new file
            if is_new_file(uploaded_file):
                # Reset state for new file
                reset_document_state()
            
            st.session_state.uploaded_file = uploaded_file
            
            # Process the uploaded file based on type
            if uploaded_file.type == "application/pdf":
                process_pdf_document(uploaded_file)
            
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                process_docx_document(uploaded_file)
        
        else:
            # Upload instructions
            render_upload_area()
    
    with col2:
        # Control panel
        render_control_panel(st.session_state.uploaded_file)


if __name__ == "__main__":
    main() 