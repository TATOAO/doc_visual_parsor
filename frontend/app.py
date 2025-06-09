import streamlit as st
from pathlib import Path
import base64
from io import BytesIO

# Add project root to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import frontend modules
from frontend.api_client import get_api_client
from frontend.session_manager import (
    initialize_session_state,
    is_new_file,
    reset_document_state,
    get_document_info,
    update_document_info,
    set_document_structure,
    set_document_pages,
    set_document_content,
    get_section_tree,
    get_llm_annotated_text,
)
from frontend.ui_components import (
    check_api_connection,
    display_pdf_viewer,
    display_pdf_with_viewer,
    display_docx_content,
    display_docx_sections_with_highlighting,
    render_sidebar_structure,
    render_section_tree_sidebar,
    render_document_info,
    render_control_panel,
    render_naive_llm_controls,
    render_upload_area,
    display_api_status,
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
    """Process PDF document using API"""
    st.info("📄 Processing PDF document...")
    
    api_client = get_api_client()
    
    # Check if document structure needs to be extracted
    if not st.session_state.document_structure:
        with st.spinner("🔍 Analyzing document structure..."):
            # Use the new PDF structure analysis endpoint (no image conversion)
            result = api_client.analyze_pdf_structure(uploaded_file)
            
            if result and result.get('success'):
                # Update session state with API response
                structure = result.get('structure', [])
                set_document_structure(structure)
                
                # Update document info
                doc_info = {
                    'filename': result.get('filename'),
                    'file_type': result.get('file_type'),
                    'size': result.get('size'),
                    'total_pages': result.get('total_pages'),
                    'structure_summary': result.get('structure_summary')
                }
                update_document_info(doc_info)
                
                if structure:
                    st.success(f"✅ Found {len(structure)} headings in document structure")
                else:
                    st.info("📝 No clear document structure detected. The document may not have distinct headings or may use non-standard formatting.")
            else:
                st.error("❌ Failed to analyze PDF document structure")
                return
    
    # Try to use the PDF viewer first for better experience with text selection
    pdf_viewer_success = display_pdf_with_viewer(uploaded_file)
    
    if not pdf_viewer_success:
        # Fallback to image-based display
        st.info("📸 Using image-based PDF display...")
        
        # Load pages only when needed for fallback
        if not st.session_state.document_pages:
            with st.spinner("🖼️ Converting PDF pages to images..."):
                pages_result = api_client.extract_pdf_pages_into_images(uploaded_file)
                if pages_result and 'pages' in pages_result:
                    pages = []
                    for page_data in pages_result['pages']:
                        img_data = base64.b64decode(page_data['image_data'])
                        from PIL import Image
                        img = Image.open(BytesIO(img_data))
                        pages.append(img)
                    set_document_pages(pages)
        
        # Use pages from session state
        if st.session_state.document_pages:
            display_pdf_viewer(st.session_state.document_pages)
            st.success(f"✅ PDF loaded successfully! ({len(st.session_state.document_pages)} pages)")
        else:
            st.error("❌ Failed to load PDF pages")


def process_docx_document(uploaded_file):
    """Process DOCX document using API"""
    st.info("📄 Processing DOCX document...")
    
    api_client = get_api_client()
    
    with st.spinner("Processing DOCX document..."):
        # Get document content
        content_result = api_client.extract_docx_content(uploaded_file)
        
        # Get document structure
        structure_result = api_client.analyze_structure(uploaded_file)
        
        if content_result and structure_result:
            # Update session state
            content = content_result.get('content', '')
            structure = structure_result.get('structure', [])
            
            set_document_content(content)
            set_document_structure(structure)
            
            # Update document info
            doc_info = {
                'filename': content_result.get('filename'),
                'file_type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'structure_summary': structure_result.get('structure_summary')
            }
            update_document_info(doc_info)
            
            st.success("✅ DOCX loaded successfully!")
            
            # Check if we have section tree data from naive_llm
            section_tree = get_section_tree()
            if section_tree and content:
                # Display with AI highlighting
                display_docx_sections_with_highlighting(content, section_tree)
            else:
                # Display regular content
                display_docx_content(content)
        else:
            st.error("Failed to load DOCX content")


def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📄 Document Visual Parser</h1>
        <p>Upload and analyze your documents with AI-powered structure detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API connection first
    if not display_api_status():
        return
    
    # Sidebar for chapters/sections
    with st.sidebar:
        st.markdown("### 📑 Document Structure")
        st.markdown("---")
        
        # Check if we have section tree from naive_llm
        section_tree = get_section_tree()
        if section_tree:
            # Render AI-parsed section tree
            render_section_tree_sidebar(section_tree)
        else:
            # Render traditional document structure
            render_sidebar_structure(st.session_state.document_structure)
        
        st.markdown("---")
        
        # Show PDF viewer status
        if st.session_state.get('uploaded_file') and st.session_state.uploaded_file.type == "application/pdf":
            try:
                import streamlit_pdf_viewer
                st.success("✅ PDF text selection enabled")
                st.info("📝 You can select and copy text from PDFs")
            except ImportError:
                st.warning("⚠️ Text selection disabled")
                st.info("📝 To enable text selection:")
                st.code("pip install streamlit-pdf-viewer")
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

        # Naive LLM controls for DOCX files
        render_naive_llm_controls(st.session_state.uploaded_file)




if __name__ == "__main__":
    main() 