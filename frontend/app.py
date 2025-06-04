import streamlit as st
import fitz  # PyMuPDF
import docx
from PIL import Image
import io
import base64
from pathlib import Path
import tempfile
import os

# Try to import streamlit-pdf-viewer, fallback to image display if not available
try:
    from streamlit_pdf_viewer import pdf_viewer
    HAS_PDF_VIEWER = True
except ImportError:
    HAS_PDF_VIEWER = False
    st.warning("⚠️ For better PDF viewing with text selection, install: pip install streamlit-pdf-viewer")

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

def extract_pdf_pages(pdf_file):
    """Extract pages from PDF as images"""
    pages = []
    pdf_doc = None
    tmp_file_path = None
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Open PDF with PyMuPDF
        pdf_doc = fitz.open(tmp_file_path)
        
        # Process each page
        for page_num in range(pdf_doc.page_count):
            try:
                page = pdf_doc.load_page(page_num)
                
                # Convert page to image with good quality
                mat = fitz.Matrix(1.5, 1.5)  # 1.5x zoom for good quality/performance balance
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                pages.append(img)
                
            except Exception as page_error:
                st.error(f"❌ Error processing page {page_num + 1}: {str(page_error)}")
                continue
        
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        
    finally:
        # Clean up resources
        if pdf_doc:
            pdf_doc.close()
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception as cleanup_error:
                st.warning(f"⚠️ Could not clean up temp file: {cleanup_error}")
    
    return pages

def extract_docx_content(docx_file):
    """Extract content from DOCX file"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            tmp_file.write(docx_file.getvalue())
            tmp_file_path = tmp_file.name
        
        doc = docx.Document(tmp_file_path)
        content = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                content.append(paragraph.text)
        
        os.unlink(tmp_file_path)  # Clean up temp file
        return content
        
    except Exception as e:
        st.error(f"Error processing DOCX: {str(e)}")
        return []

def display_pdf_viewer(pages):
    """Display PDF pages with navigation (fallback method using images)"""
    if not pages:
        return
    
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        if st.session_state.current_page > 0:
            if st.button("◀ Previous Page"):
                st.session_state.current_page -= 1
                st.rerun()
        
        st.write(f"Page {st.session_state.current_page + 1} of {len(pages)}")
        
        if st.session_state.current_page < len(pages) - 1:
            if st.button("Next Page ▶"):
                st.session_state.current_page += 1
                st.rerun()
    
    # Display current page with fixed deprecated parameter
    current_page = pages[st.session_state.current_page]
    st.image(current_page, use_container_width=True)

def display_pdf_with_viewer(pdf_file):
    """Display PDF using streamlit-pdf-viewer for better text selection"""
    if HAS_PDF_VIEWER:
        try:
            # Get binary data from uploaded file
            binary_data = pdf_file.getvalue()
            
            # Display PDF with streamlit-pdf-viewer
            pdf_viewer(
                input=binary_data,
                # width="100%",
                width="200%",
                render_text=True  # Enable text selection and copying
            )
            return True
        except Exception as e:
            st.error(f"Error displaying PDF with viewer: {str(e)}")
            return False
    return False

def display_docx_content(content):
    """Display DOCX content"""
    if not content:
        return
    
    st.markdown("### Document Content")
    for paragraph in content:
        st.write(paragraph)

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
    
    # Sidebar for chapters/sections
    with st.sidebar:
        st.markdown("### 📑 Document Structure")
        st.markdown("---")
        
        if st.session_state.chapters:
            st.markdown("#### Chapters & Sections")
            for i, chapter in enumerate(st.session_state.chapters):
                st.markdown(f"""
                <div class="chapter-item">
                    📖 {chapter}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sidebar-content">
                <p>📝 <strong>Chapters will appear here</strong></p>
                <p>Once you upload a document, the AI will automatically detect and display the document structure including:</p>
                <ul>
                    <li>📖 Chapters</li>
                    <li>📝 Sections</li>
                    <li>📊 Tables</li>
                    <li>🖼️ Images</li>
                    <li>📎 Attachments</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Document info
        if st.session_state.uploaded_file:
            st.markdown("#### Document Info")
            st.write(f"**Name:** {st.session_state.uploaded_file.name}")
            st.write(f"**Size:** {st.session_state.uploaded_file.size / 1024:.1f} KB")
            st.write(f"**Type:** {st.session_state.uploaded_file.type}")
    
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
            st.session_state.uploaded_file = uploaded_file
            
            # Process the uploaded file
            if uploaded_file.type == "application/pdf":
                st.info("📄 Processing PDF document...")
                
                # Try to use the PDF viewer first for better experience
                pdf_viewer_success = display_pdf_with_viewer(uploaded_file)
                
                if not pdf_viewer_success:
                    # Fallback to image-based display
                    st.info("Using fallback image-based PDF display...")
                    
                    # Clear previous pages
                    st.session_state.document_pages = []
                    st.session_state.current_page = 0
                    
                    # Create a container for processing status
                    status_container = st.container()
                    
                    with status_container:
                        pages = extract_pdf_pages(uploaded_file)
                    
                    if pages and len(pages) > 0:
                        st.session_state.document_pages = pages
                        st.session_state.current_page = 0
                        st.success(f"✅ PDF loaded successfully! ({len(pages)} pages)")
                        
                        # Clear the processing status and show the document
                        status_container.empty()
                        display_pdf_viewer(pages)
                    else:
                        st.error("❌ Failed to load PDF pages - no pages were extracted")
                else:
                    st.success("✅ PDF loaded successfully with text selection enabled!")
            
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                with st.spinner("Processing DOCX document..."):
                    content = extract_docx_content(uploaded_file)
                
                if content:
                    st.success("✅ DOCX loaded successfully!")
                    display_docx_content(content)
                else:
                    st.error("Failed to load DOCX content")
        
        else:
            # Upload instructions
            st.markdown("""
            <div class="upload-area">
                <h3>📤 Upload Your Document</h3>
                <p>Drag and drop a PDF or DOCX file here, or click to browse</p>
                <p><strong>Supported formats:</strong> PDF, DOCX</p>
                <p><em>For best PDF viewing experience with text selection, ensure streamlit-pdf-viewer is installed</em></p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Control panel
        st.markdown("### ⚙️ Controls")
        
        if st.session_state.uploaded_file:
            st.markdown("#### Actions")
            if st.button("🔄 Reprocess Document"):
                st.session_state.document_pages = []
                st.session_state.current_page = 0
                st.rerun()
            
            if st.button("📤 Export Results"):
                st.info("Export functionality will be available soon!")
            
            st.markdown("#### Layout Detection")
            st.selectbox("Detection Model", ["DocLayout-YOLO", "Custom Model"], disabled=True)
            st.slider("Confidence Threshold", 0.1, 1.0, 0.5, disabled=True)
            
            st.markdown("#### Visualization")
            st.checkbox("Show Bounding Boxes", disabled=True)
            st.checkbox("Show Confidence Scores", disabled=True)
            st.checkbox("Highlight Tables", disabled=True)
            st.checkbox("Highlight Images", disabled=True)
            
            # PDF Viewer status
            if st.session_state.uploaded_file and st.session_state.uploaded_file.type == "application/pdf":
                st.markdown("#### PDF Viewer")
                if HAS_PDF_VIEWER:
                    st.success("✅ Advanced PDF viewer enabled")
                    st.info("Text selection and copying available")
                else:
                    st.warning("⚠️ Using basic image viewer")
                    st.info("Install streamlit-pdf-viewer for text selection")
        
        else:
            st.info("Upload a document to access controls")

if __name__ == "__main__":
    main() 