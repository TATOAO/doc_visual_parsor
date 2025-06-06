import streamlit as st
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from frontend.api_client import get_api_client


def check_api_connection():
    """Check if the API backend is available"""
    api_client = get_api_client()
    
    if api_client.health_check():
        st.sidebar.success("🟢 Backend API Connected")
        return True
    else:
        st.sidebar.error("🔴 Backend API Disconnected")
        st.sidebar.info("Please ensure the backend server is running on http://localhost:8000")
        return False


def render_sidebar_structure(structure: List[Dict]):
    """Render document structure in sidebar"""
    if not structure:
        st.write("📝 No document structure detected")
        return
    
    st.write(f"📖 Found {len(structure)} headings:")
    
    for item in structure:
        level = item.get('level', 1)
        text = item.get('text', 'Unknown')
        page = item.get('page', 0)
        
        # Indent based on heading level
        indent = "  " * (level - 1)
        
        # Create a clickable item
        if st.button(f"{indent}📄 {text[:50]}{'...' if len(text) > 50 else ''}", 
                    key=f"nav_{page}_{text[:20]}", 
                    help=f"Page {page + 1}: {text}"):
            # Navigate to the specific page/section
            if 'current_page' in st.session_state:
                st.session_state.current_page = page
                st.rerun()


def render_document_info(doc_info: Dict):
    """Render document information"""
    if not doc_info:
        return
    
    st.markdown("### 📊 Document Info")
    
    if 'filename' in doc_info:
        st.write(f"**File:** {doc_info['filename']}")
    
    if 'file_type' in doc_info:
        st.write(f"**Type:** {doc_info['file_type']}")
    
    if 'total_pages' in doc_info:
        st.write(f"**Pages:** {doc_info['total_pages']}")
    
    if 'structure_summary' in doc_info:
        st.write(f"**Structure:** {doc_info['structure_summary']}")
    
    if 'size' in doc_info:
        size_mb = doc_info['size'] / (1024 * 1024)
        st.write(f"**Size:** {size_mb:.2f} MB")


def render_control_panel(uploaded_file):
    """Render control panel for document navigation"""
    if not uploaded_file:
        return
    
    st.markdown("### 🎛️ Controls")
    
    # API connection status
    check_api_connection()
    
    # Page navigation for PDFs
    if uploaded_file.type == "application/pdf" and 'document_pages' in st.session_state:
        total_pages = len(st.session_state.document_pages)
        
        if total_pages > 1:
            current_page = st.session_state.get('current_page', 0)
            
            new_page = st.number_input(
                "Go to page:",
                min_value=1,
                max_value=total_pages,
                value=current_page + 1,
                step=1
            ) - 1
            
            if new_page != current_page:
                st.session_state.current_page = new_page
                st.rerun()
            
            # Page navigation buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("⬅️ Previous", disabled=current_page <= 0):
                    st.session_state.current_page = current_page - 1
                    st.rerun()
            
            with col2:
                if st.button("➡️ Next", disabled=current_page >= total_pages - 1):
                    st.session_state.current_page = current_page + 1
                    st.rerun()
            
            st.write(f"Page {current_page + 1} of {total_pages}")
    
    # Refresh button
    if st.button("🔄 Refresh Document"):
        # Clear cached data and reprocess
        for key in ['document_structure', 'document_pages', 'document_content']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


def render_upload_area():
    """Render the upload area with instructions"""
    st.markdown("""
    <div class="upload-area">
        <h3>📁 Upload Your Document</h3>
        <p>Drag and drop or click to select:</p>
        <ul>
            <li>📄 PDF files (.pdf)</li>
            <li>📄 Word documents (.docx)</li>
        </ul>
        <p><small>Maximum file size: 200MB</small></p>
    </div>
    """, unsafe_allow_html=True)


def display_pdf_viewer(pages: List):
    """Display PDF pages as images"""
    if not pages:
        st.error("No pages to display")
        return
    
    current_page = st.session_state.get('current_page', 0)
    
    if 0 <= current_page < len(pages):
        # Display current page
        page_image = pages[current_page]
        st.image(page_image, caption=f"Page {current_page + 1}", use_column_width=True)
    else:
        st.error(f"Invalid page number: {current_page + 1}")


def display_pdf_with_viewer(uploaded_file) -> bool:
    """Try to display PDF with streamlit-pdf-viewer, fallback to images"""
    try:
        import streamlit_pdf_viewer as pdf_viewer
        
        # Try to display with PDF viewer
        pdf_viewer.pdf_viewer(uploaded_file.getvalue())
        return True
        
    except ImportError:
        st.info("PDF viewer not available, using image display")
        return False
    except Exception as e:
        st.warning(f"PDF viewer failed: {str(e)}, using image display")
        return False


def display_docx_content(content: str):
    """Display DOCX content"""
    if not content:
        st.error("No content to display")
        return
    
    st.markdown("### 📄 Document Content")
    
    # Create scrollable container for content
    st.markdown(
        f"""
        <div style="
            max-height: 600px;
            overflow-y: auto;
            padding: 1rem;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: white;
        ">
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )


def display_api_status():
    """Display API connection status in main area"""
    api_client = get_api_client()
    
    if api_client.health_check():
        st.success("✅ Backend API is connected and running")
    else:
        st.error("❌ Cannot connect to backend API")
        st.info("""
        **Please ensure the backend server is running:**
        
        1. Open a terminal in the project directory
        2. Run: `python -m backend.api_server`
        3. The server should start on http://localhost:8000
        
        Or use the startup script: `python run_backend.py`
        """)
        return False
    
    return True 