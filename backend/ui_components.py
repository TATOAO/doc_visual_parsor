import streamlit as st
from .session_manager import navigate_to_section

# Try to import streamlit-pdf-viewer, fallback to image display if not available
try:
    from streamlit_pdf_viewer import pdf_viewer
    HAS_PDF_VIEWER = True
except ImportError:
    HAS_PDF_VIEWER = False


def check_pdf_viewer_availability():
    """Check if PDF viewer is available and show warning if not"""
    if not HAS_PDF_VIEWER:
        st.warning("⚠️ For better PDF viewing with text selection, install: pip install streamlit-pdf-viewer")
    return HAS_PDF_VIEWER


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
            
            # Calculate the page to display based on navigation
            page_to_show = st.session_state.get('target_page', 0)
            
            # Display PDF with streamlit-pdf-viewer
            pdf_viewer(
                input=binary_data,
                width="200%",
                render_text=True,  # Enable text selection and copying
                key=f"pdf_viewer_{page_to_show}",  # Dynamic key to force re-render
                pages_to_render=list(range(max(0, page_to_show), min(page_to_show + 5, 100)))  # Show target page and a few around it
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


def render_sidebar_structure(document_structure):
    """Render the document structure in the sidebar"""
    if document_structure:
        st.markdown("#### Table of Contents")
        
        # Display the document structure as clickable items
        for i, item in enumerate(document_structure):
            # Create indentation based on heading level
            indent = "　" * (item['level'] - 1)  # Japanese space for better indentation
            
            # Create a unique key for each button
            button_key = f"nav_{i}_{item['page']}"
            
            # Style based on heading level
            if item['level'] == 1:
                icon = "📖"
                text_style = "**{}**"
            elif item['level'] == 2:
                icon = "📝"
                text_style = "{}"
            else:
                icon = "▪️"
                text_style = "{}"
            
            # Create clickable button for navigation
            display_text = f"{indent}{icon} {text_style.format(item['text'])}"
            
            if st.button(
                display_text,
                key=button_key,
                help=f"Go to page {item['page'] + 1}",
                use_container_width=True
            ):
                navigate_to_section(item['page'])
            
            # Show page number
            st.caption(f"{indent}　Page {item['page'] + 1}")
    
    else:
        st.markdown("""
        <div class="sidebar-content">
            <p>📝 <strong>Document structure will appear here</strong></p>
            <p>Once you upload a PDF document, the AI will automatically detect and display:</p>
            <ul>
                <li>📖 Chapter headings</li>
                <li>📝 Section titles</li>
                <li>▪️ Subsections</li>
            </ul>
            <p><em>Click any heading to jump to that section!</em></p>
        </div>
        """, unsafe_allow_html=True)


def render_document_info(doc_info):
    """Render document information in the sidebar"""
    if doc_info:
        st.markdown("#### Document Info")
        st.write(f"**Name:** {doc_info['name']}")
        st.write(f"**Size:** {doc_info['size']}")
        st.write(f"**Type:** {doc_info['type']}")
        
        if doc_info['sections'] > 0:
            st.write(f"**Sections:** {doc_info['sections']}")


def render_control_panel(uploaded_file):
    """Render the control panel in the right column"""
    st.markdown("### ⚙️ Controls")
    
    if uploaded_file:
        st.markdown("#### Actions")
        if st.button("🔄 Reprocess Document"):
            from .session_manager import reset_document_state
            reset_document_state()
            st.rerun()
        
        if st.button("📤 Export Structure"):
            if st.session_state.document_structure:
                # Create downloadable content structure
                structure_text = "Document Structure:\n\n"
                for item in st.session_state.document_structure:
                    indent = "  " * (item['level'] - 1)
                    structure_text += f"{indent}- {item['text']} (Page {item['page'] + 1})\n"
                
                st.download_button(
                    label="💾 Download Structure",
                    data=structure_text,
                    file_name=f"{uploaded_file.name}_structure.txt",
                    mime="text/plain"
                )
            else:
                st.info("No structure to export")
        
        st.markdown("#### Navigation")
        if st.session_state.document_pages:
            # Page navigation for image-based viewer
            page_num = st.selectbox(
                "Jump to page:",
                range(len(st.session_state.document_pages)),
                index=st.session_state.current_page,
                format_func=lambda x: f"Page {x + 1}"
            )
            if page_num != st.session_state.current_page:
                navigate_to_section(page_num)
        
        elif HAS_PDF_VIEWER and uploaded_file:
            # Show current target page for PDF viewer
            if 'target_page' in st.session_state:
                st.info(f"Current view: Page {st.session_state.target_page + 1}")
        
        st.markdown("#### Structure Detection")
        st.info("Automatic heading detection based on font size and formatting")
        
        # Show structure analysis status
        if st.session_state.document_structure:
            st.success(f"✅ {len(st.session_state.document_structure)} headings detected")
        
        # PDF Viewer status
        if uploaded_file.type == "application/pdf":
            st.markdown("#### PDF Viewer")
            if HAS_PDF_VIEWER:
                st.success("✅ Advanced PDF viewer enabled")
                st.info("Text selection and copying available")
            else:
                st.warning("⚠️ Using basic image viewer")
                st.info("Install streamlit-pdf-viewer for text selection")
    
    else:
        st.info("Upload a document to access controls")


def render_upload_area():
    """Render the file upload area"""
    st.markdown("""
    <div class="upload-area">
        <h3>📤 Upload Your Document</h3>
        <p>Drag and drop a PDF or DOCX file here, or click to browse</p>
        <p><strong>Supported formats:</strong> PDF, DOCX</p>
        <p><em>The document structure will be automatically analyzed and displayed in the sidebar</em></p>
    </div>
    """, unsafe_allow_html=True) 