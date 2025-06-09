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
    """Display PDF with streamlit-pdf-viewer for text selection"""
    try:
        import streamlit_pdf_viewer as pdf_viewer
        
        # Display PDF with streamlit-pdf-viewer with text selection enabled
        pdf_viewer.pdf_viewer(
            input=uploaded_file.getvalue(),
            width="100%",
            height=600,
            render_text=True,  # Enable text selection and copying
            key="pdf_viewer_main"
        )
        
        # Show text selection instructions
        st.info("💡 **Text Selection Enabled**: You can now select and copy text directly from the PDF above!")
        return True
        
    except ImportError:
        st.warning("📄 **streamlit-pdf-viewer not installed**")
        st.info("To enable text selection from PDFs, install: `pip install streamlit-pdf-viewer`")
        return False
    except Exception as e:
        st.warning(f"PDF viewer failed: {str(e)}")
        st.info("Falling back to image-based display...")
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


def display_docx_sections_with_highlighting(raw_text: str, section_tree: Dict[str, Any]):
    """Display DOCX content with section highlighting based on position indices"""
    if not raw_text or not section_tree:
        st.error("No content or section tree to display")
        return
    
    st.markdown("### 📄 Document Sections (LLM Parsed)")
    
    # Function to recursively collect all sections
    def collect_sections(section_data):
        sections = []
        if isinstance(section_data, dict):
            if section_data.get('title') and section_data.get('title_position_index'):
                sections.append(section_data)
            
            for sub_section in section_data.get('sub_sections', []):
                sections.extend(collect_sections(sub_section))
        
        return sections
    
    # Collect all sections with position indices
    all_sections = collect_sections(section_tree)
    
    if not all_sections:
        st.info("No sections with position indices found")
        return
    
    # Sort sections by title position for proper highlighting
    all_sections.sort(key=lambda x: x.get('title_position_index', [0, 0])[0])
    
    # Create HTML with highlighting
    html_content = ""
    last_pos = 0
    
    for i, section in enumerate(all_sections):
        title_pos = section.get('title_position_index', [-1, -1])
        content_pos = section.get('content_position_index', [-1, -1])
        level = section.get('level', 1)
        title = section.get('title', 'Untitled')
        
        if title_pos[0] >= 0 and title_pos[1] >= 0:
            # Add text before this section
            if title_pos[0] > last_pos:
                html_content += f'<span>{raw_text[last_pos:title_pos[0]]}</span>'
            
            # Add highlighted title
            color_intensity = max(0.3, 1.0 - (level - 1) * 0.15)
            title_color = f"rgba(255, 215, 0, {color_intensity})"  # Gold color with varying intensity
            
            title_text = raw_text[title_pos[0]:title_pos[1]]
            html_content += f'<span style="background-color: {title_color}; padding: 2px 4px; border-left: 3px solid #ff6b35; font-weight: bold;" title="Section: {title} (Level {level})">{title_text}</span>'
            
            # Add highlighted content if available
            if content_pos[0] >= 0 and content_pos[1] >= 0 and content_pos[0] < content_pos[1]:
                content_color = f"rgba(173, 216, 230, {color_intensity * 0.5})"  # Light blue
                content_text = raw_text[content_pos[0]:content_pos[1]]
                html_content += f'<span style="background-color: {content_color}; padding: 1px 2px;" title="Content for: {title}">{content_text}</span>'
                last_pos = content_pos[1]
            else:
                last_pos = title_pos[1]
    
    # Add any remaining text
    if last_pos < len(raw_text):
        html_content += f'<span>{raw_text[last_pos:]}</span>'
    
    # Display in a scrollable container
    st.markdown(
        f"""
        <div style="
            max-height: 600px;
            overflow-y: auto;
            padding: 1rem;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: white;
            font-family: monospace;
            line-height: 1.6;
            white-space: pre-wrap;
        ">
            {html_content}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Add legend
    st.markdown("""
    **Legend:**
    - 🟡 **Yellow highlights**: Section titles (darker = higher level)
    - 🔵 **Blue highlights**: Section content
    - 🟠 **Orange left border**: Section boundary
    """)


def render_section_tree_sidebar(section_tree: Dict[str, Any]):
    """Render the section tree in sidebar with hierarchical structure"""
    if not section_tree:
        st.write("📝 No section tree available")
        return
    
    st.markdown("### 🌳 Section Tree (LLM)")
    
    def render_section_node(section_data, level=0):
        """Recursively render section nodes"""
        if not isinstance(section_data, dict):
            return
        
        title = section_data.get('title', 'Untitled')
        section_level = section_data.get('level', level)
        title_pos = section_data.get('title_position_index', [-1, -1])
        content_pos = section_data.get('content_position_index', [-1, -1])
        
        # Create indentation for hierarchical display
        indent = "  " * level
        
        # Create a button for this section
        if title and title.strip():
            button_text = f"{indent}{'📁' if section_data.get('sub_sections') else '📄'} {title[:40]}{'...' if len(title) > 40 else ''}"
            
            if st.button(
                button_text,
                key=f"section_tree_{level}_{title[:20]}_{title_pos[0]}",
                help=f"Level {section_level}: {title}\nTitle: {title_pos}\nContent: {content_pos}"
            ):
                st.info(f"Selected section: {title}")
                # You can add navigation logic here if needed
        
        # Render sub-sections
        for sub_section in section_data.get('sub_sections', []):
            render_section_node(sub_section, level + 1)
    
    # Start rendering from root
    render_section_node(section_tree)


def render_naive_llm_controls(uploaded_file):
    """Render controls for naive_llm processing"""
    if not uploaded_file or uploaded_file.type != "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return
    
    st.markdown("### 🧠 AI Structure Analysis")
    
    # Button to trigger naive_llm processing
    if st.button("🚀 Parse with AI (Naive LLM)", 
                key="naive_llm_button",
                help="Use AI to intelligently parse document structure"):
        
        api_client = get_api_client()
        
        with st.spinner("🧠 AI is analyzing document structure..."):
            result = api_client.analyze_docx_with_naive_llm(uploaded_file)
            
            if result and result.get('success'):
                # Import session manager functions
                from frontend.session_manager import set_section_tree, set_llm_annotated_text
                
                # Store results in session state
                set_section_tree(result['section_tree'])
                set_llm_annotated_text(result['llm_annotated_text'])
                
                st.success("✅ AI analysis completed!")
                st.rerun()
            else:
                st.error("❌ AI analysis failed")
    
    # Show status if we have section tree data
    if st.session_state.get('section_tree'):
        st.success("🧠 AI analysis available")
        if st.button("🔄 Clear AI Analysis"):
            st.session_state.section_tree = None
            st.session_state.llm_annotated_text = ""
            st.rerun() 