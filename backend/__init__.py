# Backend package for Document Visual Parser

from .pdf_processor import extract_pdf_pages_into_images, get_pdf_document_object, close_pdf_document
from .docx_processor import extract_docx_content, extract_docx_structure
from .document_analyzer import (
    extract_pdf_document_structure, 
    analyze_document_structure, 
    get_structure_summary
)
from .session_manager import (
    initialize_session_state, 
    reset_document_state, 
    is_new_file, 
    navigate_to_section, 
    get_document_info
)
from .ui_components import (
    check_pdf_viewer_availability,
    display_pdf_viewer,
    display_pdf_with_viewer,
    display_docx_content,
    render_sidebar_structure,
    render_document_info,
    render_control_panel,
    render_upload_area
)

__all__ = [
    # PDF processing
    'extract_pdf_pages_into_images',
    'get_pdf_document_object',
    'close_pdf_document',
    
    # DOCX processing
    'extract_docx_content',
    'extract_docx_structure',
    
    # Document analysis
    'extract_pdf_document_structure',
    'analyze_document_structure',
    'get_structure_summary',
    
    # Session management
    'initialize_session_state',
    'reset_document_state',
    'is_new_file',
    'navigate_to_section',
    'get_document_info',
    
    # UI components
    'check_pdf_viewer_availability',
    'display_pdf_viewer',
    'display_pdf_with_viewer',
    'display_docx_content',
    'render_sidebar_structure',
    'render_document_info',
    'render_control_panel',
    'render_upload_area',
] 