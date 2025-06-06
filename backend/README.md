# Backend Module Documentation

## Overview

The backend module contains the core functionality for the Document Visual Parser application, organized into logical components for better maintainability and code organization.

## Module Structure

### 📄 `pdf_processor.py`

Handles all PDF-related processing operations:

- `extract_pdf_pages_into_images(pdf_file)` - Extract PDF pages as images for display
- `get_pdf_document_object(pdf_file)` - Get PyMuPDF document object for advanced operations
- `close_pdf_document(pdf_doc)` - Safely close PDF document objects

### 📝 `docx_processor.py`

Handles all DOCX document processing:

- `extract_docx_content(docx_file)` - Extract text content from DOCX files
- `extract_docx_structure(docx_file)` - Extract document structure based on heading styles

### 🔍 `document_analyzer.py`

Provides document structure analysis capabilities:

- `extract_pdf_document_structure(pdf_file)` - Extract headings and structure from PDF
- `analyze_document_structure(uploaded_file)` - Universal document structure analyzer
- `get_structure_summary(structure)` - Generate summary of document structure

### 🔧 `session_manager.py`

Manages Streamlit session state and navigation:

- `initialize_session_state()` - Initialize all session variables
- `reset_document_state()` - Reset state when loading new documents
- `is_new_file(uploaded_file)` - Check if a new file has been uploaded
- `navigate_to_section(page_num)` - Navigate to specific document sections
- `get_document_info()` - Get current document information

### 🎨 `ui_components.py`

Contains all UI rendering and display components:

- `check_pdf_viewer_availability()` - Check for PDF viewer plugin
- `display_pdf_viewer(pages)` - Display PDF using image-based viewer
- `display_pdf_with_viewer(pdf_file)` - Display PDF using advanced viewer
- `display_docx_content(content)` - Display DOCX content
- `render_sidebar_structure(document_structure)` - Render document structure in sidebar
- `render_document_info(doc_info)` - Render document information
- `render_control_panel(uploaded_file)` - Render control panel
- `render_upload_area()` - Render file upload area

## Usage

The backend modules are imported and used in the main `app.py` file:

```python
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
    display_pdf_viewer,
    render_sidebar_structure,
    # ... etc
)
```

## Benefits of This Structure

1. **Separation of Concerns**: Each module handles a specific aspect of functionality
2. **Maintainability**: Easier to locate and modify specific features
3. **Reusability**: Functions can be easily reused across different parts of the application
4. **Testing**: Each module can be tested independently
5. **Scalability**: New features can be added to appropriate modules without cluttering the main file

## Dependencies

The backend modules require the following packages:

- `streamlit` - Web framework
- `PyMuPDF` (fitz) - PDF processing
- `python-docx` - DOCX processing
- `Pillow` (PIL) - Image processing
- `streamlit-pdf-viewer` (optional) - Enhanced PDF viewing

## Error Handling

Each module includes appropriate error handling and user feedback through Streamlit's messaging system.
