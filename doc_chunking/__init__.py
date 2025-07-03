"""
Doc Chunking Library

A powerful document analysis and chunking library for PDF and DOCX files using AI-powered processing.
Provides document layout detection, structure analysis, and intelligent chunking capabilities.
"""
__version__ = "0.1.0"
__author__ = "TATOAO"
__email__ = "tatoao@126.com"

# Import essential schemas that are least likely to have circular dependencies
from .schemas.layout_schemas import LayoutExtractionResult, LayoutElement
from .schemas.schemas import Section, Positions

from .layout_structuring.title_structure_builder_llm.layout_displayer import display_layout
from .documents_chunking.chunker import Chunker

import logging

logger = logging.getLogger(__name__)

# Make commonly used classes and functions available at package level
__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    
    # Schema classes
    "LayoutExtractionResult",
    "LayoutElement", 
    "Section",
    "Positions",
    
    # Convenience functions
    "display_layout",
    "get_pdf_extractor",
    "get_docx_extractor",
    "get_pdf_cv_extractor",
    "get_pdf_text",
    "get_word_text",
    "quick_cv_mix_pdf_chunking",
    "quick_docx_chunking",
    
    # Input validation utilities
    "validate_input",
    "normalize_input",
    "get_document_type",

    # Chunker
    "Chunker"
]

# Doc Chunking Package - Document Analysis and Chunking Library

from .api import app as fastapi_app
from .documents_chunking.chunker import Chunker
from .processors import extract_pdf_pages_into_images, extract_docx_content
from .schemas.schemas import Section

__version__ = "0.1.0"
__author__ = "Doc Chunking Team"
__description__ = "A powerful document analysis and chunking library for PDF and DOCX files using AI-powered processing"

# Export main components for easy importing
__all__ = [
    'fastapi_app',
    'Chunker',
    'extract_pdf_pages_into_images', 
    'extract_docx_content',
    'Section',
]

# For backward compatibility and easy access
app = fastapi_app