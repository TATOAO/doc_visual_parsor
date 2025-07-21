"""
Doc Chunking Library

A powerful document analysis and chunking library for PDF and DOCX files using AI-powered processing.
Provides document layout detection, structure analysis, and intelligent chunking capabilities.
"""
__version__ = "0.1.0"
__author__ = "TATOAO"
__email__ = "tatoao@126.com"

import logging

logger = logging.getLogger(__name__)

# Import essential schemas that are least likely to have circular dependencies
from .schemas.layout_schemas import LayoutExtractionResult, LayoutElement
from .schemas.schemas import Section, Positions

# Import layout and chunking components
from .layout_structuring.title_structure_builder_llm.layout_displayer import display_layout
from .documents_chunking.chunker import Chunker

# Import API components
from .api import app as fastapi_app, router as chunking_router

# Import processors
from .processors import extract_pdf_pages_into_images, extract_docx_content


from .core.processors.bbox_nlp_processor import BboxNLPProcessor
from .core.processors.page_chunker import PdfPageImageSplitterProcessor
from .core.processors.page_image_layout_processor import PageImageLayoutProcessor
from .core.processors.title_structure_processor import TitleStructureProcessor
from .core.processors.rechunking_base_on_title import RechunkingBaseOnTitleProcessor

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
    
    # Core components
    "Chunker",
    "fastapi_app",
    "chunking_router",
    
    # Processors
    "extract_pdf_pages_into_images",
    "extract_docx_content",
    
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


    ## processors
    "BboxNLPProcessor",
    "PdfPageImageSplitterProcessor",
    "PageImageLayoutProcessor",
    "TitleStructureProcessor",
    "RechunkingBaseOnTitleProcessor",
]

# For backward compatibility and easy access
app = fastapi_app