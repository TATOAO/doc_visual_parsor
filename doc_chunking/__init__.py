"""
Doc Chunking Library

A powerful document analysis and chunking library for PDF and DOCX files using AI-powered processing.
Provides document layout detection, structure analysis, and intelligent chunking capabilities.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import essential schemas that are least likely to have circular dependencies
from .schemas.layout_schemas import LayoutExtractionResult, LayoutElement
from .schemas.schemas import Section, Positions

from .layout_structuring.title_structure_builder_llm.layout_displayer import display_layout

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
    "quick_docx_chunking"
]

# Lazy import functions to avoid circular dependencies
def get_pdf_extractor():
    """Get a PDF layout extractor instance."""
    from .layout_detection.layout_extraction.pdf_layout_extractor import PdfLayoutExtractor
    return PdfLayoutExtractor()

def get_pdf_cv_extractor():
    """Get a PDF layout extractor instance."""
    from .layout_detection.layout_extraction.pdf_style_cv_mix_extractor import PdfStyleCVMixLayoutExtractor
    return PdfStyleCVMixLayoutExtractor()

def get_docx_extractor():
    """Get a DOCX layout extractor instance."""
    from .layout_detection.layout_extraction.docx_layout_extractor import DocxLayoutExtrator
    return DocxLayoutExtrator()

def get_section_reconstructor():
    """Get the section reconstructor function."""
    from .layout_structuring.title_structure_builder_llm.section_reconstructor import section_reconstructor
    return section_reconstructor

def get_title_structure_builder():
    """Get the title structure builder function."""
    from .layout_structuring.title_structure_builder_llm.structurer_llm import title_structure_builder_llm
    return title_structure_builder_llm

def get_pdf_text(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        str: Extracted text content
    """
    extractor = get_pdf_extractor()
    layout_result = extractor.detect(pdf_path)
    
    # Combine all text from layout elements
    text_parts = []
    for element in layout_result.elements:
        if element.text and element.text.strip():
            text_parts.append(element.text.strip())
    
    return '\n'.join(text_parts)

def quick_cv_mix_pdf_chunking(pdf_path: str, **kwargs):
    """
    Quick chunking workflow for PDF files.
    
    Args:
        pdf_path: Path to PDF file
        **kwargs: Additional arguments for chunking
        
    Returns:
        Section: Hierarchical section tree
    """
    extractor = get_pdf_cv_extractor()
    layout_result = extractor.detect(pdf_path)
    
    # Use LLM to build title structure
    title_structure_builder = get_title_structure_builder()
    title_structure = title_structure_builder(layout_result)
    
    # Reconstruct sections
    section_reconstructor = get_section_reconstructor()
    return section_reconstructor(title_structure, layout_result)

def quick_docx_chunking(docx_path: str, **kwargs):
    """
    Quick chunking workflow for DOCX files.
    
    Args:
        docx_path: Path to DOCX file
        **kwargs: Additional arguments for chunking
        
    Returns:
        Section: Hierarchical section tree
    """
    extractor = get_docx_extractor()
    layout_result = extractor.detect(docx_path)
    
    # Use LLM to build title structure  
    title_structure_builder = get_title_structure_builder()
    title_structure = title_structure_builder(layout_result)
    
    # Reconstruct sections
    section_reconstructor = get_section_reconstructor()
    return section_reconstructor(title_structure, layout_result)
