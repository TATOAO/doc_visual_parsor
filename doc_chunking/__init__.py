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
from .schemas.schemas import Section, Positions, InputDataType, DocumentType

from .layout_structuring.title_structure_builder_llm.layout_displayer import display_layout

from typing import Optional
from pathlib import Path
import io
import logging
import aiofiles

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
    "get_document_type"
]

# Input validation and normalization utilities
def validate_input(input_data: InputDataType, expected_types: Optional[list] = None) -> bool:
    """
    Validate input data for document processing.
    
    Args:
        input_data: Input data to validate
        expected_types: List of expected document types (e.g., ['pdf', 'docx'])
        
    Returns:
        bool: True if input is valid
        
    Raises:
        ValueError: If input is invalid with descriptive error message
    """
    if input_data is None:
        raise ValueError("Input data cannot be None")
    
    # Validate file path inputs
    if isinstance(input_data, (str, Path)):
        file_path = Path(input_data)
        
        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        # Check file extension
        ext = file_path.suffix.lower()
        if ext not in ['.pdf', '.docx', '.doc', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        # Check expected types if specified
        if expected_types:
            doc_type = 'pdf' if ext == '.pdf' else 'docx' if ext in ['.docx', '.doc'] else 'image'
            if doc_type not in expected_types:
                raise ValueError(f"Expected {expected_types}, but got {doc_type} file")
    
    # Validate bytes input
    elif isinstance(input_data, bytes):
        if len(input_data) == 0:
            raise ValueError("Empty bytes data provided")
    
    # Validate file-like objects
    elif hasattr(input_data, 'read'):
        if not callable(getattr(input_data, 'read')):
            raise ValueError("Object has 'read' attribute but it's not callable")
    
    elif hasattr(input_data, 'getvalue'):
        if not callable(getattr(input_data, 'getvalue')):
            raise ValueError("Object has 'getvalue' attribute but it's not callable")
        try:
            data = input_data.getvalue()
            if len(data) == 0:
                raise ValueError("Empty data in BytesIO object")
        except Exception as e:
            raise ValueError(f"Failed to read data from BytesIO object: {e}")
    
    else:
        # Check for numpy array or PIL Image (if available)
        input_type = type(input_data).__name__
        if input_type not in ['ndarray', 'Image']:
            raise ValueError(f"Unsupported input data type: {input_type}")
    
    return True

def normalize_input(input_data: InputDataType) -> InputDataType:
    """
    Normalize input data to a consistent format for processing.
    
    Args:
        input_data: Input data to normalize
        
    Returns:
        InputDataType: Normalized input data
    """
    # Convert string paths to Path objects for consistent handling
    if isinstance(input_data, str):
        return Path(input_data)
    
    # Reset file-like objects to beginning if they have seek method
    if hasattr(input_data, 'seek') and hasattr(input_data, 'read'):
        try:
            input_data.seek(0)
        except (OSError, io.UnsupportedOperation):
            # Some file-like objects don't support seek
            pass
    
    return input_data

def get_document_type(input_data: InputDataType) -> DocumentType:
    """
    Determine document type from input data.
    
    Args:
        input_data: Input data to analyze
        
    Returns:
        DocumentType: Detected document type
        
    Raises:
        ValueError: If document type cannot be determined
    """
    if isinstance(input_data, (str, Path)):
        ext = Path(input_data).suffix.lower()
        if ext == '.pdf':
            return DocumentType.PDF
        elif ext in ['.docx', '.doc']:
            return DocumentType.DOCX
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            return DocumentType.HTML  # Using HTML as placeholder for images
        else:
            raise ValueError(f"Cannot determine document type from extension: {ext}")
    
    elif isinstance(input_data, bytes):
        # Try to detect PDF magic bytes
        if input_data.startswith(b'%PDF'):
            return DocumentType.PDF
        # Try to detect DOCX magic bytes (ZIP format)
        elif input_data.startswith(b'PK'):
            return DocumentType.DOCX
        else:
            raise ValueError("Cannot determine document type from bytes data")
    
    elif hasattr(input_data, 'read') or hasattr(input_data, 'getvalue'):
        # For file-like objects, we can't easily determine type without reading
        # Assume PDF for now (most common use case)
        return DocumentType.PDF
    
    else:
        raise ValueError(f"Cannot determine document type from input: {type(input_data)}")

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

def get_pdf_text(pdf_input: InputDataType) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_input: Path to PDF file or file object
        
    Returns:
        str: Extracted text content
        
    Raises:
        ValueError: If input is invalid
        RuntimeError: If text extraction fails
    """
    try:
        # Validate and normalize input
        validate_input(pdf_input, expected_types=['pdf'])
        pdf_input = normalize_input(pdf_input)
        
        extractor = get_pdf_extractor()
        layout_result = extractor.detect(pdf_input)
        
        # Combine all text from layout elements
        text_parts = []
        for element in layout_result.elements:
            if element.text and element.text.strip():
                text_parts.append(element.text.strip())
        
        if not text_parts:
            logger.warning("No text content found in PDF")
            return ""
        
        return '\n'.join(text_parts)
        
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise RuntimeError(f"Text extraction failed: {e}") from e

def get_word_text(docx_input: InputDataType) -> str:
    """
    Extract text content from a DOCX file.
    
    Args:
        docx_input: Path to DOCX file or file object
        
    Returns:
        str: Extracted text content
        
    Raises:
        ValueError: If input is invalid
        RuntimeError: If text extraction fails
    """
    try:
        # Validate and normalize input
        validate_input(docx_input, expected_types=['docx'])
        docx_input = normalize_input(docx_input)
        
        extractor = get_docx_extractor()
        layout_result = extractor.detect(docx_input)
        
        # Combine all text from layout elements
        text_parts = []
        for element in layout_result.elements:
            if element.text and element.text.strip():
                text_parts.append(element.text.strip())
        
        if not text_parts:
            logger.warning("No text content found in DOCX")
            return ""
        
        return '\n'.join(text_parts)
        
    except Exception as e:
        logger.error(f"Failed to extract text from DOCX: {e}")
        raise RuntimeError(f"Text extraction failed: {e}") from e

def quick_cv_mix_pdf_chunking(pdf_input: InputDataType, **kwargs) -> Section:
    """
    Enhanced quick chunking workflow for PDF files with comprehensive input validation.
    
    Args:
        pdf_input: Path to PDF file, file object, or bytes data
        **kwargs: Additional arguments for chunking including:
            - confidence_threshold: Detection confidence threshold
            - image_size: Input image size for CV model
            - pdf_dpi: DPI for PDF to image conversion
            - validate_input: Whether to validate input (default: True)
        
    Returns:
        Section: Hierarchical section tree
        
    Raises:
        ValueError: If input is invalid or not a PDF
        RuntimeError: If processing fails
    """
    try:
        # Input validation (can be disabled for performance)
        if kwargs.get('validate_input', True):
            validate_input(pdf_input, expected_types=['pdf'])
            pdf_input = normalize_input(pdf_input)
            
            # Log input information
            if isinstance(pdf_input, Path):
                logger.info(f"Processing PDF file: {pdf_input} (size: {pdf_input.stat().st_size} bytes)")
            else:
                logger.info(f"Processing PDF from {type(pdf_input).__name__}")
        
        # Extract layout using CV-enhanced extractor
        extractor = get_pdf_cv_extractor()
        layout_result = extractor.detect(pdf_input)
        
        if not layout_result.elements:
            logger.warning("No layout elements detected in PDF")
            return Section(title="Empty Document", content="", level=0)
        
        logger.info(f"Detected {len(layout_result.elements)} layout elements")
        
        # Use LLM to build title structure
        title_structure_builder = get_title_structure_builder()
        title_structure = title_structure_builder(layout_result)
        
        # Reconstruct sections
        section_reconstructor = get_section_reconstructor()
        result = section_reconstructor(title_structure, layout_result)
        
        logger.info("PDF chunking completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"PDF chunking failed: {e}")
        raise RuntimeError(f"Failed to process PDF: {e}") from e

def quick_docx_chunking(docx_input: InputDataType, **kwargs) -> Section:
    """
    Enhanced quick chunking workflow for DOCX files with comprehensive input validation.
    
    Args:
        docx_input: Path to DOCX file, file object, or bytes data
        **kwargs: Additional arguments for chunking including:
            - validate_input: Whether to validate input (default: True)
        
    Returns:
        Section: Hierarchical section tree
        
    Raises:
        ValueError: If input is invalid or not a DOCX
        RuntimeError: If processing fails
    """
    try:
        # Input validation (can be disabled for performance)
        if kwargs.get('validate_input', True):
            validate_input(docx_input, expected_types=['docx'])
            docx_input = normalize_input(docx_input)
            
            # Log input information
            if isinstance(docx_input, Path):
                logger.info(f"Processing DOCX file: {docx_input} (size: {docx_input.stat().st_size} bytes)")
            else:
                logger.info(f"Processing DOCX from {type(docx_input).__name__}")
        
        # Extract layout using DOCX extractor
        extractor = get_docx_extractor()
        layout_result = extractor.detect(docx_input)
        
        if not layout_result.elements:
            logger.warning("No layout elements detected in DOCX")
            return Section(title="Empty Document", content="", level=0)
        
        logger.info(f"Detected {len(layout_result.elements)} layout elements")
        
        # Use LLM to build title structure  
        title_structure_builder = get_title_structure_builder()
        title_structure = title_structure_builder(layout_result)
        
        # Reconstruct sections
        section_reconstructor = get_section_reconstructor()
        result = section_reconstructor(title_structure, layout_result)
        
        logger.info("DOCX chunking completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"DOCX chunking failed: {e}")
        raise RuntimeError(f"Failed to process DOCX: {e}") from e

# python -m doc_chunking.__init__
if __name__ == "__main__":
    import asyncio
    import aiofiles
    
    async def test_pdf_chunking():
        # set logger level to debug
        logging.basicConfig(level=logging.INFO)
        pdf_input = "/Users/tatoaoliang/Downloads/金蝶/合同/第一编/1-1 买卖合同（通用版）.pdf"
        
        # Test with UploadFile
        print("\nTesting with UploadFile:")
        async with aiofiles.open(pdf_input, "rb") as file:
            file_content = await file.read()
            # Create a BytesIO object with the PDF content
            pdf_bytes = io.BytesIO(file_content)
            result2 = quick_cv_mix_pdf_chunking(pdf_bytes)
            print(result2)

    # Run the async test
    asyncio.run(test_pdf_chunking())