# Document processors for PDF and DOCX files

from .pdf_processor import extract_pdf_pages_into_images, get_pdf_document_object, close_pdf_document
from .docx_processor import extract_docx_content

__all__ = [
    'extract_pdf_pages_into_images',
    'get_pdf_document_object',
    'close_pdf_document',
    'extract_docx_content',
] 