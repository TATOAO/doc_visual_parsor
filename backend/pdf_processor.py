import fitz  # PyMuPDF
from PIL import Image
import io
import tempfile
import os
"""
This module is responsible for extracting pages from a PDF file and converting them to images.
It also provides a function to get a PyMuPDF document object for advanced operations.
It also provides a function to close a PDF document.

- extract_pdf_pages: Extract pages from PDF as images
- get_pdf_document_object: Get PyMuPDF document object for advanced operations
- close_pdf_document: Safely close PDF document
"""


def extract_pdf_pages(pdf_file):
    """Extract pages from PDF as images"""
    pages = []
    pdf_doc = None
    tmp_file_path = None
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Open PDF with PyMuPDF
        pdf_doc = fitz.open(tmp_file_path)
        
        # Process each page
        for page_num in range(pdf_doc.page_count):
            try:
                page = pdf_doc.load_page(page_num)
                
                # Convert page to image with good quality
                mat = fitz.Matrix(1.5, 1.5)  # 1.5x zoom for good quality/performance balance
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                pages.append(img)
                
            except Exception as page_error:
                print(f"Error processing page {page_num + 1}: {str(page_error)}")
                continue
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        
    finally:
        # Clean up resources
        if pdf_doc:
            pdf_doc.close()
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up temp file: {cleanup_error}")
    
    return pages


def get_pdf_document_object(pdf_file):
    """Get PyMuPDF document object for advanced operations"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Open PDF with PyMuPDF
        pdf_doc = fitz.open(tmp_file_path)
        
        # Clean up temp file (keep pdf_doc open)
        os.unlink(tmp_file_path)
        
        return pdf_doc
        
    except Exception as e:
        print(f"Error opening PDF document: {str(e)}")
        return None


def close_pdf_document(pdf_doc):
    """Safely close PDF document"""
    if pdf_doc:
        try:
            pdf_doc.close()
        except Exception as e:
            print(f"Warning: Error closing PDF document: {str(e)}") 


__all__ = ["extract_pdf_pages", "get_pdf_document_object", "close_pdf_document"]