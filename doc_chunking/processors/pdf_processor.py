import io
import fitz  # PyMuPDF
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def extract_pdf_pages_into_images(uploaded_file):
    """
    Extract pages from a PDF file and convert them to PIL Image objects.
    
    Args:
        uploaded_file: File-like object with PDF content
        
    Returns:
        List of PIL Image objects, one for each page
    """
    logger.info("Starting PDF page extraction")
    
    try:
        # Get PDF content
        if hasattr(uploaded_file, 'getvalue'):
            pdf_content = uploaded_file.getvalue()
        elif hasattr(uploaded_file, 'read'):
            pdf_content = uploaded_file.read()
        else:
            pdf_content = uploaded_file
            
        # Open PDF from memory
        pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")
        page_images = []
        
        logger.info(f"PDF has {len(pdf_doc)} pages")
        
        for page_num in range(len(pdf_doc)):
            logger.debug(f"Processing page {page_num + 1}")
            
            # Get page
            page = pdf_doc[page_num]
            
            # Convert to image with high resolution
            mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            page_images.append(img)
            
        pdf_doc.close()
        logger.info(f"Successfully extracted {len(page_images)} pages")
        return page_images
        
    except Exception as e:
        logger.error(f"Error extracting PDF pages: {str(e)}")
        return []


def get_pdf_document_object(uploaded_file):
    """
    Get a PyMuPDF document object from an uploaded file.
    
    Args:
        uploaded_file: File-like object with PDF content
        
    Returns:
        PyMuPDF Document object or None if error
    """
    try:
        if hasattr(uploaded_file, 'getvalue'):
            pdf_content = uploaded_file.getvalue()
        elif hasattr(uploaded_file, 'read'):
            pdf_content = uploaded_file.read()
        else:
            pdf_content = uploaded_file
            
        return fitz.open(stream=pdf_content, filetype="pdf")
    except Exception as e:
        logger.error(f"Error opening PDF document: {str(e)}")
        return None


def close_pdf_document(pdf_doc):
    """
    Safely close a PyMuPDF document.
    
    Args:
        pdf_doc: PyMuPDF Document object
    """
    try:
        if pdf_doc:
            pdf_doc.close()
    except Exception as e:
        logger.error(f"Error closing PDF document: {str(e)}")


def extract_pdf_text(uploaded_file):
    """
    Extract text content from a PDF file.
    
    Args:
        uploaded_file: File-like object with PDF content
        
    Returns:
        String with extracted text
    """
    logger.info("Starting PDF text extraction")
    
    try:
        pdf_doc = get_pdf_document_object(uploaded_file)
        if not pdf_doc:
            return ""
            
        text_content = []
        
        for page_num in range(len(pdf_doc)):
            logger.debug(f"Extracting text from page {page_num + 1}")
            page = pdf_doc[page_num]
            text = page.get_text()
            text_content.append(text)
            
        close_pdf_document(pdf_doc)
        
        full_text = "\n\n".join(text_content)
        logger.info(f"Successfully extracted {len(full_text)} characters of text")
        return full_text
        
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        return "" 