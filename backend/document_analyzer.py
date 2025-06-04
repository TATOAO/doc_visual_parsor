import re
import streamlit as st
from typing import List, Dict
from .pdf_processor import get_pdf_document_object

"""
This module is responsible for analyzing the structure of a document.
It provides a function to extract the structure of a PDF document.
It also provides a function to analyze the structure of a DOCX document.
It also provides a function to get a summary of the document structure.

- extract_pdf_document_structure: Extract document structure (headings, titles) from PDF
- analyze_document_structure: Analyze document structure based on file type
- get_structure_summary: Get a summary of the document structure
"""


def extract_pdf_document_structure(pdf_file: bytes) -> List[Dict]:
    """
    Extract document structure (headings, titles) from PDF

    Args:
        pdf_file: bytes - The PDF file to extract structure from

    Returns:
        List[Dict] - The document structure
    
        sample:
            [
                {
                    "text": "Chapter 1",
                    "page": 1,
                    "level": 1,
                    "font_size": 18,
                    "y_position": 100
                },
                {
                    "text": "Section 1.1",
                    "page": 1,
                    "level": 2,
                    "font_size": 16,
                    "y_position": 150
                }
            ]
    """
    structure = []
    pdf_doc = None
    
    try:
        # Get PDF document object
        pdf_doc = get_pdf_document_object(pdf_file)
        if not pdf_doc:
            return structure
        
        # Store pdf_doc in session state for navigation
        st.session_state.pdf_document = pdf_doc
        
        # Extract text blocks with formatting information
        for page_num in range(pdf_doc.page_count):
            try:
                page = pdf_doc.load_page(page_num)
                blocks = page.get_text("dict")
                
                for block in blocks["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                font_size = span["size"]
                                font_flags = span["flags"]  # Bold, italic, etc.
                                
                                # Skip empty text or very short text
                                if len(text) < 3:
                                    continue
                                
                                # Identify potential headings based on font size and formatting
                                is_heading = False
                                heading_level = 0
                                
                                # Check if text looks like a heading
                                if (font_size > 12 and  # Larger font size
                                    (font_flags & 2**4 or  # Bold
                                     font_flags & 2**6) and  # Superscript/emphasis
                                    len(text) < 100 and  # Not too long
                                    not text.endswith('.') and  # Not ending with period
                                    any(c.isupper() for c in text)):  # Contains uppercase
                                    
                                    is_heading = True
                                    # Determine heading level based on font size
                                    if font_size >= 18:
                                        heading_level = 1
                                    elif font_size >= 16:
                                        heading_level = 2
                                    elif font_size >= 14:
                                        heading_level = 3
                                    else:
                                        heading_level = 4
                                
                                # Also check for common heading patterns
                                heading_patterns = [
                                    r'^Chapter\s+\d+',
                                    r'^\d+\.\s+',
                                    r'^\d+\.\d+\s+',
                                    r'^[A-Z][A-Z\s]{5,}$',  # ALL CAPS headings
                                    r'^[A-Z][a-z]+\s+[A-Z][a-z]+',  # Title Case
                                ]
                                
                                for pattern in heading_patterns:
                                    if re.match(pattern, text):
                                        is_heading = True
                                        if not heading_level:
                                            heading_level = 2
                                        break
                                
                                if is_heading:
                                    structure.append({
                                        'text': text,
                                        'page': page_num,
                                        'level': heading_level,
                                        'font_size': font_size,
                                        'y_position': span.get('bbox', [0, 0, 0, 0])[1]  # Y coordinate for position
                                    })
                                    
            except Exception as page_error:
                continue
        
        # Remove duplicates and sort by page and position
        seen = set()
        unique_structure = []
        for item in structure:
            key = (item['text'], item['page'])
            if key not in seen:
                seen.add(key)
                unique_structure.append(item)
        
        # Sort by page number and then by Y position (top to bottom)
        unique_structure.sort(key=lambda x: (x['page'], -x['y_position']))
        
    except Exception as e:
        st.error(f"❌ Error extracting document structure: {str(e)}")
        
    return unique_structure


def analyze_document_structure(uploaded_file):
    """Analyze document structure based on file type"""
    if uploaded_file.type == "application/pdf":
        return extract_pdf_document_structure(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        from .docx_processor import extract_docx_structure
        docx_structure = extract_docx_structure(uploaded_file)
        # Convert DOCX structure to match PDF structure format
        converted_structure = []
        for item in docx_structure:
            converted_structure.append({
                'text': item['text'],
                'page': 0,  # DOCX doesn't have pages in the same way
                'level': item['level'],
                'paragraph_index': item['paragraph_index']
            })
        return converted_structure
    else:
        st.warning(f"Document structure analysis not supported for file type: {uploaded_file.type}")
        return []


def get_structure_summary(structure):
    """Get a summary of the document structure"""
    if not structure:
        return "No structure detected"
    
    level_counts = {}
    for item in structure:
        level = item['level']
        level_counts[level] = level_counts.get(level, 0) + 1
    
    summary_parts = []
    level_names = {1: "Main Headings", 2: "Sections", 3: "Subsections", 4: "Sub-subsections"}
    
    for level in sorted(level_counts.keys()):
        count = level_counts[level]
        name = level_names.get(level, f"Level {level} headings")
        summary_parts.append(f"{count} {name}")
    
    return ", ".join(summary_parts) 

__all__ = ["extract_pdf_document_structure", "analyze_document_structure", "get_structure_summary"]


# python -m backend.document_analyzer
if __name__ == "__main__":
    with open("tests/test_data/1-1 买卖合同（通用版）.pdf", "rb") as f:
        print(extract_pdf_document_structure(f))