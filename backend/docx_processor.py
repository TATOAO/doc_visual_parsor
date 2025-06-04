import docx
import tempfile
import os
import streamlit as st


def extract_docx_content(docx_file):
    """Extract content from DOCX file"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            tmp_file.write(docx_file.getvalue())
            tmp_file_path = tmp_file.name
        
        doc = docx.Document(tmp_file_path)
        content = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                content.append(paragraph.text)
        
        os.unlink(tmp_file_path)  # Clean up temp file
        return content
        
    except Exception as e:
        st.error(f"Error processing DOCX: {str(e)}")
        return []


def extract_docx_structure(docx_file):
    """Extract document structure from DOCX file based on styles"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            tmp_file.write(docx_file.getvalue())
            tmp_file_path = tmp_file.name
        
        doc = docx.Document(tmp_file_path)
        structure = []
        
        for i, paragraph in enumerate(doc.paragraphs):
            if paragraph.text.strip():
                # Check if paragraph has a heading style
                style_name = paragraph.style.name.lower()
                
                if 'heading' in style_name:
                    # Extract heading level from style name
                    level = 1
                    if 'heading 1' in style_name:
                        level = 1
                    elif 'heading 2' in style_name:
                        level = 2
                    elif 'heading 3' in style_name:
                        level = 3
                    elif 'heading 4' in style_name:
                        level = 4
                    else:
                        # Try to extract number from style name
                        import re
                        match = re.search(r'heading (\d+)', style_name)
                        if match:
                            level = min(int(match.group(1)), 6)
                    
                    structure.append({
                        'text': paragraph.text.strip(),
                        'level': level,
                        'paragraph_index': i,
                        'style': style_name
                    })
        
        os.unlink(tmp_file_path)  # Clean up temp file
        return structure
        
    except Exception as e:
        st.error(f"Error extracting DOCX structure: {str(e)}")
        return [] 