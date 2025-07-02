import docx
import tempfile
import os
from typing import Union, Any
from pathlib import Path

def extract_docx_content(docx_file: Union[str, bytes, Path, Any]) -> str:
    """Extract content from DOCX file"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            if isinstance(docx_file, bytes):
                tmp_file.write(docx_file)
            elif isinstance(docx_file, str):
                with open(docx_file, "rb") as f:
                    tmp_file.write(f.read())
            elif isinstance(docx_file, Path):
                with open(docx_file, "rb") as f:
                    tmp_file.write(f.read())
            elif hasattr(docx_file, 'getvalue'):
                # Handle uploaded file objects (e.g., MockUploadedFile)
                tmp_file.write(docx_file.getvalue())
            else:
                raise ValueError(f"Unsupported file type: {type(docx_file)}")
            tmp_file_path = tmp_file.name
        
        doc = docx.Document(tmp_file_path)
        content = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                content.append(paragraph.text)
        
        os.unlink(tmp_file_path)  # Clean up temp file
        return "\n\n".join(content)
        
    except Exception as e:
        print(f"Error processing DOCX: {str(e)}")
        return ""


# python -m backend.docx_processor
if __name__ == "__main__":
    docx_path = "/Users/tatoaoliang/Downloads/Work/doc_visual_parsor/tests/test_data/1-1 买卖合同（通用版）.docx"
    text = extract_docx_content(docx_path)
    print(text)