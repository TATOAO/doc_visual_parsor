import docx
import tempfile
import os
from typing import Union, Any, List
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


def extract_docx_structure_with_naive_llm(docx_file: Union[str, bytes, Path, Any]) -> dict:
    """Extract document structure from DOCX file using naive_llm method"""
    try:
        # Import here to avoid circular imports
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        
        # TODO: use control for finer cut
        from models.naive_llm import get_section_tree_by_llm
        from models.naive_llm.helpers import generate_section_tree_from_tokens, set_section_position_index
        
        # Extract text content
        raw_text = extract_docx_content(docx_file)
        if not raw_text:
            return {"success": False, "error": "Failed to extract text content from DOCX"}
        
        # Step 1: Use LLM to insert section tokens
        llm_response = get_section_tree_by_llm(raw_text)
        if not llm_response:
            return {"success": False, "error": "Failed to get LLM response"}
        
        # Step 2: Parse tokens into section tree
        section_tree = generate_section_tree_from_tokens(llm_response)
        if not section_tree:
            return {"success": False, "error": "Failed to parse section tree from tokens"}
        
        # Step 3: Set position indices
        section_tree = set_section_position_index(section_tree, raw_text)
        
        # Step 4: Remove circular references for JSON serialization
        remove_circular_references(section_tree)
        
        return {
            "success": True,
            "section_tree": section_tree.model_dump(),
            "raw_text": raw_text,
            "llm_annotated_text": llm_response
        }
        
    except Exception as e:
        print(f"Error processing DOCX with naive_llm: {str(e)}")
        return {"success": False, "error": str(e)}


def extract_docx_structure(docx_file: Union[str, bytes, Path, Any]):
    """Extract document structure from DOCX file based on styles"""
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
        print(f"Error extracting DOCX structure: {str(e)}")
        return [] 

# python -m backend.docx_processor
if __name__ == "__main__":
    docx_path = "/Users/tatoaoliang/Downloads/Work/doc_visual_parsor/tests/test_data/1-1 买卖合同（通用版）.docx"
    text = extract_docx_content(docx_path)
    print(text)