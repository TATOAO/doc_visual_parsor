import re
from doc_chunking.schemas.schemas import Section
from doc_chunking.utils.llm import get_llm_client

# python -m models.naive_llm.__init__
if __name__ == "__main__":
    from backend.docx_processor import extract_docx_content
    docx_path = "/Users/tatoaoliang/Downloads/Work/doc_visual_parsor/tests/test_data/1-1 买卖合同（通用版）.docx"
    text = extract_docx_content(docx_path)
    section_tree = get_section_tree_by_llm(text)
    print(section_tree)