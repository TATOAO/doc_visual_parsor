# given a tree like title structure (which extracted from the raw text), 
# we now need to insert/apply the highlight (with special index token) into the raw text, which is like a middle step of the section token parsor
# I want the process is purely based on text processing without models

from models.utils.schemas import Section
import re
from difflib import SequenceMatcher
from typing import List, Tuple

from fuzzysearch import find_near_matches

def find_ordered_fuzzy_sequence(text: str, keywords: List[str], max_l_dist: int = 3) -> List[Tuple[int, int]]:
    """
    Search for keywords in order with fuzzy matching. Return list of (start, end) index for each match.

    ## TODO: This is a naive implementation, it failes when the keywords are not unique in the raw text
    """
    matches = []
    cursor = 0  # Where to start search in the text

    for keyword in keywords:
        # Search for keyword in remaining text with fuzzy match
        sub_text = text[cursor:]
        result = find_near_matches(keyword, sub_text, max_l_dist=max_l_dist)
        if not result:
            return []  # One keyword not found in order

        # Choose first match (closest in order)
        match = result[0]
        start, end = match.start + cursor, match.end + cursor
        matches.append((start, end))
        cursor = end  # Move cursor forward
        print(matches)

    return matches

def flatten_section_tree_to_tokens(section_tree: Section) -> List[str]:
    """
    Flatten the section tree into a list of sections
    <start-section-title-1>
    <end-section-title-1>
    <start-section-content-1>
    <start-section-title-1-1>
    <end-section-title-1-1>
    <start-section-content-1-1>
    <end-section-content-1-1>

    <start-section-title-1-2>
    1.2 另一个子标题
    <end-section-title-1-2>
    <start-section-content-1-2>
    这里是1.2节的具体内容...
    <end-section-content-1-2>

    <end-section-content-1>

    <start-section-title-2>
    第二章 标题
    <end-section-title-2>
    <start-section-content-2>
    这里是第二章的内容...
    <end-section-content-2>

    """
    def _flatten_section(section: Section, section_id: str) -> List[str]:
        tokens = []
        
        # Add section title tokens
        if section.title:
            tokens.append(f"<start-section-title-{section_id}>")
            tokens.append(section.title)
            tokens.append(f"<end-section-title-{section_id}>")
        
        # Add section content start token
        tokens.append(f"<start-section-content-{section_id}>")
        
        # Add section content (if any)
        if section.content:
            tokens.append(section.content)
        
        # Process all subsections recursively
        for i, sub_section in enumerate(section.sub_sections, 1):
            sub_section_id = f"{section_id}-{i}"
            tokens.extend(_flatten_section(sub_section, sub_section_id))
        
        # Add section content end token (after all subsections)
        tokens.append(f"<end-section-content-{section_id}>")
        
        return tokens
    
    # Start with root sections
    if section_tree.sub_sections:
        all_tokens = []
        for i, section in enumerate(section_tree.sub_sections, 1):
            all_tokens.extend(_flatten_section(section, str(i)))
        return all_tokens
    else:
        # If no subsections, treat the section itself as the root
        return _flatten_section(section_tree, "1")

def insert_tokens_into_raw_text_with_given_section_tree(section_tree: Section, raw_text: str) -> str:
    """
    Insert section tokens into raw text based on the section tree structure
    """


    pass

# python -m models.naive_llm.tree_like_structure_mapping
if __name__ == "__main__":

    # from backend.docx_processor import extract_docx_content
    # docx_path = "/Users/tatoaoliang/Downloads/Work/doc_visual_parsor/tests/test_data/1-1 买卖合同（通用版）.docx"
    # raw_text = extract_docx_content(docx_path)


#     raw_titles = """第一章 买卖合同:
# 第一节 买卖合同（通用版）:
# 一、定义
# 二、主要风险及常见易发问题提示:
# （一）主要业务风险
# （二）合同管理常见易发问题
# 三、风险防范措施
# 四、相关法律法规:
# 《民法典》
# 《最高人民法院关于审理买卖合同纠纷案件适用法律问题的解释》
# 买卖合同（通用版）"""

#     title_list = [t.strip() for t in raw_titles.split('\n')]
#     matches = find_ordered_fuzzy_sequence(raw_text, title_list)
#     print(matches)



    import json 
    with open('section_tree.json', 'r') as f:
        j = json.load(f)

    section_tree = Section.model_validate(j)

    tokens = flatten_section_tree_to_tokens(section_tree)
    print(tokens)

    # Now we can use the implemented function
    # import ipdb; ipdb.set_trace()
    # result = insert_tokens_into_raw_text_with_given_section_tree(section_tree, raw_text)
    # print(result)

    print('end')
