# given a "Section Tree Object" (which extracted from the raw text) find the position of the section title in the raw text

from models.utils.schemas import Section
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

    return matches

def flatten_section_tree_to_tokens(section_tree: Section) -> List[Section]:
    """
    Flatten the section tree into a flat list of Section objects, with the root section at the first position
    """
    def _flatten_section(section: Section) -> List[Section]:
        sections = [section]  # Start with the current section
        
        # Recursively add all subsections
        for sub_section in section.sub_sections:
            sections.extend(_flatten_section(sub_section))
        
        return sections
    
    # Start with root sections
    if section_tree.sub_sections:
        all_sections = [section_tree]
        for section in section_tree.sub_sections:
            all_sections.extend(_flatten_section(section))
        return all_sections
    else:
        # If no subsections, treat the section itself as the root
        return _flatten_section(section_tree)

def set_section_position_index(section_tree: Section, raw_text: str) -> str:
    """
    Set the position index of the section title and content in the raw text
    """

    sections = flatten_section_tree_to_tokens(section_tree)
    title_list = [section.title for section in sections]

    matches = find_ordered_fuzzy_sequence(raw_text, title_list)
    
    # First pass: set all title positions
    for i, (section, match) in enumerate(zip(sections, matches)):
        section.title_position_index = match
    
    # Second pass: set content positions based on hierarchical structure
    def set_content_positions(section: Section, flattened_sections: List[Section]) -> None:
        """Recursively set content positions for sections and their subsections"""
        
        if section.sub_sections:
            # Section has subsections - process them first
            for sub_section in section.sub_sections:
                set_content_positions(sub_section, flattened_sections)
            
            # Content spans from title end to last subsection's content end
            last_subsection = section.sub_sections[-1]
            section.content_position_index = (
                section.title_position_index[1] + 1, 
                last_subsection.content_position_index[1]
            )
        else:
            # Section has no subsections - find next sibling or next section in hierarchy
            current_idx = flattened_sections.index(section)
            
            if current_idx < len(flattened_sections) - 1:
                next_section = flattened_sections[current_idx + 1]
                section.content_position_index = (
                    section.title_position_index[1] + 1, 
                    next_section.title_position_index[0] - 1
                )
            else:
                # Last section overall
                section.content_position_index = (
                    section.title_position_index[1] + 1, 
                    len(raw_text)
                )
    
    # Start from root and process hierarchically
    set_content_positions(section_tree, sections)

    return section_tree

# python -m models.naive_llm.tree_like_structure_mapping
if __name__ == "__main__":

    from backend.docx_processor import extract_docx_content
    docx_path = "/Users/tatoaoliang/Downloads/Work/doc_visual_parsor/tests/test_data/1-1 买卖合同（通用版）.docx"
    raw_text = extract_docx_content(docx_path)

    import json 
    with open('section_tree.json', 'r') as f:
        j = json.load(f)

    section_tree = Section.model_validate(j)

    section_tree = set_section_position_index(section_tree, raw_text)

    for section in flatten_section_tree_to_tokens(section_tree):
        print(section.level, section.title, section.title_position_index, section.content_position_index)

    print('end')
