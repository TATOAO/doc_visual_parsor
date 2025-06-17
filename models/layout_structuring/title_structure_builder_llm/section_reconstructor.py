from models.layout_structuring.title_structure_builder_llm.structurer_llm import title_structure_builder_llm
from models.layout_structuring.title_structure_builder_llm.layout_displayer import display_layout
from models.naive_llm.helpers.tree_like_structure_mapping import find_ordered_fuzzy_sequence
from models.schemas.layout_schemas import LayoutExtractionResult
from models.schemas.schemas import Section, Positions



def section_reconstructor(title_raw_structure: str, layout_extraction_result: LayoutExtractionResult) -> Section:
    """
    Building a Section Tree Object from a title_raw_structure and a layout_extraction_result.

    The general idea is first parsing the title structure from the title_raw_structure, then find the position of the title in the layout_extraction_result, and then build the section tree object.

    Args:
        title_raw_structure: A string containing the hierarchical title structure
        layout_extraction_result: A LayoutExtractionResult object containing the document layout

    Returns:
        Section: A hierarchical Section tree object representing the document structure
    """
    # Parse title structure into a list of (level, title) tuples
    title_structure = []
    for line in title_raw_structure.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Extract level from numbering (e.g., "1.1.2" -> level 2)
        parts = line.split(' ', 1)
        if len(parts) != 2:
            continue
            
        number, title = parts
        level = number.count('.')
        title_structure.append((level, title.strip(), number.strip('.')))

    # Create root section
    root_section = Section(
        title="",
        content="",
        level=-1,  # Root level is -1 to make actual sections start at level 0
        element_id=-1
    )
    
    # Keep track of sections at each level for building the hierarchy
    level_sections = {-1: root_section}
    
    # Get layout elements in reading order and convert to display format
    sorted_elements = layout_extraction_result.elements
    layout_lines = display_layout(layout_extraction_result)
    
    # Create mapping from line index to element ID
    line_to_element_id = {}
    for i, line in enumerate(layout_lines):
        if line.startswith('[id:'):
            try:
                element_id = int(line[4:line.index(']')])
                line_to_element_id[i] = element_id
            except ValueError:
                continue
    
    # Map titles to elements
    title_elements = []
    current_element_idx = 0  # Keep track of where we are in sorted_elements
    
    for _, title_text, _ in title_structure:
        found_element = None
        # Search from current position to end
        while current_element_idx < len(sorted_elements):
            element = sorted_elements[current_element_idx]
            # Clean and compare the text
            element_text = element.text.strip() if element.text else ""
            if title_text in element_text:
                found_element = element
                current_element_idx += 1  # Move to next element for next search
                break
            current_element_idx += 1
            
        if found_element:
            title_elements.append(found_element)
    import ipdb; ipdb.set_trace()
    
    if len(title_elements) != len(title_structure):
        print(f"Warning: Found {len(title_elements)} matching elements for {len(title_structure)} titles")
        return root_section
    
    # Build section tree
    for idx, ((level, title_text, number), matching_element) in enumerate(zip(title_structure, title_elements)):
        # Create new section
        section = Section(
            title=title_text,
            content="",
            level=level,
            element_id=matching_element.id if matching_element else -1
        )
        
        # Find parent section (section at previous level)
        parent_level = level - 1
        while parent_level >= -1:
            if parent_level in level_sections:
                parent_section = level_sections[parent_level]
                section.parent_section = parent_section
                parent_section.sub_sections.append(section)
                break
            parent_level -= 1
            
        # Update level_sections map
        level_sections[level] = section
        
        # Extract content between this title and the next
        if matching_element:
            # Find the next title element's index in sorted_elements
            current_element_index = sorted_elements.index(matching_element)
            next_title_index = len(sorted_elements)
            
            if idx + 1 < len(title_elements):
                next_title = title_elements[idx + 1]
                try:
                    next_title_index = sorted_elements.index(next_title)
                except ValueError:
                    pass
            
            # Collect content elements between this title and the next
            content_elements = []
            for i in range(current_element_index + 1, next_title_index):
                element = sorted_elements[i]
                # Skip if this element is a title
                if element in title_elements:
                    continue
                if element.text:
                    content_elements.append(element.text)
            
            section.content = "\n".join(content_elements)
    
    return root_section


# python -m models.layout_structuring.title_structure_builder_llm.section_reconstructor
if __name__ == "__main__":
    # Example usage
    import json
    
    # Load layout extraction result
    with open("./hybrid_extraction_result.json", "r") as f:
        layout_data = json.load(f)
    layout_result = LayoutExtractionResult.model_validate(layout_data)
    
    # Example title structure
    title_structure = """
1. 第一章买卖合同  
1.1. 第一节买卖合同（通用版）  
1.1.1. 一、定义  
1.1.2. 二、主要风险及常见易发问题提示  
1.1.2.1. （一）主要业务风险  
1.1.2.2. （二）合同管理常见易发问题  
1.1.2.3. （三）风险防范措施  
1.1.2.4. （四）相关法律法规  
1.1.2.4.1. 《民法典》  
1.1.2.4.2. 《最高人民法院关于审理买卖合同纠纷案件适用法律问题的解释》  
1.1.2.4.3. 《最高人民法院关于适用<中华人民共和国民法典>合同编通则若干问题的解释》  
1.1.2.4.4. 《标准化法》  
1.1.3. 买卖合同（通用版）  
1.1.4. 附件
    """
    
    # Build section tree
    section_tree = section_reconstructor(title_structure, layout_result)
    from models.naive_llm.helpers.section_token_parsor import remove_circular_references
    remove_circular_references(section_tree)
    
    json.dump(section_tree.model_dump(), open("./section_tree_0617.json", "w"), indent=4, ensure_ascii=False)