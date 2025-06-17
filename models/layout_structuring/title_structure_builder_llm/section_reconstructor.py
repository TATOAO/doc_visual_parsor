from models.layout_structuring.title_structure_builder_llm.structurer_llm import title_structure_builder_llm
from models.layout_structuring.title_structure_builder_llm.layout_displayer import display_layout
from models.naive_llm.helpers.tree_like_structure_mapping import find_ordered_fuzzy_sequence
from models.schemas.layout_schemas import LayoutExtractionResult
from models.schemas.schemas import Section, Positions
from fuzzysearch import find_near_matches



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
    
    def fuzzy_match_title(title_text: str, element_text: str, max_l_dist: int = 3) -> bool:
        """
        Fuzzy match title text against element text using fuzzy search.
        
        Args:
            title_text: The title to search for
            element_text: The element text to search in
            max_l_dist: Maximum Levenshtein distance for fuzzy matching
            
        Returns:
            True if fuzzy match found, False otherwise
        """
        if not title_text or not element_text:
            return False
        
        # Clean both texts by removing extra whitespace and normalizing
        title_clean = " ".join(title_text.strip().split())
        element_clean = " ".join(element_text.strip().split())
        
        # Try exact match first
        if title_clean in element_clean:
            return True
            
        # Try fuzzy matching with the cleaned text
        matches = find_near_matches(title_clean, element_clean, max_l_dist=max_l_dist)
        if matches:
            return True
            
        # Try fuzzy matching with a shorter version of the title (first few words)
        # This helps when titles are truncated or have additional formatting
        title_words = title_clean.split()
        if len(title_words) > 2:
            short_title = " ".join(title_words[:2])
            matches = find_near_matches(short_title, element_clean, max_l_dist=max_l_dist)
            if matches:
                return True
        
        return False
    
    # Map titles to elements
    title_elements = []
    current_element_idx = 0  # Keep track of where we are in sorted_elements
    
    for _, title_text, _ in title_structure:
        found_element = None
        # Search from current position to end
        while current_element_idx < len(sorted_elements):
            element = sorted_elements[current_element_idx]
            # Clean and compare the text using fuzzy matching
            element_text = element.text.strip() if element.text else ""
            if fuzzy_match_title(title_text, element_text):
                found_element = element
                current_element_idx += 1  # Move to next element for next search
                break
            current_element_idx += 1


            if title_text == "附件：":
                import ipdb; ipdb.set_trace()
                print(element_text)
                pass
            
        if found_element:
            title_elements.append(found_element)
    
    if len(title_elements) != len(title_structure):
        print(f"Warning: Found {len(title_elements)} matching elements for {len(title_structure)} titles")
        # Print debug information to help identify matching issues
        for i, (_, title_text, _) in enumerate(title_structure):
            if i < len(title_elements):
                matched_text = title_elements[i].text[:100] + "..." if len(title_elements[i].text) > 100 else title_elements[i].text
                print(f"  Title '{title_text}' matched with: '{matched_text}'")
            else:
                print(f"  Title '{title_text}' had no match")
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
        1.1.3. 三、风险防范措施  
        1.1.4. 四、相关法律法规  
            1.1.4.1. 《民法典》  
            1.1.4.2. 《最高人民法院关于审理买卖合同纠纷案件适用法律问题的解释》  
            1.1.4.3. 《最高人民法院关于适用<中华人民共和国民法典>合同编通则若干问题的解释》  
            1.1.4.4. 《标准化法》  
    1.2. 买卖合同（通用版）  
        1.2.1. 合同编号：签订地点：  
        1.2.2. 出卖人：买受人：送达地址：送达地址：法定代表人：法定代表人：电话：电话：邮编：邮编：开户行：开户行：账号：账号：税号：税号：  
        1.2.3. 买受人为了的目的1,在平等、自愿的基础上，依据《中华人民共和国民法典》等法律法规，经与出卖人协商一致，就事宜达成以下条款，双方均需遵照执行。  
        1.2.4. 第一条合同标的物2  
        1.2.5. 第二条合同价款及支付方式  
        1.2.6. 第三条质量要求6  
        1.2.7. 第四条交付  
        1.2.8. 第五条货物所有权转移及风险转移  
        1.2.9. 第六条检验验收  
        1.2.10. 第七条包装条款  
        1.2.11. 第八条知识产权  
        1.2.12. 第九条违约责任12  
        1.2.13. 第十条不可抗力  
        1.2.14. 第十一条合同的生效、变更、解除和终止14  
        1.2.15. 第十二条争议解决  
        1.2.16. 第十三条廉洁合作  
        1.2.17. 第十四条其他约定  
        1.2.18. 附件：
    """
    
    # Build section tree
    section_tree = section_reconstructor(title_structure, layout_result)
    from models.naive_llm.helpers.section_token_parsor import remove_circular_references
    remove_circular_references(section_tree)
    
    json.dump(section_tree.model_dump(), open("./section_tree_0617.json", "w"), indent=4, ensure_ascii=False)