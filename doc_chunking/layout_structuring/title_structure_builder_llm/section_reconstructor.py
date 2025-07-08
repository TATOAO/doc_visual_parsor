from doc_chunking.layout_structuring.title_structure_builder_llm.structurer_llm import title_structure_builder_llm
from doc_chunking.layout_structuring.title_structure_builder_llm.layout_displayer import display_layout
from doc_chunking.utils.helper import remove_circular_references
from doc_chunking.schemas.layout_schemas import LayoutExtractionResult
from doc_chunking.schemas.schemas import Section, Positions
from rapidfuzz import fuzz, process
import re
import asyncio
from typing import Generator, AsyncGenerator, Union, Any


def enhanced_fuzzy_match_title(title_text: str, element_text: str) -> bool:
    """
    Enhanced fuzzy match title text against element text.
    
    Args:
        title_text: The title to search for
        element_text: The element text to search in
        
    Returns:
        True if fuzzy match found, False otherwise
    """
    if not title_text or not element_text:
        return False
    
    # Clean both texts by removing extra whitespace and normalizing
    title_clean = " ".join(title_text.strip().split())
    element_clean = " ".join(element_text.strip().split())
    
    # Try exact match first (most reliable)
    if title_clean in element_clean:
        return True
    
    # Calculate minimum similarity threshold based on title length
    title_len = len(title_clean)
    if title_len <= 2:
        min_similarity = 95  # Very short titles need high similarity
    elif title_len <= 4:
        min_similarity = 85  # Short titles need high similarity  
    elif title_len <= 8:
        min_similarity = 75  # Medium titles need good similarity
    else:
        min_similarity = 65  # Longer titles can have lower similarity
    
    # Use different fuzzy matching strategies
    
    # 1. Overall ratio (entire strings)
    ratio = fuzz.ratio(title_clean, element_clean)
    if ratio >= min_similarity:
        return True
        
    # 2. Partial ratio (best matching substring)
    partial_ratio = fuzz.partial_ratio(title_clean, element_clean)
    if partial_ratio >= min_similarity:
        return True
        
    # 3. For very short titles, also check if title appears as separate words
    if title_len <= 4:
        # Split element text into words and check individual words
        element_words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', element_clean)
        for word in element_words:
            if fuzz.ratio(title_clean, word) >= min_similarity:
                return True
    
    # 4. Token-based matching for longer titles
    if title_len > 4:
        token_ratio = fuzz.token_sort_ratio(title_clean, element_clean)
        if token_ratio >= min_similarity - 10:  # Slightly lower threshold for token matching
            return True
    
    return False


def _build_section_tree_from_structure(title_structure: list, layout_extraction_result: LayoutExtractionResult) -> Section:
    """
    Helper function to build section tree from parsed title structure.
    
    Args:
        title_structure: List of (level, title_text, number) tuples
        layout_extraction_result: A LayoutExtractionResult object containing the document layout
        
    Returns:
        Section: A hierarchical Section tree object
    """
    # Create root section
    root_section = Section(
        title="",
        content="",
        level=-1,  # Root level is -1 to make actual sections start at level 0
        element_id=-1
    )
    
    # Keep track of sections at each level for building the hierarchy
    level_sections = {-1: root_section}
    
    # Get layout elements in reading order
    sorted_elements = layout_extraction_result.elements
    
    # Map titles to elements
    title_elements = []
    current_element_idx = 0  # Keep track of where we are in sorted_elements
    
    for _, title_text, _ in title_structure:
        found_element = None
        # Search from current position to end
        while current_element_idx < len(sorted_elements):
            element = sorted_elements[current_element_idx]
            # Clean and compare the text using enhanced fuzzy matching
            element_text = element.text.strip() if element.text else ""

            if enhanced_fuzzy_match_title(title_text, element_text):
                found_element = element
                current_element_idx += 1  # Move to next element for next search
                break
            current_element_idx += 1
            
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

    return _build_section_tree_from_structure(title_structure, layout_extraction_result)


async def streaming_section_reconstructor(
    title_structure_stream: AsyncGenerator[str, None], 
    layout_extraction_result: LayoutExtractionResult
) -> AsyncGenerator[Section, None]:
    """
    Async streaming version of section_reconstructor that processes title structure chunks
    as they arrive and yields incremental section trees.

    Args:
        title_structure_stream: An async generator yielding string chunks of the title structure
        layout_extraction_result: A LayoutExtractionResult object containing the document layout

    Yields:
        Section: Incremental Section tree objects as more structure is parsed
    """
    # Initialize state
    accumulated_text = ""
    title_structure = []
    
    def parse_new_lines(text_chunk: str) -> list:
        """Parse complete lines from accumulated text."""
        lines = text_chunk.split('\n')
        complete_lines = []
        
        for line in lines[:-1]:  # All but the last line are complete
            line = line.strip()
            if not line:
                continue
                
            # Extract level from numbering (e.g., "1.1.2" -> level 2)
            parts = line.split(' ', 1)
            if len(parts) != 2:
                continue
                
            number, title = parts
            level = number.count('.')
            complete_lines.append((level, title.strip(), number.strip('.')))
        
        return complete_lines
    
    # Process streaming input
    async for chunk in title_structure_stream:
        accumulated_text += chunk
        
        # Check if we have any complete lines
        lines = accumulated_text.split('\n')
        if len(lines) > 1:
            # Process complete lines
            complete_text = '\n'.join(lines[:-1])
            accumulated_text = lines[-1]  # Keep the incomplete last line
            
            # Parse new complete lines
            new_lines = parse_new_lines(complete_text + '\n')
            
            if new_lines:
                # Add new lines to our title structure
                title_structure.extend(new_lines)
                
                # Build and yield incremental section tree
                current_root = _build_section_tree_from_structure(title_structure, layout_extraction_result)
                yield current_root
    
    # Process any remaining content
    if accumulated_text.strip():
        final_lines = parse_new_lines(accumulated_text + '\n')
        if final_lines:
            title_structure.extend(final_lines)
            final_root = _build_section_tree_from_structure(title_structure, layout_extraction_result)
            yield final_root


# python -m doc_chunking.layout_structuring.title_structure_builder_llm.section_reconstructor
if __name__ == "__main__":
    # Example usage
    import json
    
    # Load layout extraction result
    with open("./hybrid_extraction_result.json", "r") as f:
        layout_data = json.load(f)
    layout_result = LayoutExtractionResult.model_validate(layout_data)
    
    # Example title structure
    title_structure = """
1. 第一章  买卖合同  
    1.1 第一节  买卖合同（通用版）  
        1.1.1 一、定义  
        1.1.2 二、主要风险及常见易发问题提示  
            1.1.2.1 （一）主要业务风险  
            1.1.2.2 （二）合同管理常见易发问题  
        1.1.3 三、风险防范措施  
        1.1.4 四、相关法律法规  
            1.1.4.1 《民法典》  
            1.1.4.2 《最高人民法院关于审理买卖合同纠纷案件适用法律问题的解释》  
            1.1.4.3 《最高人民法院关于适用<中华人民共和国民法典>合同编通则若干问题的解释》  
            1.1.4.4 《标准化法》  
        1.1.5 买卖合同（通用版）  
            1.1.5.1 合同编号：  
            1.1.5.2 签订地点：  
            1.1.5.3 出 卖 人：                          买 受 人：  
            1.1.5.4 送达地址：                          送达地址：  
            1.1.5.5 法定代表人：                        法定代表人：  
            1.1.5.6 电    话：                          电    话：  
            1.1.5.7 邮    编：                          邮    编：  
            1.1.5.8 开 户 行：                          开 户 行：  
            1.1.5.9 账    号：                          账    号：  
            1.1.5.10 税    号：                          税    号：  
            1.1.5.11 买受人为了    的目的,在平等、自愿的基础上，依据《中华人民共和国民法典》等法律法规，经与出卖人协商一致，就    事宜达成以下条款，双方均需遵照执行。  
            1.1.5.12 第一条  合同标的物  
            1.1.5.13 第二条  合同价款及支付方式  
            1.1.5.14 第三条  质量要求  
            1.1.5.15 第四条  交付  
            1.1.5.16 第五条  货物所有权转移及风险转移  
            1.1.5.17 第六条  检验验收  
            1.1.5.18 第七条  包装条款  
            1.1.5.19 第八条  知识产权  
            1.1.5.20 第九条  违约责任  
            1.1.5.21 第十条  不可抗力  
            1.1.5.22 第十一条  合同的生效、变更、解除和终止  
            1.1.5.23 第十二条  争议解决  
            1.1.5.24 第十三条  廉洁合作  
            1.1.5.25 第十四条  其他约定  
            1.1.5.26 附件：
    """
    
    # Test regular section reconstructor
    print("=== Testing Regular Section Reconstructor ===")
    section_tree = section_reconstructor(title_structure, layout_result)
    from doc_chunking.utils.helper import remove_circular_references
    remove_circular_references(section_tree)
    
    json.dump(section_tree.model_dump(), open("./section_tree_0617.json", "w"), indent=4, ensure_ascii=False)
    
    # Test streaming section reconstructor
    print("\n=== Testing Streaming Section Reconstructor ===")
    
    async def simulate_llm_stream():
        """Simulate an LLM streaming title structure."""
        lines = title_structure.strip().split('\n')
        for line in lines:
            # Simulate streaming by yielding chunks of each line
            line_with_newline = line + '\n'
            # Split each line into smaller chunks to simulate streaming
            chunk_size = max(1, len(line_with_newline) // 3)
            for i in range(0, len(line_with_newline), chunk_size):
                chunk = line_with_newline[i:i+chunk_size]
                yield chunk
                await asyncio.sleep(0.01)  # Simulate delay
    
    async def test_streaming():
        """Test the streaming section reconstructor."""
        print("Starting streaming reconstruction...")
        
        section_count = 0
        async for partial_section_tree in streaming_section_reconstructor(
            simulate_llm_stream(), layout_result
        ):
            section_count += 1
            print(f"Received partial section tree #{section_count} with {len(partial_section_tree.sub_sections)} top-level sections")
            
            # Save each intermediate result
            remove_circular_references(partial_section_tree)
            filename = f"./section_tree_streaming_{section_count:03d}.json"
            json.dump(partial_section_tree.model_dump(), open(filename, "w"), indent=4, ensure_ascii=False)
        
        print(f"Streaming reconstruction completed with {section_count} intermediate results")
    
    # Run the streaming test
    asyncio.run(test_streaming())