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
    
    # Special handling for Chinese parenthetical titles like "丁方（担保方）"
    if "（" in title_text and "）" in title_text:
        # Try without parentheses
        title_no_parens = title_text.replace("（", "").replace("）", "")
        if title_no_parens in element_text:
            return True
        
        # Try with different parentheses styles
        title_with_regular_parens = title_text.replace("（", "(").replace("）", ")")
        if title_with_regular_parens in element_text:
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


def _extract_content_from_shared_element(element_text: str, title_text: str, next_title_text: str = None) -> str:
    """
    Extract content for a specific title from a shared element containing multiple titles.
    
    Args:
        element_text: The full text of the element
        title_text: The current title to extract content for
        next_title_text: The next title (if any) to determine where to stop
        
    Returns:
        The extracted content portion for this title
    """
    if not element_text or not title_text:
        return ""
    
    # Find the position of the current title in the element text
    title_start = element_text.find(title_text)
    if title_start == -1:
        # Try fuzzy matching to find approximate position
        # Remove special characters and try to match pattern
        title_clean = re.sub(r'[（）]', '', title_text)
        title_start = element_text.find(title_clean)
        if title_start == -1:
            return ""
    
    # Find where the content for this title starts (after the title)
    content_start = title_start + len(title_text)
    
    # Find where the content for this title ends
    content_end = len(element_text)
    
    if next_title_text:
        # Find the next title position
        next_title_start = element_text.find(next_title_text, content_start)
        if next_title_start == -1:
            # Try fuzzy matching for next title
            next_title_clean = re.sub(r'[（）]', '', next_title_text)
            next_title_start = element_text.find(next_title_clean, content_start)
        
        if next_title_start != -1:
            content_end = next_title_start
    
    # Extract the content portion
    content = element_text[content_start:content_end].strip()
    
    # Clean up the content (remove leading/trailing colons, etc.)
    content = re.sub(r'^[：:]+', '', content)
    content = re.sub(r'^[：:]+', '', content)
    
    return content.strip()


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
    
    # Map titles to elements with robust matching
    title_elements = []
    last_matched_idx = -1  # Track the last successfully matched element index
    
    for title_idx, (_, title_text, _) in enumerate(title_structure):
        found_element = None
        found_idx = -1
        
        # Search for the title starting from the current position
        # Allow searching from the same element as we may have multiple titles in one element
        search_start = max(0, last_matched_idx)
        
        for element_idx in range(search_start, len(sorted_elements)):
            element = sorted_elements[element_idx]
            element_text = element.text.strip() if element.text else ""

            if enhanced_fuzzy_match_title(title_text, element_text):
                found_element = element
                found_idx = element_idx
                break
        
        if found_element:
            title_elements.append(found_element)
            # Update last_matched_idx to this element's index
            # This allows subsequent titles to be found in the same element or later elements
            last_matched_idx = found_idx
            print(f"Matched title '{title_text}' with element at index {found_idx}")
        else:
            # Title not found - append None to maintain alignment with title_structure
            title_elements.append(None)
            print(f"Warning: Title '{title_text}' not found in layout elements")
    
    # Verify we have the same number of elements as titles (including None entries)
    assert len(title_elements) == len(title_structure), f"Mismatch: {len(title_elements)} elements vs {len(title_structure)} titles"
    
    # Build section tree - now handles None elements gracefully
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
        
        # Extract content for this section
        if matching_element:
            # Check if the next title is in the same element
            next_title_text = None
            next_element = None
            
            # Find the next title and its element
            for next_idx in range(idx + 1, len(title_elements)):
                next_element = title_elements[next_idx]
                if next_element is not None:
                    next_title_text = title_structure[next_idx][1]  # Get title text
                    break
            
            # Case 1: Next title is in the same element - extract content within this element
            if next_element is not None and next_element == matching_element:
                element_text = matching_element.text.strip() if matching_element.text else ""
                section.content = _extract_content_from_shared_element(element_text, title_text, next_title_text)
            
            # Case 2: Next title is in a different element or no next title - use original logic
            else:
                # Find the next title element's index in sorted_elements
                current_element_index = sorted_elements.index(matching_element)
                next_title_index = len(sorted_elements)
                
                # Look for the next matched title element (skip None entries)
                for next_idx in range(idx + 1, len(title_elements)):
                    next_title = title_elements[next_idx]
                    if next_title is not None:
                        try:
                            next_title_index = sorted_elements.index(next_title)
                            break
                        except ValueError:
                            continue
                
                # If this is the last title in the current element, extract remaining content from the element
                if next_title_index > current_element_index:
                    # First, try to extract content from the current element itself
                    element_text = matching_element.text.strip() if matching_element.text else ""
                    element_content = _extract_content_from_shared_element(element_text, title_text, None)
                    
                    # Then collect content from subsequent elements
                    content_elements = [element_content] if element_content else []
                    
                    for i in range(current_element_index + 1, next_title_index):
                        element = sorted_elements[i]
                        # Skip if this element is a matched title (not None)
                        if element in [te for te in title_elements if te is not None]:
                            continue
                        if element.text:
                            content_elements.append(element.text)
                    
                    section.content = "\n".join(content_elements)
                else:
                    # Same element case - extract portion within element
                    element_text = matching_element.text.strip() if matching_element.text else ""
                    section.content = _extract_content_from_shared_element(element_text, title_text, next_title_text)
        else:
            # For unmatched titles, we can't extract content reliably
            section.content = ""
    
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

    title_structure = """
1. 合同名称：智能设备采及运合作合同购维XX
2. 日期：2025年月日签订0708
3. 甲方（采购方）：购
    3.1. 甲方户名：科技有限公司账称XX
    3.2. 甲方纳税人识别号：91110101XXXXXXXXXX
    3.3. 甲方开户银行：中国工商银行支行开XX
    3.4. 甲方联系地址：北京市朝阳街道号区XXXX
    3.5. 甲方邮政编码：100000
    3.6. 甲方联系电话：010-XXXXXXXX
    3.7. 甲方联系邮箱：contact@jiafang.com
    3.8. 甲方联系人：李四
    3.9. 甲方传真：010-XXXXXXXX
4. 乙方（供货方）：货
    4.1. 乙方户名：电子设备有限公司账称XX
    4.2. 乙方纳税人识别号：91310101XXXXXXXXXX
    4.3. 乙方开户银行：中国建设银行支行开XX
    4.4. 乙方银行账号：账621700XXXXXXXXXXXX
    4.5. 乙方联系地址：上海市浦东新街道号区XXXX
    4.6. 乙方邮政编码：200000
    4.7. 乙方联系电话：021-XXXXXXXX
    4.8. 乙方联系邮箱：contact@yifang.com
    4.9. 乙方联系人：王五
    4.10. 乙方传真：021-XXXXXXXX
5. 丙方（运输方）：维
    5.1. 丙方户名：设备维护有限公司账称维XX
    5.2. 丙方纳税人识别号：91440101XXXXXXXXXX
    5.3. 丙方开户银行：中国银行支行开XX
    5.4. 丙方银行账号：账621661XXXXXXXXXXXX
    5.5. 丙方联系地址：广州市天河街道号区XXXX
    5.6. 丙方邮政编码：510000
    5.7. 丙方联系电话：020-XXXXXXXX
    5.8. 丙方联系邮箱：contact@sanfang.com
    5.9. 丙方联系人：六赵
    5.10. 丙方传真：020-XXXXXXXX
6. 丁方（担保方）：丁方户名：担保有限公司账称XX
    6.1. 丁方纳税人识别号：91510101XXXXXXXXXX
    6.2. 丁方开户银行：中国农业银行支行开XX
    6.3. 丁方银行账号：账622848XXXXXXXXXXXX
    6.4. 丁方联系地址：成都市江街道号锦区XXXX
    6.5. 丁方邮政编码：610000
    6.6. 丁方联系电话：028-XXXXXXXX
    6.7. 丁方联系邮箱：contact@dingfang.com
    6.8. 丁方联系人：六赵
    6.9. 丁方传真：028-XXXXXXXX
7. 一、项目概况
8. 二、产品信息及金额
9. 三、付款与结算
10. 四、履行约定
11. 五、税费相关
12. 六、保密义务
13. 七、违约责任
14. 八、不可抗力
15. 九、争议解决
16. 十、通知送达
17. 十一、合同生效与终止
18. 十二、其他
"""
    
    # Test regular section reconstructor
    print("=== Testing Regular Section Reconstructor ===")
    section_tree = section_reconstructor(title_structure, layout_result)
    from doc_chunking.utils.helper import remove_circular_references
    remove_circular_references(section_tree)
    
    json.dump(section_tree.model_dump(), open("./section_tree_0724.json", "w"), indent=4, ensure_ascii=False)
    
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