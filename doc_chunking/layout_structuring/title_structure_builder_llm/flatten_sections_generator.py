from typing import AsyncGenerator, List, Dict, Any
from doc_chunking.schemas.layout_schemas import LayoutExtractionResult
from doc_chunking.schemas.schemas import Section
from doc_chunking.layout_structuring.title_structure_builder_llm.section_reconstructor import _build_section_tree_from_structure


def _flatten_section_tree(section: Section, parent_title: str = "") -> List[Section]:
    """
    Helper function to flatten a hierarchical section tree into a list of dictionaries.
    
    Args:
        section: The section to flatten
        parent_title: The accumulated parent title path (used for recursion)
    
    Returns:
        List of dictionaries with flattened section data
    """
    flattened_sections = []
    
    # Build the full title path
    if parent_title:
        full_title = f"{parent_title}-{section.title}" if section.title else parent_title
    else:
        full_title = section.title
    
    
    # Add current section to flattened list (only if it has content or title)
    if section.title or section.content:
        flattened_sections.append(section)
    
    # Recursively flatten sub-sections
    for sub_section in section.sub_sections:
        flattened_sections.extend(_flatten_section_tree(sub_section, full_title))
    
    return flattened_sections


async def streaming_flatten_sections_generator(
    title_structure_stream: AsyncGenerator[str, None], 
    layout_extraction_result: LayoutExtractionResult
) -> AsyncGenerator[List[Section], None]:
    """
    Async streaming version that processes title structure chunks as they arrive
    and yields flattened section dictionaries with joined parent titles.
    
    Each yielded list contains dictionaries with:
    - 'title': The title joined by all parent titles with format "titleA-titleA1-..."
    - 'content': The content of the section
    - 'level': The level of the section

    Args:
        title_structure_stream: An async generator yielding string chunks of the title structure
        layout_extraction_result: A LayoutExtractionResult object containing the document layout

    Yields:
        List[Dict[str, Any]]: Lists of flattened section dictionaries as more structure is parsed
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
                
                # Build incremental section tree
                current_root = _build_section_tree_from_structure(title_structure, layout_extraction_result)
                
                # Flatten the section tree and yield
                flattened_sections = _flatten_section_tree(current_root)
                yield flattened_sections
    
    # Process any remaining content
    if accumulated_text.strip():
        final_lines = parse_new_lines(accumulated_text + '\n')
        if final_lines:
            title_structure.extend(final_lines)
            final_root = _build_section_tree_from_structure(title_structure, layout_extraction_result)
            
            # Flatten the final section tree and yield
            flattened_sections = _flatten_section_tree(final_root)
            yield flattened_sections


def flatten_sections_generator(
    title_raw_structure: str, 
    layout_extraction_result: LayoutExtractionResult
) -> List[Dict[str, Any]]:
    """
    Non-streaming version that processes a complete title structure and returns
    flattened section dictionaries with joined parent titles.
    
    Each dictionary contains:
    - 'title': The title joined by all parent titles with format "titleA-titleA1-..."
    - 'content': The content of the section
    - 'level': The level of the section

    Args:
        title_raw_structure: A string containing the hierarchical title structure
        layout_extraction_result: A LayoutExtractionResult object containing the document layout

    Returns:
        List[Dict[str, Any]]: List of flattened section dictionaries
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

    # Build section tree
    section_tree = _build_section_tree_from_structure(title_structure, layout_extraction_result)
    
    # Flatten and return
    return _flatten_section_tree(section_tree)


# Example usage and test function
# python -m doc_chunking.layout_structuring.title_structure_builder_llm.flatten_sections_generator
if __name__ == "__main__":
    import json
    import asyncio
    
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
    """
    
    # Test non-streaming version
    print("=== Testing Non-Streaming Flatten Sections Generator ===")
    flattened = flatten_sections_generator(title_structure, layout_result)
    
    for i, section in enumerate(flattened):
        print(f"{i+1}. Title: '{section['title']}', Level: {section['level']}")
        print(f"   Content: '{section['content'][:100]}...' " if section['content'] else "   Content: (empty)")
        print()
    
    # Test streaming version
    print("\n=== Testing Streaming Flatten Sections Generator ===")
    
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
        """Test the streaming flatten sections generator."""
        print("Starting streaming flattening...")
        
        result_count = 0
        async for flattened_sections in streaming_flatten_sections_generator(
            simulate_llm_stream(), layout_result
        ):
            result_count += 1
            print(f"Received flattened sections #{result_count} with {len(flattened_sections)} sections")
            
            # Show first few sections
            for i, section in enumerate(flattened_sections[:3]):
                print(f"  {i+1}. Title: '{section['title']}', Level: {section['level']}")
            
            if len(flattened_sections) > 3:
                print(f"  ... and {len(flattened_sections) - 3} more sections")
            print()
        
        print(f"Streaming flattening completed with {result_count} intermediate results")
    
    # Run the streaming test
    asyncio.run(test_streaming())

