import re
from typing import Dict, List, Tuple
from models.utils.schemas import Section

sample_text = """
<start-section-title-1>
第一章 买卖合同
<end-section-title-1>
<start-section-content-1>
共计5个合同示范文本,分别是买卖合同（通用版）、设备采购合同、产品经销合同、集采零配件买卖合同、原材料采购合同。编制主要依据是《民法典》《最高人民法院关于审理买卖合同纠纷案件适用法律问题的解释》《最高人民法院关于适用<中华人民共和国民法典>合同编通则若干问题的解释》。

<start-section-title-1-1>
第一节 买卖合同（通用版）
<end-section-title-1-1>
<start-section-content-1-1>
<start-section-title-1-1-1>
一、定义
<end-section-title-1-1-1>
<start-section-content-1-1-1>
买卖合同是出卖人转移标的物的所有权于买受人，买受人支付价款的合同。买卖合同属于《民法典》合同编规定的典型合同。
<end-section-content-1-1-1>

<start-section-title-1-1-2>
二、主要风险及常见易发问题提示
<end-section-title-1-1-2>
<start-section-content-1-1-2>
<start-section-title-1-1-2-1>
（一）主要业务风险
<end-section-title-1-1-2-1>
<start-section-content-1-1-2-1>
[此处省略具体风险内容，保持结构标记完整]
<end-section-content-1-1-2-1>

<start-section-title-1-1-2-2>
（二）合同管理常见易发问题
<end-section-title-1-1-2-2>
<start-section-content-1-1-2-2>
[此处省略具体内容，保持结构标记完整]
<end-section-content-1-1-2-2>
<end-section-content-1-1-2>

<start-section-title-1-1-3>
三、风险防范措施
<end-section-title-1-1-3>
<start-section-content-1-1-3>
[此处省略具体措施内容，保持结构标记完整]
<end-section-content-1-1-3>

<start-section-title-1-1-4>
四、相关法律法规
<end-section-title-1-1-4>
<start-section-content-1-1-4>
<start-section-title-1-1-4-1>
《民法典》
<end-section-title-1-1-4-1>
<start-section-content-1-1-4-1>
[此处省略《民法典》具体条款内容，保持结构标记完整]
<end-section-content-1-1-4-1>

<start-section-title-1-1-4-2>
《最高人民法院关于审理买卖合同纠纷案件适用法律问题的解释》
<end-section-title-1-1-4-2>
<start-section-content-1-1-4-2>
[此处省略解释具体内容，保持结构标记完整]
<end-section-content-1-1-4-2>

<start-section-title-1-1-5>
买卖合同（通用版）
<end-section-title-1-1-5>
<start-section-content-1-1-5>
[此处省略合同具体内容，保持结构标记完整]
<end-section-content-1-1-5>
<end-section-content-1-1>
<end-section-content-1>
"""

def generate_section_tree_from_tokens(text: str) -> Section:
    """
    parsing the special tokens in the text into a section tree
    """
    # Find all section title and content tags with their positions and levels
    title_start_pattern = r"<start-section-title-([\d-]+)>"
    title_end_pattern = r"<end-section-title-([\d-]+)>"
    content_start_pattern = r"<start-section-content-([\d-]+)>"
    content_end_pattern = r"<end-section-content-([\d-]+)>"
    
    # Find all title and content tags with their positions
    title_start_matches = [(m.start(), m.end(), m.group(1)) for m in re.finditer(title_start_pattern, text)]
    title_end_matches = [(m.start(), m.end(), m.group(1)) for m in re.finditer(title_end_pattern, text)]
    content_start_matches = [(m.start(), m.end(), m.group(1)) for m in re.finditer(content_start_pattern, text)]
    content_end_matches = [(m.start(), m.end(), m.group(1)) for m in re.finditer(content_end_pattern, text)]
    
    # Create a mapping of section IDs to their title and content positions
    sections_info = {}
    
    # Match title start and end tags
    for start_pos, start_end_pos, section_id in title_start_matches:
        # Find the corresponding title end tag
        for end_pos, end_end_pos, end_section_id in title_end_matches:
            if section_id == end_section_id:
                if section_id not in sections_info:
                    sections_info[section_id] = {}
                sections_info[section_id]['title'] = {
                    'start_pos': start_pos,
                    'start_tag_end': start_end_pos,
                    'end_pos': end_pos,
                    'end_tag_end': end_end_pos
                }
                break
    
    # Match content start and end tags
    for start_pos, start_end_pos, section_id in content_start_matches:
        # Find the corresponding content end tag
        for end_pos, end_end_pos, end_section_id in content_end_matches:
            if section_id == end_section_id:
                if section_id not in sections_info:
                    sections_info[section_id] = {}
                sections_info[section_id]['content'] = {
                    'start_pos': start_pos,
                    'start_tag_end': start_end_pos,
                    'end_pos': end_pos,
                    'end_tag_end': end_end_pos
                }
                break
    
    # Add level information to sections
    for section_id in sections_info:
        sections_info[section_id]['level'] = len(section_id.split('-'))
    
    # Create Section objects
    sections = {}
    root_section = Section(title="Root", level=0)
    
    # Sort sections by their level and then by position to process them in order
    sorted_sections = sorted(sections_info.items(), key=lambda x: (x[1]['level'], x[1].get('title', {}).get('start_pos', float('inf'))))
    
    for section_id, info in sorted_sections:
        # Extract title
        title = ""
        if 'title' in info:
            title_start = info['title']['start_tag_end']
            title_end = info['title']['end_pos']
            title = text[title_start:title_end].strip()
        
        # Extract content (excluding nested sections)
        content = ""
        if 'content' in info:
            content_start = info['content']['start_tag_end']
            content_end = info['content']['end_pos']
            raw_content = text[content_start:content_end]
            
            # Remove nested section tags from content
            content_lines = []
            lines = raw_content.split('\n')
            skip_lines = False
            nested_level = 0
            
            for line in lines:
                line_stripped = line.strip()
                
                # Check for nested section start tags
                if re.search(r'<start-section-(?:title|content)-([\d-]+)>', line_stripped):
                    nested_match = re.search(r'<start-section-(?:title|content)-([\d-]+)>', line_stripped)
                    if nested_match:
                        nested_id = nested_match.group(1)
                        # Check if this is a direct child section
                        if nested_id != section_id and nested_id.startswith(section_id + '-'):
                            skip_lines = True
                            nested_level += 1
                            continue
                
                # Check for nested section end tags
                elif re.search(r'<end-section-(?:title|content)-([\d-]+)>', line_stripped):
                    nested_match = re.search(r'<end-section-(?:title|content)-([\d-]+)>', line_stripped)
                    if nested_match and skip_lines:
                        nested_id = nested_match.group(1)
                        if nested_id != section_id and nested_id.startswith(section_id + '-'):
                            nested_level -= 1
                            if nested_level == 0:
                                skip_lines = False
                            continue
                
                # Add line to content if not in nested section
                if not skip_lines:
                    content_lines.append(line)
            
            content = '\n'.join(content_lines).strip()
        
        # Create the Section object
        section = Section(
            title=title,
            content=content,
            level=info['level']
        )
        
        sections[section_id] = section
    
    # Build the hierarchy
    for section_id, section in sections.items():
        section_parts = section_id.split('-')
        
        if len(section_parts) == 1:
            # Root level section
            root_section.sub_sections.append(section)
            section.parent_section = root_section
        else:
            # Find parent section
            parent_id = '-'.join(section_parts[:-1])
            if parent_id in sections:
                parent_section = sections[parent_id]
                parent_section.sub_sections.append(section)
                section.parent_section = parent_section
    
    # If there's only one root section, return it directly
    if len(root_section.sub_sections) == 1:
        return root_section.sub_sections[0]
    
    return root_section

def remove_circular_references(section: Section):
    """
    Recursively remove parent_section references to avoid circular reference in JSON serialization
    """
    section.parent_section = None
    for sub_section in section.sub_sections:
        remove_circular_references(sub_section)

# python -m models.naive_llm.section_token_parsor
if __name__ == "__main__":
    section_tree = generate_section_tree_from_tokens(sample_text)
    # import ipdb; ipdb.set_trace()
    
    # Remove circular references before JSON serialization
    remove_circular_references(section_tree)
    
    json_sample = section_tree.model_dump_json(indent=2)
    with open("section_tree.json", "w", encoding="utf-8") as f:
        f.write(json_sample)