import re
from typing import Dict, List, Tuple, Optional
from doc_chunking.schemas.schemas import Section, Positions
from .tree_like_structure_mapping import set_section_position_index, flatten_section_tree_to_tokens


"""
Parse from "Section Token trees" to "Section Tree Object"
"""

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


def judge_whether_missing_content(text: str) -> bool:
    """
    Check if there's any content left outside of section tags.
    Returns True if there's untagged content (missing content that should be tagged).
    """
    all_tag_pattern = r"<(start|end)-section-(title|content)-([\d-]+)>"
    tag_matches = [(match.start(), match.end()) for match in re.finditer(all_tag_pattern, text)]
    tag_matches.sort()
    
    # Check content before first tag, between tags, and after last tag
    if tag_matches:
        # Before first tag
        if text[:tag_matches[0][0]].strip():
            return True
        
        # Between tags
        for i in range(len(tag_matches) - 1):
            between_content = text[tag_matches[i][1]:tag_matches[i + 1][0]].strip()
            if between_content:
                return True
        
        # After last tag
        if text[tag_matches[-1][1]:].strip():
            return True
    
    return False


class SectionTagMatcher:
    """Helper class to extract and match section tags"""
    
    def __init__(self, text: str, text_offset: int = 0):
        self.text = text
        self.text_offset = text_offset
        self.title_pattern = r"<(start|end)-section-title-([\d-]+)>"
        self.content_pattern = r"<(start|end)-section-content-([\d-]+)>"
    
    def extract_tag_matches(self) -> Dict[str, Dict]:
        """Extract all section tags and their positions"""
        sections_info = {}
        
        # Extract title tags
        self._match_tag_pairs(self.title_pattern, 'title', sections_info)
        
        # Extract content tags
        self._match_tag_pairs(self.content_pattern, 'content', sections_info)
        
        return sections_info
    
    def _match_tag_pairs(self, pattern: str, tag_type: str, sections_info: Dict):
        """Match start and end tags for a given pattern"""
        matches = list(re.finditer(pattern, self.text))
        start_tags = [(m.start(), m.end(), m.group(2)) for m in matches if m.group(1) == 'start']
        end_tags = [(m.start(), m.end(), m.group(2)) for m in matches if m.group(1) == 'end']
        
        # Match corresponding start and end tags
        for start_pos, start_end_pos, section_id in start_tags:
            for end_pos, end_end_pos, end_section_id in end_tags:
                if section_id == end_section_id:
                    if section_id not in sections_info:
                        sections_info[section_id] = {}
                    
                    sections_info[section_id][tag_type] = {
                        'start_pos': start_pos + self.text_offset,
                        'start_tag_end': start_end_pos + self.text_offset,
                        'end_pos': end_pos + self.text_offset,
                        'end_tag_end': end_end_pos + self.text_offset,
                        'local_start_pos': start_pos,
                        'local_start_tag_end': start_end_pos,
                        'local_end_pos': end_pos,
                        'local_end_tag_end': end_end_pos
                    }
                    break


class SectionContentExtractor:
    """Helper class to extract section content and handle nested sections"""
    
    def __init__(self, text: str):
        self.text = text
    
    def extract_title(self, title_info: Dict) -> str:
        """Extract section title from tag info"""
        if not title_info:
            return ""
        
        start = title_info['local_start_tag_end']
        end = title_info['local_end_pos']
        return self.text[start:end].strip()
    
    def extract_content_and_nested(self, content_info: Dict, section_id: str) -> Tuple[str, str]:
        """Extract section content and separate nested sections for recursion"""
        if not content_info:
            return "", ""
        
        start = content_info['local_start_tag_end']
        end = content_info['local_end_pos']
        raw_content = self.text[start:end]
        
        content_lines = []
        nested_lines = []
        lines = raw_content.split('\n')
        
        in_nested_section = False
        nested_level = 0
        
        for line in lines:
            if self._is_nested_section_start(line, section_id):
                in_nested_section = True
                nested_level += 1
                nested_lines.append(line)
            elif self._is_nested_section_end(line, section_id) and in_nested_section:
                nested_lines.append(line)
                nested_level -= 1
                if nested_level == 0:
                    in_nested_section = False
            elif in_nested_section:
                nested_lines.append(line)
            else:
                content_lines.append(line)
        
        content = '\n'.join(content_lines).strip()
        nested_content = '\n'.join(nested_lines).strip()
        
        return content, nested_content
    
    def _is_nested_section_start(self, line: str, section_id: str) -> bool:
        """Check if line contains a nested section start tag"""
        match = re.search(r'<start-section-(?:title|content)-([\d-]+)>', line.strip())
        if match:
            nested_id = match.group(1)
            return nested_id != section_id and nested_id.startswith(section_id + '-')
        return False
    
    def _is_nested_section_end(self, line: str, section_id: str) -> bool:
        """Check if line contains a nested section end tag"""
        match = re.search(r'<end-section-(?:title|content)-([\d-]+)>', line.strip())
        if match:
            nested_id = match.group(1)
            return nested_id != section_id and nested_id.startswith(section_id + '-')
        return False


class SectionHierarchyBuilder:
    """Helper class to build section hierarchy"""
    
    def __init__(self, sections: Dict[str, Section], parent_section: Optional[Section], max_depth: int):
        self.sections = sections
        self.parent_section = parent_section
        self.max_depth = max_depth
    
    def build_hierarchy(self) -> Section:
        """Build the section hierarchy"""
        root_section = self.parent_section or Section(title="Root", level=0)
        
        for section_id, section in self.sections.items():
            self._attach_section_to_parent(section_id, section, root_section)
        
        return root_section
    
    def _attach_section_to_parent(self, section_id: str, section: Section, root_section: Section):
        """Attach a section to its appropriate parent"""
        section_parts = section_id.split('-')
        
        if len(section_parts) == 1:
            # Top-level section (relative to current processing)
            target_parent = self.parent_section or root_section
            target_parent.sub_sections.append(section)
            section.parent_section = target_parent
        else:
            # Find parent section
            parent_id = '-'.join(section_parts[:-1])
            if parent_id in self.sections:
                parent_section_obj = self.sections[parent_id]
                
                # Check if parent is within depth limit
                parent_level = self._calculate_section_level(parent_id)
                if self.max_depth == -1 or parent_level < self.max_depth + 1:
                    parent_section_obj.sub_sections.append(section)
                    section.parent_section = parent_section_obj
    
    def _calculate_section_level(self, section_id: str) -> int:
        """Calculate the actual level of a section considering parent depth"""
        base_level = len(section_id.split('-'))
        if self.parent_section:
            base_level += self.parent_section.level
        return base_level


def generate_section_tree_from_tokens(
    token_text: str, 
    raw_text: Optional[str] = None, 
    max_depth: int = -1, 
    parent_section: Optional[Section] = None,
    text_offset: int = 0
) -> Section:
    """
    Parse special tokens in text into a section tree structure.
    
    Args:
        text: Input text with section tokens
        raw_text: Original raw text for position calculation (defaults to text if not provided)
        max_depth: Maximum depth of sections to include
                  -1: no limit (default)
                   0: only first level titles, no sub-sections
                   1: first and second level titles, second level are leaf nodes
                   n: include sections up to level n+1, level n+1 sections are leaf nodes
        parent_section: Parent section object for recursive processing
        text_offset: Offset position in the original raw_text (for recursive calls)
    
    Returns:
        Root section of the parsed tree
    """
    # Handle raw_text for parent section case
    if parent_section is not None:
        # When we have a parent section, raw_text should be the parent's content
        if raw_text is None:
            raw_text = f"{parent_section.title_parsed}\n{parent_section.content_parsed}"
        # Don't use text_offset for SectionTagMatcher since we're working with substring
        tag_matcher = SectionTagMatcher(token_text, 0)
    else:
        if raw_text is None:
            raw_text = token_text
        # Extract section tags and their positions
        tag_matcher = SectionTagMatcher(token_text, text_offset)
    
    sections_info = tag_matcher.extract_tag_matches()
    
    # Filter sections by depth and add level information
    sections_info = _filter_sections_by_depth(sections_info, parent_section, max_depth)
    
    # Create section objects
    sections = _create_section_objects(token_text, sections_info, parent_section, max_depth)

    # Build hierarchy
    hierarchy_builder = SectionHierarchyBuilder(sections, parent_section, max_depth)
    root_section = hierarchy_builder.build_hierarchy()
    
    # Set positions for all sections
    root_section = _set_section_positions(root_section, raw_text, parent_section)
    
    # Return single root section if only one exists and no parent_section
    if parent_section is None and len(root_section.sub_sections) == 1:
        return root_section.sub_sections[0]
    
    return root_section


def _filter_sections_by_depth(
    sections_info: Dict, 
    parent_section: Optional[Section], 
    max_depth: int
) -> Dict:
    """Filter sections based on maximum depth"""
    filtered_sections = {}
    
    for section_id, info in sections_info.items():
        level = len(section_id.split('-'))
        
        # Adjust level based on parent section depth
        if parent_section is not None:
            level += parent_section.level
        
        info['level'] = level
        
        # Include section if within depth limit
        if max_depth == -1 or level <= max_depth + 1:
            filtered_sections[section_id] = info
    
    return filtered_sections


def _create_section_objects(
    token_text: str, 
    sections_info: Dict, 
    parent_section: Optional[Section],
    max_depth: int
) -> Dict[str, Section]:
    """Create Section objects from extracted information"""
    sections = {}
    content_extractor = SectionContentExtractor(token_text)
    
    # Sort sections by level and position for proper processing order
    sorted_sections = sorted(
        sections_info.items(), 
        key=lambda x: (x[1]['level'], x[1].get('title', {}).get('local_start_pos', float('inf')))
    )
    
    for section_id, info in sorted_sections:
        # Extract title
        title = content_extractor.extract_title(info.get('title', {}))
        
        # Extract content and nested sections
        content, nested_content = content_extractor.extract_content_and_nested(
            info.get('content', {}), section_id
        )
        
        # Create section object
        section = Section(
            title=title,
            content=content,
            level=info['level']
        )
        
        # Store metadata for potential recursion
        section._recursion_content = nested_content
        section._global_positions = {
            'title': info.get('title', {}),
            'content': info.get('content', {})
        }
        
        # Apply depth limit for leaf nodes
        if info['level'] == _get_max_allowed_level(parent_section, max_depth):
            section.sub_sections = []
        
        sections[section_id] = section
    
    return sections


def _get_max_allowed_level(parent_section: Optional[Section], max_depth: int) -> int:
    """Calculate the maximum allowed level for sections"""
    if max_depth == -1:
        return float('inf')
    
    base_level = max_depth + 1
    if parent_section is not None:
        base_level += parent_section.level
    
    return base_level


def _set_section_positions(root_section: Section, raw_text: str, parent_section: Optional[Section] = None) -> Section:
    """Set position information for all sections in the tree"""
    
    # Only calculate positions for newly parsed sub-sections, not for existing parent section
    if parent_section is None:
        # No parent section, set positions for all sections normally
        root_section = set_section_position_index(root_section, raw_text)
    else:
        # We have a parent section, only set positions for sub-sections
        if root_section.sub_sections:
            # Create a temporary section tree with only the sub-sections for position calculation
            temp_section = Section(title="temp", level=0)
            temp_section.sub_sections = root_section.sub_sections
            temp_section = set_section_position_index(temp_section, raw_text)
            # Copy back the calculated positions to the original sub-sections
            root_section.sub_sections = temp_section.sub_sections
    
    # Calculate parent offset if we have a parent section
    parent_offset = 0
    if parent_section is not None and hasattr(parent_section, 'content_position'):
        parent_offset = parent_section.content_position.text_position.start
    
    # Process all sections but skip the root section if it's a parent section
    all_sections = flatten_section_tree_to_tokens(root_section)
    sections_to_process = all_sections[1:] if parent_section is not None else all_sections
    
    for section in sections_to_process:
        if hasattr(section, 'title_position') and hasattr(section, 'content_position'):
            # Apply parent offset to make positions absolute in the original document
            if parent_section is not None:
                section.title_position.text_position.start += parent_offset
                section.title_position.text_position.end += parent_offset
                section.content_position.text_position.start += parent_offset
                section.content_position.text_position.end += parent_offset
            
            section.title_parsed = raw_text[
                section.title_position.text_position.start - parent_offset:section.title_position.text_position.end - parent_offset
            ]
            section.content_parsed = raw_text[
                section.content_position.text_position.start - parent_offset:section.content_position.text_position.end - parent_offset
            ]
    
    return root_section


def process_section_recursively(
    section: Section, 
    raw_text: str, 
    max_depth: int = -1
) -> Section:
    """
    Recursively process a section's content for nested sections.
    Useful when LLM generates incomplete sections that need later processing.
    
    Args:
        section: Section object to process recursively
        raw_text: Original raw text for position calculation
        max_depth: Maximum depth for recursive processing
    
    Returns:
        The processed section with updated sub-sections
    """
    if not hasattr(section, '_recursion_content') or not section._recursion_content:
        return section
    
    # Calculate content offset in original text
    content_offset = 0
    if hasattr(section, '_global_positions') and 'content' in section._global_positions:
        content_offset = section._global_positions['content'].get('start_tag_end', 0)
    
    # Process nested content recursively
    # For recursive processing, pass the full raw_text so positions are calculated correctly
    generate_section_tree_from_tokens(
        token_text=section._recursion_content,
        raw_text=raw_text,
        max_depth=max_depth,
        parent_section=section,
        text_offset=content_offset
    )
    
    # Clean up temporary attributes
    _cleanup_section_metadata(section)
    
    return section


def _cleanup_section_metadata(section: Section):
    """Remove temporary metadata attributes from section"""
    for attr in ['_recursion_content', '_global_positions']:
        if hasattr(section, attr):
            delattr(section, attr)


def remove_circular_references(section: Section):
    """
    Recursively remove parent_section references to avoid circular reference in JSON serialization
    """
    section.parent_section = None
    for sub_section in section.sub_sections:
        remove_circular_references(sub_section)


# python -m models.naive_llm.helpers.section_token_parsor
if __name__ == "__main__":

    from doc_chunking.schemas.schemas import TextPosition, DocumentType

    root_section = Section(title="Root", level=3)
    root_section.title_position = Positions(document_type=DocumentType.TEXT, text_position=TextPosition(start=10019, end=10024))
    root_section.content_position = Positions(document_type=DocumentType.TEXT, text_position=TextPosition(start=10025, end=10025 + len(sample_text)))
    root_section.title_parsed = "Root"
    root_section.content_parsed = sample_text
    
    section_tree = generate_section_tree_from_tokens(sample_text, raw_text=sample_text, max_depth=5, parent_section=root_section)
    
    # Remove circular references before JSON serialization
    remove_circular_references(section_tree)
    
    json_sample = section_tree.model_dump_json(indent=2)
    with open("section_tree_unit_test_depth_2.json", "w", encoding="utf-8") as f:
        f.write(json_sample)