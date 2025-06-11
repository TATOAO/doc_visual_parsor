from models.schemas.schemas import Section, Positions, DocumentType
from models.layout_detection.base_detector import LayoutDetectionResult, BaseLayoutDetector, ElementType
from models.layout_detection.document_detector import DocumentLayoutDetector
from typing import List, Optional, Dict, Any
import logging
import re

logger = logging.getLogger(__name__)

class SectionParser:
    """
    Parser that converts layout detection results into hierarchical Section trees.
    """
    
    def __init__(self, detector: Optional[BaseLayoutDetector] = None):
        """
        Initialize the section parser.
        
        Args:
            detector: Layout detector to use. If None, will use DocumentLayoutDetector.
        """
        self.detector = detector or DocumentLayoutDetector(extract_text=True, analyze_styles=True)
    
    def parse_from_layout_result(self, layout_result: LayoutDetectionResult) -> Section:
        """
        Parse a Section tree from layout detection results.
        
        Args:
            layout_result: Results from layout detection
            
        Returns:
            Root Section containing the hierarchical document structure
        """
        elements = layout_result.get_elements()
        if not elements:
            return Section(title="Empty Document", content="")
        
        # Group elements into sections based on headings and titles
        sections = self._build_section_hierarchy(elements)
        
        # Create root section
        root_section = self._create_root_section(sections)
        
        return root_section
    
    def _build_section_hierarchy(self, elements: List) -> List[Section]:
        """
        Build hierarchical sections from layout elements.
        
        Args:
            elements: List of LayoutElements from detection
            
        Returns:
            List of top-level sections with proper hierarchy
        """
        sections = []
        current_section = None
        current_content = []
        
        # Track heading levels for hierarchy
        section_stack = []  # Stack to maintain parent-child relationships
        
        for i, element in enumerate(elements):
            element_type = element.element_type
            
            # Check if this element starts a new section
            # Include TEXT elements that look like section headers
            is_section_header = (
                element_type in [ElementType.TITLE, ElementType.HEADING] or
                self._is_section_header(element)
            )
            
            if is_section_header:
                # Save current section if it exists
                if current_section is not None:
                    current_section.content = self._combine_content(current_content)
                    current_section.content_parsed = current_section.content
                    sections.append(current_section)
                
                # Determine section level
                level = self._determine_section_level(element)
                
                # Create new section
                current_section = Section(
                    title=element.text or f"Section {len(sections) + 1}",
                    title_parsed=element.text or f"Section {len(sections) + 1}",
                    content="",
                    level=level,
                    title_position=self._create_position_from_element(element, i),
                    content_position=Positions.from_docx(0, 0, 0)  # Will be updated when content is added
                )
                
                # Handle hierarchy based on level
                self._handle_section_hierarchy(current_section, section_stack)
                
                current_content = []
                
            else:
                # Add to current section's content
                if element.text:
                    content_item = {
                        'type': element_type.value,
                        'text': element.text,
                        'position': self._create_position_from_element(element, i),
                        'metadata': element.metadata or {}
                    }
                    current_content.append(content_item)
        
        # Don't forget the last section
        if current_section is not None:
            current_section.content = self._combine_content(current_content)
            current_section.content_parsed = current_section.content
            sections.append(current_section)
        
        return sections
    
    def _determine_section_level(self, element) -> int:
        """
        Determine the hierarchical level of a section based on the element.
        
        Args:
            element: LayoutElement representing a heading or title
            
        Returns:
            Integer level (0 = root, 1 = main section, etc.)
        """
        if element.element_type == ElementType.TITLE:
            return 0  # Top level
        elif element.element_type == ElementType.HEADING:
            # Try to extract level from style or metadata
            if element.metadata and 'style_name' in element.metadata:
                style_name = element.metadata['style_name'].lower()
                if 'heading 1' in style_name:
                    return 1
                elif 'heading 2' in style_name:
                    return 2
                elif 'heading 3' in style_name:
                    return 3
                elif 'heading 4' in style_name:
                    return 4
                else:
                    return 1  # Default heading level
            else:
                return 1  # Default heading level
        else:
            return 2  # Default for other elements that might be section starts
    
    def _handle_section_hierarchy(self, current_section: Section, section_stack: List[Section]):
        """
        Handle parent-child relationships between sections based on their levels.
        
        Args:
            current_section: The section being processed
            section_stack: Stack of parent sections
        """
        current_level = current_section.level
        
        # Pop sections from stack that are at same or deeper level
        while section_stack and section_stack[-1].level >= current_level:
            section_stack.pop()
        
        # Set parent relationship
        if section_stack:
            parent = section_stack[-1]
            current_section.parent_section = parent
            parent.sub_sections.append(current_section)
        
        # Add current section to stack
        section_stack.append(current_section)
    
    def _combine_content(self, content_items: List[Dict[str, Any]]) -> str:
        """
        Combine content items into a single text string.
        
        Args:
            content_items: List of content dictionaries
            
        Returns:
            Combined content text
        """
        if not content_items:
            return ""
        
        content_parts = []
        for item in content_items:
            text = item.get('text', '').strip()
            if text:
                content_parts.append(text)
        
        return '\n\n'.join(content_parts)
    
    def _create_position_from_element(self, element, element_index: int) -> Positions:
        """
        Create a Positions object from a LayoutElement.
        
        Args:
            element: LayoutElement
            element_index: Index of the element in the document
            
        Returns:
            Positions object
        """
        # Extract position information from element's metadata
        metadata = element.metadata or {}
        
        if 'paragraph_index' in metadata:
            # DOCX position
            paragraph_index = metadata['paragraph_index']
            return Positions.from_docx(
                paragraph_index=paragraph_index,
                character_start=0,  # Could be enhanced to track character positions
                character_end=len(element.text) if element.text else 0,
                element_index=element_index,
                detection_method=metadata.get('detection_method', 'unknown')
            )
        else:
            # Generic text position
            return Positions.from_text(
                start=element_index * 100,  # Rough approximation
                end=element_index * 100 + (len(element.text) if element.text else 0),
                element_index=element_index,
                detection_method=metadata.get('detection_method', 'unknown')
            )
    
    def _create_root_section(self, sections: List[Section]) -> Section:
        """
        Create a root section that contains all top-level sections.
        
        Args:
            sections: List of sections to organize under root
            
        Returns:
            Root Section object
        """
        if not sections:
            return Section(title="Empty Document", content="")
        
        # Find sections without parents (top-level sections)
        top_level_sections = [s for s in sections if s.parent_section is None]
        
        if len(top_level_sections) == 1 and top_level_sections[0].level == 0:
            # If there's already a single root-level section, return it
            return top_level_sections[0]
        else:
            # Create a root section containing all top-level sections
            root = Section(
                title="Document Root",
                title_parsed="Document Root",
                content="",
                content_parsed="",
                level=-1,  # Root level
                sub_sections=top_level_sections
            )
            
            # Update parent relationships
            for section in top_level_sections:
                section.parent_section = root
            
            return root
    
    def _is_section_header(self, element) -> bool:
        """
        Determine if an element should be treated as a section header based on content patterns.
        
        Args:
            element: LayoutElement to check
            
        Returns:
            True if element should be treated as section header
        """
        if not element.text:
            return False
        
        text = element.text.strip()
        
        # Check for common section header patterns
        section_patterns = [
            r'^第[一二三四五六七八九十\d]+章\s*',  # 第一章
            r'^第[一二三四五六七八九十\d]+节\s*',  # 第一节
            r'^[一二三四五六七八九十\d]+、',      # 一、 二、
            r'^\([一二三四五六七八九十\d]+\)',    # (一) (二)
            r'^\d+\.',                            # 1. 2. 3.
            r'^\d+\.\d+',                         # 1.1 1.2
        ]
        
        for pattern in section_patterns:
            if re.match(pattern, text):
                return True
        
        # Check if it's a short text that could be a title (less than 50 characters and not ending with period)
        if len(text) < 50 and not text.endswith('.') and not text.endswith('。'):
            # Additional heuristics for titles
            if any(keyword in text for keyword in ['风险', '措施', '定义', '问题', '提示']):
                return True
        
        return False


def parse_section_tree_from_docx(input_data: str) -> Section:
    """
    Parse the section tree from a docx file using layout detection.
    
    Args:
        input_data: Path to the DOCX file
        
    Returns:
        Root Section containing the hierarchical document structure
    """
    try:
        # Create section parser with document detector
        parser = SectionParser()
        
        # Run layout detection
        layout_result = parser.detector.detect(input_data)
        
        # Parse sections from layout results
        section_tree = parser.parse_from_layout_result(layout_result)
        
        logger.info(f"Successfully parsed section tree with {len(section_tree.sub_sections)} top-level sections")
        return section_tree
        
    except Exception as e:
        logger.error(f"Failed to parse section tree from {input_data}: {str(e)}")
        # Return empty section as fallback
        return Section(
            title="Parse Error",
            content=f"Failed to parse document: {str(e)}",
            title_parsed="Parse Error",
            content_parsed=f"Failed to parse document: {str(e)}"
        )


# unit test
if __name__ == "__main__":
    parsed_section_tree = parse_section_tree_from_docx("tests/test_data/1-1 买卖合同（通用版）.docx")
    print("Root section:", parsed_section_tree.title)
    print("Number of sub-sections:", len(parsed_section_tree.sub_sections))
    
    def print_section_tree(section: Section, indent: int = 0):
        """Helper function to print the section tree structure"""
        prefix = "  " * indent
        print(f"{prefix}- {section.title} (Level: {section.level})")
        if section.content:
            content_preview = section.content[:100] + "..." if len(section.content) > 100 else section.content
            print(f"{prefix}  Content: {content_preview}")
        
        for sub_section in section.sub_sections:
            print_section_tree(sub_section, indent + 1)
    
    print("\nSection Tree Structure:")
    print_section_tree(parsed_section_tree)