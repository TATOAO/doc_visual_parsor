#!/usr/bin/env python3
"""
Section Tree Builder using Element Encoder System.
Constructs hierarchical document structure based on element encoding and distance analysis.
"""

import json
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass

from models.schemas.layout_schemas import LayoutElement, ElementType, LayoutExtractionResult
from models.schemas.schemas import Section, Positions, DocumentType
from .element_encoder import ElementEncoder, FeatureWeights, ElementCode
from .helper import title_number_extraction, TitleNumber


@dataclass
class TreeBuildingConfig:
    """Configuration for tree building process"""
    # Distance thresholds
    sibling_distance_threshold: float = 0.2
    parent_child_distance_threshold: float = 0.4
    merge_distance_threshold: float = 0.15
    
    # Hierarchy rules
    max_hierarchy_gap: int = 2  # Max levels to skip when finding parent
    min_content_length: int = 5  # Minimum content length to create section
    
    # Element type priorities for section creation
    section_element_types: Set[str] = None
    content_element_types: Set[str] = None
    
    def __post_init__(self):
        if self.section_element_types is None:
            self.section_element_types = {
                'Title', 'Heading', 'TEXT'
            }
        if self.content_element_types is None:
            self.content_element_types = {
                'Paragraph', 'TEXT', 'List'
            }


class SectionTreeBuilder:
    """
    Builds section trees from layout elements using element encoder analysis
    """
    
    def __init__(self, 
                 encoder: Optional[ElementEncoder] = None,
                 config: Optional[TreeBuildingConfig] = None):
        self.encoder = encoder or ElementEncoder(use_semantic_embeddings=False)
        self.config = config or TreeBuildingConfig()
        
        # State tracking
        self.element_codes: Dict[int, ElementCode] = {}
        self.elements_by_id: Dict[int, LayoutElement] = {}
        self.section_counter = 0
    
    async def build_tree(self, elements: List[LayoutElement]) -> Section:
        """
        Build section tree from layout elements
        
        Args:
            elements: List of layout elements
            
        Returns:
            Root section containing the full tree
        """
        
        print(f"Building section tree from {len(elements)} elements...")
        
        # Step 1: Encode all elements
        await self._encode_elements(elements)
        
        # Step 2: Preprocess and clean elements
        processed_elements = await self._preprocess_elements(elements)
        
        # Step 3: Identify section candidates
        section_candidates = await self._identify_section_candidates(processed_elements)
        
        # Step 4: Build initial sections
        initial_sections = await self._build_initial_sections(section_candidates, processed_elements)
        
        # Step 5: Build hierarchical relationships
        root_section = await self._build_hierarchy(initial_sections)
        
        # Step 6: Refine and merge sections
        await self._refine_tree(root_section)
        
        print(f"Built tree with {len(self._get_all_descendants(root_section)) + 1} total sections")
        
        return root_section
    
    async def _encode_elements(self, elements: List[LayoutElement]):
        """Encode all elements and build lookup structures"""
        
        print("Encoding elements...")
        element_codes = self.encoder.encode_elements(elements)
        
        self.element_codes = {code.element_id: code for code in element_codes}
        self.elements_by_id = {elem.id: elem for elem in elements}
        
        print(f"Encoded {len(element_codes)} elements")
    
    async def _preprocess_elements(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """Preprocess elements: merge candidates, clean content, etc."""
        
        print("Preprocessing elements...")
        processed = []
        i = 0
        
        while i < len(elements):
            current = elements[i]
            
            # Find elements to merge with current
            merge_candidates = []
            j = i + 1
            
            while j < len(elements):
                next_elem = elements[j]
                
                # Check if should merge using encoder distance
                current_code = self.element_codes.get(current.id)
                next_code = self.element_codes.get(next_elem.id)
                
                if current_code and next_code:
                    distance = self.encoder.calculate_distance(current_code, next_code)
                    
                    if distance < self.config.merge_distance_threshold:
                        merge_candidates.append(next_elem)
                        j += 1
                    else:
                        break
                else:
                    break
            
            # Merge if candidates found
            if merge_candidates:
                merged_element = await self._merge_elements([current] + merge_candidates)
                processed.append(merged_element)
                i = j
            else:
                processed.append(current)
                i += 1
        
        print(f"Preprocessed {len(elements)} -> {len(processed)} elements")
        return processed
    
    async def _merge_elements(self, elements: List[LayoutElement]) -> LayoutElement:
        """Merge multiple elements into one"""
        
        if len(elements) == 1:
            return elements[0]
        
        base = elements[0]
        combined_text = " ".join(elem.text or "" for elem in elements if elem.text)
        
        # Update element codes for merged element
        merged_element = LayoutElement(
            id=base.id,
            element_type=base.element_type,
            confidence=min(elem.confidence for elem in elements),
            bbox=base.bbox,
            text=combined_text,
            style=base.style,
            metadata={
                **(base.metadata or {}),
                'merged_from': [elem.id for elem in elements[1:]],
                'original_count': len(elements)
            }
        )
        
        return merged_element
    
    async def _identify_section_candidates(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """Identify elements that should become section headers"""
        
        print("Identifying section candidates...")
        candidates = []
        
        for element in elements:
            code = self.element_codes.get(element.id)
            if not code:
                continue
            
            # Criteria for section candidates
            is_candidate = False
            
            # 1. Has title numbering
            if code.title_number_type:
                is_candidate = True
            
            # 2. Is title/heading element type
            elif element.element_type.value in ['Title', 'Heading']:
                is_candidate = True
            
            # 3. Has distinctive formatting (large font, bold, etc.)
            elif (code.font_size_norm > 0.7 or  # Large font
                  code.font_style_code & 1):      # Bold
                is_candidate = True
            
            # 4. Has hierarchical structure patterns
            elif code.hierarchy_level < 3:  # Upper hierarchy levels
                is_candidate = True
            
            if is_candidate:
                candidates.append(element)
        
        print(f"Found {len(candidates)} section candidates")
        return candidates
    
    async def _build_initial_sections(self, candidates: List[LayoutElement], 
                                    all_elements: List[LayoutElement]) -> List[Section]:
        """Build initial section nodes from candidates"""
        
        print("Building initial sections...")
        sections = []
        
        for candidate in candidates:
            code = self.element_codes.get(candidate.id)
            if not code:
                continue
            
            # Extract title (first part of text, typically)
            title = self._extract_title(candidate.text or "")
            
            # Find content elements for this section
            content_elements = await self._find_content_elements(candidate, all_elements)
            content = " ".join(elem.text or "" for elem in content_elements if elem.text)
            
            # Create position information
            title_position = self._create_position_from_element(candidate)
            content_position = self._create_position_from_elements(content_elements)
            
            section = Section(
                title=title,
                content=content,
                level=code.hierarchy_level,
                title_position=title_position,
                content_position=content_position,
                title_parsed=title,  # Initial parsed title
                content_parsed=content  # Initial parsed content
            )
            
            # Store metadata for later use
            if not hasattr(section, '_metadata'):
                section._metadata = {}
            section._metadata.update({
                'font_size_norm': code.font_size_norm,
                'font_style_code': code.font_style_code,
                'title_number_type': code.title_number_type,
                'source_candidate_id': candidate.id,
                'element_ids': [candidate.id] + [elem.id for elem in content_elements],
                'hierarchy_path': code.hierarchy_path,
                'structural_signature': code.structural_signature,
                'element_type': candidate.element_type.value
            })
            
            sections.append(section)
        
        print(f"Built {len(sections)} initial sections")
        return sections
    
    def _create_position_from_element(self, element: LayoutElement) -> Positions:
        """Create position from a single element"""
        if element.bbox:
            return Positions.from_pdf(
                page_number=0,  # Default page
                bounding_box=element.bbox,
                metadata={'element_id': element.id}
            )
        else:
            return Positions.from_text(0, len(element.text or ""), element_id=element.id)
    
    def _create_position_from_elements(self, elements: List[LayoutElement]) -> Positions:
        """Create position from multiple content elements"""
        if not elements:
            return Positions.from_text(0, 0)
        
        if elements[0].bbox:
            # Use first element's bbox as representative
            return Positions.from_pdf(
                page_number=0,
                bounding_box=elements[0].bbox,
                metadata={'element_ids': [elem.id for elem in elements]}
            )
        else:
            total_length = sum(len(elem.text or "") for elem in elements)
            return Positions.from_text(0, total_length, element_ids=[elem.id for elem in elements])
    
    def _extract_title(self, text: str) -> str:
        """Extract clean title from element text"""
        if not text:
            return "Untitled Section"
        
        # Remove common prefixes and clean up
        title = text.strip()
        
        # Take first line or first sentence
        lines = title.split('\n')
        if lines:
            title = lines[0].strip()
        
        # Limit length
        if len(title) > 100:
            title = title[:97] + "..."
        
        return title or "Untitled Section"
    
    async def _find_content_elements(self, section_candidate: LayoutElement, 
                                   all_elements: List[LayoutElement]) -> List[LayoutElement]:
        """Find content elements that belong to a section"""
        
        content_elements = []
        candidate_code = self.element_codes.get(section_candidate.id)
        
        if not candidate_code:
            return content_elements
        
        # Find elements after this candidate that could be content
        candidate_index = None
        for i, elem in enumerate(all_elements):
            if elem.id == section_candidate.id:
                candidate_index = i
                break
        
        if candidate_index is None:
            return content_elements
        
        # Look for content elements after the candidate
        for i in range(candidate_index + 1, len(all_elements)):
            elem = all_elements[i]
            elem_code = self.element_codes.get(elem.id)
            
            if not elem_code:
                continue
            
            # Stop if we hit another section candidate of same or higher level
            if (elem_code.hierarchy_level <= candidate_code.hierarchy_level and
                elem_code.title_number_type):
                break
            
            # Stop if we hit a clear section header
            if (elem.element_type.value in ['Title', 'Heading'] and
                elem_code.font_size_norm >= candidate_code.font_size_norm):
                break
            
            # Add if it's content-type element
            if (elem.element_type.value in self.config.content_element_types or
                elem_code.hierarchy_level > candidate_code.hierarchy_level):
                content_elements.append(elem)
            
            # Limit content search to reasonable distance
            if len(content_elements) > 10:  # Reasonable limit
                break
        
        return content_elements
    
    async def _build_hierarchy(self, sections: List[Section]) -> Section:
        """Build hierarchical relationships between sections"""
        
        print("Building hierarchical relationships...")
        
        # Create root section
        root = Section(
            title="Document Root",
            content="",
            level=-1,
            title_position=Positions.from_text(0, 0),
            content_position=Positions.from_text(0, 0)
        )
        
        # Sort sections by hierarchy level and position
        def sort_key(s):
            element_ids = getattr(s, '_metadata', {}).get('element_ids', [])
            return (s.level, element_ids[0] if element_ids else 0)
        
        sections_sorted = sorted(sections, key=sort_key)
        
        # Stack to track current path in hierarchy
        section_stack = [root]
        
        for section in sections_sorted:
            # Find appropriate parent
            parent = await self._find_parent_section(section, section_stack)
            
            # Add to parent
            parent.sub_sections.append(section)
            section.parent_section = parent
            
            # Update stack - remove sections at same or lower level
            while len(section_stack) > 1 and section_stack[-1].level >= section.level:
                section_stack.pop()
            
            # Add current section to stack
            section_stack.append(section)
        
        return root
    
    async def _find_parent_section(self, section: Section, 
                                 section_stack: List[Section]) -> Section:
        """Find the appropriate parent for a section"""
        
        # Start from the end of stack (most recent sections)
        for i in range(len(section_stack) - 1, -1, -1):
            potential_parent = section_stack[i]
            
            # Parent must be at higher level (lower number)
            if potential_parent.level < section.level:
                # Check if level gap is reasonable
                level_gap = section.level - potential_parent.level
                if level_gap <= self.config.max_hierarchy_gap:
                    return potential_parent
        
        # Default to root if no suitable parent found
        return section_stack[0]
    
    async def _refine_tree(self, root: Section):
        """Refine the tree structure: merge similar sections, clean up, etc."""
        
        print("Refining tree structure...")
        
        # Remove empty sections
        await self._remove_empty_sections(root)
        
        # Merge similar adjacent sections
        await self._merge_similar_sections(root)
        
        # Clean up single-child chains
        await self._clean_single_child_chains(root)
    
    async def _remove_empty_sections(self, node: Section):
        """Remove sections with no meaningful content"""
        
        children_to_remove = []
        
        for child in node.sub_sections:
            # Check if section is essentially empty
            metadata = getattr(child, '_metadata', {})
            hierarchy_path = metadata.get('hierarchy_path', '')
            
            if (len(child.content.strip()) < self.config.min_content_length and
                len(child.sub_sections) == 0 and
                not hierarchy_path):
                children_to_remove.append(child)
            else:
                # Recursively clean children
                await self._remove_empty_sections(child)
        
        # Remove empty children
        for child in children_to_remove:
            node.sub_sections.remove(child)
    
    async def _merge_similar_sections(self, node: Section):
        """Merge sections that are very similar"""
        
        # Process children first
        for child in node.sub_sections:
            await self._merge_similar_sections(child)
        
        # Look for adjacent similar children to merge
        i = 0
        while i < len(node.sub_sections) - 1:
            current = node.sub_sections[i]
            next_section = node.sub_sections[i + 1]
            
            # Check if should merge based on structural similarity
            should_merge = await self._should_merge_sections(current, next_section)
            
            if should_merge:
                # Merge next into current
                current.content += " " + next_section.content
                current.content_parsed += " " + next_section.content_parsed
                current.sub_sections.extend(next_section.sub_sections)
                
                # Update parent references for moved children
                for child in next_section.sub_sections:
                    child.parent_section = current
                
                # Merge metadata
                if hasattr(current, '_metadata') and hasattr(next_section, '_metadata'):
                    current_metadata = current._metadata
                    next_metadata = next_section._metadata
                    
                    # Merge element IDs
                    current_ids = current_metadata.get('element_ids', [])
                    next_ids = next_metadata.get('element_ids', [])
                    current_metadata['element_ids'] = current_ids + next_ids
                
                # Remove the merged section
                node.sub_sections.remove(next_section)
                
                print(f"Merged sections: {current.title} + {next_section.title}")
            else:
                i += 1
    
    async def _should_merge_sections(self, section1: Section, section2: Section) -> bool:
        """Determine if two sections should be merged"""
        
        # Don't merge if they have different hierarchy levels
        if section1.level != section2.level:
            return False
        
        # Don't merge if either has children (preserve structure)
        if section1.sub_sections or section2.sub_sections:
            return False
        
        # Check structural similarity using signatures
        metadata1 = getattr(section1, '_metadata', {})
        metadata2 = getattr(section2, '_metadata', {})
        
        sig1 = metadata1.get('structural_signature', '')
        sig2 = metadata2.get('structural_signature', '')
        
        if sig1 and sig2 and sig1 == sig2:
            return True
        
        # Check if both have very short content (might be fragments)
        if (len(section1.content.strip()) < 20 and 
            len(section2.content.strip()) < 20):
            return True
        
        return False
    
    async def _clean_single_child_chains(self, node: Section):
        """Clean up unnecessary single-child chains"""
        
        # Process children first
        for child in node.sub_sections:
            await self._clean_single_child_chains(child)
        
        # Look for single-child chains to collapse
        children_to_remove = []
        children_to_add = []
        
        for child in node.sub_sections:
            # If child has only one child and minimal content, consider collapsing
            if (len(child.sub_sections) == 1 and 
                len(child.content.strip()) < self.config.min_content_length):
                
                grandchild = child.sub_sections[0]
                
                # Promote grandchild
                grandchild.parent_section = node
                children_to_add.append(grandchild)
                children_to_remove.append(child)
                
                print(f"Collapsed single-child chain: {child.title} -> {grandchild.title}")
        
        # Apply changes
        for child in children_to_remove:
            node.sub_sections.remove(child)
        
        for child in children_to_add:
            node.sub_sections.append(child)
    
    def _get_all_descendants(self, section: Section) -> List[Section]:
        """Get all descendant sections"""
        descendants = []
        for child in section.sub_sections:
            descendants.append(child)
            descendants.extend(self._get_all_descendants(child))
        return descendants
    
    def print_tree(self, node: Section, indent: int = 0):
        """Print tree structure for debugging"""
        
        prefix = "  " * indent
        title_preview = node.title[:50] + "..." if len(node.title) > 50 else node.title
        content_preview = node.content[:30] + "..." if len(node.content) > 30 else node.content
        
        metadata = getattr(node, '_metadata', {})
        element_ids = metadata.get('element_ids', [])
        
        print(f"{prefix}├─ L{node.level}: {title_preview}")
        print(f"{prefix}   Content: {content_preview}")
        print(f"{prefix}   Elements: {len(element_ids)}, Children: {len(node.sub_sections)}")
        
        for child in node.sub_sections:
            self.print_tree(child, indent + 1)
    
    def export_tree(self, root: Section, output_file: str):
        """Export tree to JSON file"""
        
        def section_to_dict(section: Section, include_metadata=False) -> Dict[str, Any]:
            result = {
                "title": section.title,
                "content": section.content,
                "level": section.level,
                "title_parsed": section.title_parsed,
                "content_parsed": section.content_parsed,
                "section_hash": section.section_hash,
                "sub_sections": [section_to_dict(child, include_metadata) for child in section.sub_sections]
            }
            
            if include_metadata and hasattr(section, '_metadata'):
                result['_metadata'] = section._metadata
            
            return result
        
        tree_dict = section_to_dict(root, include_metadata=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tree_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Exported tree to {output_file}")
    
    def get_tree_statistics(self, root: Section) -> Dict[str, Any]:
        """Get statistics about the tree"""
        
        all_nodes = [root] + self._get_all_descendants(root)
        
        # Level distribution
        level_counts = {}
        for node in all_nodes:
            level_counts[node.level] = level_counts.get(node.level, 0) + 1
        
        # Content statistics
        content_lengths = [len(node.content) for node in all_nodes if node.content]
        
        return {
            "total_sections": len(all_nodes),
            "max_depth": self._get_max_depth(root),
            "level_distribution": level_counts,
            "avg_content_length": sum(content_lengths) / len(content_lengths) if content_lengths else 0,
            "sections_with_children": len([n for n in all_nodes if n.sub_sections]),
            "leaf_sections": len([n for n in all_nodes if not n.sub_sections])
        }
    
    def _get_max_depth(self, section: Section) -> int:
        """Get maximum depth of the tree"""
        if not section.sub_sections:
            return 1
        return 1 + max(self._get_max_depth(child) for child in section.sub_sections)


# python -m models.layout_structuring.content_merger.section_tree_builder
if __name__ == "__main__":
    async def main():
        section_tree_builder = SectionTreeBuilder()
        import json
        with open("./hybrid_extraction_result.json", "r") as f:
            layout_data = json.load(f)
        layout_extraction_result = LayoutExtractionResult(
            elements=layout_data["elements"], 
            metadata=layout_data["metadata"]
        )
        section = await section_tree_builder.build_tree(layout_extraction_result.elements)
        
        section_tree_builder.export_tree(section, "./section_tree_builder.json")

    import asyncio
    asyncio.run(main())