#
from models.utils.llm import get_llm_client
from models.schemas.layout_schemas import LayoutExtractionResult, LayoutElement, ElementType
from models.schemas.schemas import Section
from typing import List, Optional, Dict, Any
from .helper import (
    TitleNumberType, 
    TitleNumberUnit, 
    TitleNumber, 
    compare_title_numbers, 
    get_title_hierarchy_path, 
    title_number_extraction
)
from .element_encoder import ElementEncoder, FeatureWeights, ElementCode
from .section_tree_builder import SectionTreeBuilder, TreeBuildingConfig

class ContentMerger:
    """
    This class is used to merge the content of the elements using both rule-based 
    and distance-based analysis with compact element representations and construct 
    hierarchical section trees.
    """
    def __init__(self, use_element_encoder: bool = True, 
                 encoder_weights: Optional[FeatureWeights] = None,
                 tree_config: Optional[TreeBuildingConfig] = None):
        self.llm = get_llm_client()
        self.use_element_encoder = use_element_encoder
        
        # Initialize element encoder
        if self.use_element_encoder:
            self.encoder = ElementEncoder(
                weights=encoder_weights,
                use_semantic_embeddings=False  # Can be enabled if needed
            )
            self.element_codes: Dict[int, ElementCode] = {}
            
            # Initialize section tree builder
            self.tree_builder = SectionTreeBuilder(
                encoder=self.encoder,
                config=tree_config
            )
        else:
            self.encoder = None
            self.tree_builder = None

    async def construct_section_tree(self, elements: List[LayoutElement]) -> Section:
        """
        Construct a hierarchical section tree from layout elements.
        
        Args:
            elements: List of layout elements
            
        Returns:
            Root section containing the full hierarchical tree
        """
        
        if not elements:
            # Return empty root section
            from models.schemas.schemas import Positions
            return Section(
                title="Empty Document",
                content="",
                level=-1,
                title_position=Positions.from_text(0, 0),
                content_position=Positions.from_text(0, 0)
            )
        
        if not self.use_element_encoder or not self.tree_builder:
            # Fallback to simple section creation
            return await self._create_simple_section_tree(elements)
        
        # Use advanced tree builder
        print("Using SectionTreeBuilder to construct hierarchical tree...")
        root_section = await self.tree_builder.build_tree(elements)
        
        return root_section

    async def _create_simple_section_tree(self, elements: List[LayoutElement]) -> Section:
        """Fallback method to create a simple section tree without encoder"""
        
        from models.schemas.schemas import Positions
        
        # Create root section
        root_section = Section(
            title="Document Root",
            content="",
            level=-1,
            title_position=Positions.from_text(0, 0),
            content_position=Positions.from_text(0, 0)
        )
        
        # Simple approach: create one section per element that looks like a title
        current_section = None
        content_buffer = []
        
        for element in elements:
            # Simple heuristics to detect section headers
            is_section_header = False
            
            if element.element_type.value in ['Title', 'Heading']:
                is_section_header = True
            elif element.text and title_number_extraction(element.text):
                is_section_header = True
            elif (element.style and element.style.primary_font and 
                  element.style.primary_font.bold):
                is_section_header = True
            
            if is_section_header:
                # Finalize previous section
                if current_section is not None:
                    current_section.content = " ".join(content_buffer)
                    current_section.content_parsed = current_section.content
                    root_section.sub_sections.append(current_section)
                
                # Start new section
                title = element.text or "Untitled Section"
                
                current_section = Section(
                    title=title,
                    content="",
                    level=0,  # Simple flat structure
                    title_position=Positions.from_text(0, len(title)),
                    content_position=Positions.from_text(0, 0),
                    title_parsed=title,
                    content_parsed="",
                    parent_section=root_section
                )
                
                content_buffer = []
            else:
                # Add to content buffer
                if element.text:
                    content_buffer.append(element.text)
        
        # Finalize last section
        if current_section is not None:
            current_section.content = " ".join(content_buffer)
            current_section.content_parsed = current_section.content
            root_section.sub_sections.append(current_section)
        
        return root_section

    async def construct_section(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """
        Legacy method - construct sections using enhanced element analysis.
        
        Note: This method is deprecated. Use construct_section_tree() instead.
        """
        
        if not elements:
            return []
        
        # Encode elements if using element encoder
        if self.use_element_encoder:
            print("Encoding elements for analysis...")
            element_codes = self.encoder.encode_elements(elements)
            self.element_codes = {code.element_id: code for code in element_codes}
            
            # Analyze element relationships
            await self._analyze_element_relationships(elements, element_codes)
        
        # Process elements iteratively
        processed_elements = []
        i = 0
        
        while i < len(elements):
            current_element = elements[i]
            
            # Check if current element should be merged with next elements
            merge_candidates = await self._find_merge_candidates(current_element, elements[i+1:])
            
            if merge_candidates:
                # Merge elements
                merged_element = await self._merge_elements([current_element] + merge_candidates)
                processed_elements.append(merged_element)
                
                # Skip the merged elements
                i += len(merge_candidates) + 1
            else:
                processed_elements.append(current_element)
                i += 1
        
        return processed_elements

    async def _analyze_element_relationships(self, elements: List[LayoutElement], codes: List[ElementCode]):
        """Analyze relationships between elements using the encoder"""
        
        if not self.use_element_encoder:
            return
        
        print("Analyzing element relationships...")
        
        # Cluster elements by hierarchy and structure
        clusters = self.encoder.cluster_by_hierarchy(codes)
        
        print(f"Found {len(clusters)} structural clusters")
        
        # Analyze each cluster
        for cluster_key, cluster_codes in clusters.items():
            if len(cluster_codes) > 1:
                await self._analyze_cluster(cluster_codes, elements)

    async def _analyze_cluster(self, cluster_codes: List[ElementCode], all_elements: List[LayoutElement]):
        """Analyze a specific cluster of similar elements"""
        
        # Get the actual elements
        cluster_elements = []
        for code in cluster_codes:
            element = next(elem for elem in all_elements if elem.id == code.element_id)
            cluster_elements.append(element)
        
        # Analyze similarities within cluster
        print(f"Cluster analysis - {len(cluster_elements)} elements:")
        for i, element in enumerate(cluster_elements[:3]):  # Show first 3
            text_preview = (element.text or "")[:40].replace('\n', ' ')
            print(f"  {element.id}: {text_preview}")

    async def _find_merge_candidates(self, current_element: LayoutElement, 
                                   remaining_elements: List[LayoutElement]) -> List[LayoutElement]:
        """Find elements that should be merged with the current element"""
        
        candidates = []
        
        for next_element in remaining_elements:
            # Use both traditional and encoder-based analysis
            should_merge = False
            
            if self.use_element_encoder:
                # Distance-based analysis
                current_code = self.element_codes.get(current_element.id)
                next_code = self.element_codes.get(next_element.id)
                
                if current_code and next_code:
                    distance = self.encoder.calculate_distance(current_code, next_code)
                    
                    # Elements with very low distance might be candidates for merging
                    if distance < 0.2:  # Threshold for merging
                        should_merge = True
                        print(f"Distance-based merge candidate: {current_element.id} -> {next_element.id} (distance: {distance:.3f})")
            
            # Traditional rule-based analysis
            if not should_merge:
                should_merge = await self.is_sibling(current_element, next_element)
            
            if should_merge:
                candidates.append(next_element)
            else:
                # Stop at first non-mergeable element (preserves order)
                break
        
        return candidates

    async def _merge_elements(self, elements: List[LayoutElement]) -> LayoutElement:
        """Merge multiple elements into one"""
        
        if len(elements) == 1:
            return elements[0]
        
        # Use the first element as base
        base_element = elements[0]
        
        # Combine text content
        combined_text = ""
        for element in elements:
            if element.text:
                if combined_text:
                    combined_text += " "
                combined_text += element.text
        
        # Create merged element
        merged_element = LayoutElement(
            id=base_element.id,
            element_type=base_element.element_type,
            confidence=min(elem.confidence for elem in elements),
            bbox=base_element.bbox,
            text=combined_text,
            style=base_element.style,
            metadata={
                **(base_element.metadata or {}),
                'merged_from': [elem.id for elem in elements[1:]],
                'merge_method': 'encoder_based' if self.use_element_encoder else 'rule_based'
            }
        )
        
        return merged_element

    async def is_sibling(self, element1: LayoutElement, element2: LayoutElement) -> bool:
        """
        Check if two elements are siblings using multiple approaches
        """
        
        if not element1.text or not element2.text:
            return False
        
        # Method 1: Title number analysis
        title_num1 = title_number_extraction(element1.text)
        title_num2 = title_number_extraction(element2.text)
        
        if title_num1 and title_num2:
            # Both have title numbers - check if they're siblings
            return (title_num1.level == title_num2.level and 
                   title_num1.number_type == title_num2.number_type)
        
        # Method 2: Element encoder distance analysis
        if self.use_element_encoder:
            code1 = self.element_codes.get(element1.id)
            code2 = self.element_codes.get(element2.id)
            
            if code1 and code2:
                # Use hierarchical distance for sibling detection
                hier_distance = self.encoder.calculate_distance(code1, code2, "hierarchical")
                
                # Low hierarchical distance suggests sibling relationship
                if hier_distance < 0.15:  # Threshold for sibling relationship
                    return True
        
        # Method 3: Style-based similarity
        if element1.style and element2.style:
            # Check if fonts and styles are similar
            font1 = element1.style.primary_font
            font2 = element2.style.primary_font
            
            if font1 and font2:
                # Similar font size and style suggests same level
                size_similar = abs((font1.size or 12) - (font2.size or 12)) < 2
                style_similar = (font1.bold == font2.bold and 
                               font1.italic == font2.italic)
                
                if size_similar and style_similar:
                    return True
        
        return False

    async def get_element_analysis_report(self, elements: List[LayoutElement]) -> Dict[str, Any]:
        """Generate a comprehensive analysis report of the elements"""
        
        if not self.use_element_encoder:
            return {"error": "Element encoder not enabled"}
        
        # Encode elements
        element_codes = self.encoder.encode_elements(elements)
        
        # Generate clusters
        clusters = self.encoder.cluster_by_hierarchy(element_codes)
        
        # Analyze similarities
        similarity_analysis = {}
        title_elements = [elem for elem in elements if elem.element_type in [ElementType.TITLE, ElementType.HEADING]]
        
        if len(title_elements) >= 2:
            for i, elem1 in enumerate(title_elements[:5]):  # Analyze first 5 titles
                code1 = next(code for code in element_codes if code.element_id == elem1.id)
                similar_elements = self.encoder.find_similar_elements(code1, element_codes, threshold=0.4)
                
                similarity_analysis[elem1.id] = {
                    "text": elem1.text[:50] if elem1.text else "",
                    "similar_count": len(similar_elements),
                    "similar_elements": [
                        {
                            "id": sim_code.element_id,
                            "distance": distance,
                            "text": next(elem.text[:30] for elem in elements if elem.id == sim_code.element_id) or ""
                        }
                        for sim_code, distance in similar_elements[:3]
                    ]
                }
        
        return {
            "total_elements": len(elements),
            "total_clusters": len(clusters),
            "cluster_distribution": {k: len(v) for k, v in clusters.items()},
            "similarity_analysis": similarity_analysis,
            "encoder_stats": {
                "font_size_range": f"{self.encoder.font_size_stats['min']:.1f} - {self.encoder.font_size_stats['max']:.1f}",
                "position_stats": self.encoder.position_stats,
                "content_length_range": f"{self.encoder.content_length_stats['min']} - {self.encoder.content_length_stats['max']}"
            }
        }

    async def get_tree_analysis_report(self, root_section: Section) -> Dict[str, Any]:
        """Generate analysis report for a section tree"""
        
        if not self.tree_builder:
            return {"error": "Tree builder not available"}
        
        # Get tree statistics
        stats = self.tree_builder.get_tree_statistics(root_section)
        
        # Additional analysis
        all_sections = [root_section] + self.tree_builder._get_all_descendants(root_section)
        content_sections = [s for s in all_sections if s.level >= 0]
        
        quality_metrics = {}
        if content_sections:
            # Title quality
            titled_sections = [s for s in content_sections if s.title and s.title != "Untitled Section"]
            quality_metrics["title_quality"] = len(titled_sections) / len(content_sections)
            
            # Content coverage
            content_sections_with_text = [s for s in content_sections if s.content.strip()]
            quality_metrics["content_coverage"] = len(content_sections_with_text) / len(content_sections)
            
            # Content distribution
            content_lengths = [len(s.content) for s in content_sections if s.content]
            if content_lengths:
                quality_metrics["avg_content_length"] = sum(content_lengths) / len(content_lengths)
                quality_metrics["max_content_length"] = max(content_lengths)
                quality_metrics["min_content_length"] = min(content_lengths)
        
        return {
            **stats,
            "quality_metrics": quality_metrics,
            "tree_builder_config": {
                "merge_distance_threshold": self.tree_builder.config.merge_distance_threshold,
                "max_hierarchy_gap": self.tree_builder.config.max_hierarchy_gap,
                "min_content_length": self.tree_builder.config.min_content_length
            }
        }

# python -m models.layout_structuring.__init__
# if __name__ == "__main__":
#     async def main():
#         content_merger = ContentMerger()
#         import json
#         with open("hybrid_extraction_result.json", "r") as f:
#             layout_extraction_result = json.load(f)
#         layout_extraction_result = LayoutExtractionResult(elements=layout_extraction_result["elements"], 
#                                                           metadata=layout_extraction_result["metadata"])
#         section = await content_merger.construct_section(layout_extraction_result)
#         print(section)

#     import asyncio
#     asyncio.run(main())


# python -m models.layout_structuring.content_merger.__init__
if __name__ == "__main__":
    async def main():
        content_merger = ContentMerger()
        import json
        
        # Load JSON data
        with open("./hybrid_extraction_result.json", "r") as f:
            layout_data = json.load(f)
        
        # Create LayoutExtractionResult object
        layout_extraction_result = LayoutExtractionResult(
            elements=layout_data["elements"], 
            metadata=layout_data["metadata"]
        )
        
        # Construct the document sections
        section = await content_merger.construct_section(layout_extraction_result)

        from models.naive_llm.helpers import remove_circular_references
        remove_circular_references(section)

        with open("./section_post_merger.json", "w") as f:
            json.dump(section.model_dump(), f, indent=4, ensure_ascii=False)
        