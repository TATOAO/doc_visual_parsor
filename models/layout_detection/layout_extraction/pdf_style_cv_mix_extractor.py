"""
Hybrid PDF Style and CV Mix Layout Extractor

This module provides a hybrid implementation that combines PDF-native style extraction
with Computer Vision guidance for improved fragment merging and layout detection.
"""

import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

from ..base.base_layout_extractor import BaseLayoutExtractor
from ..visual_detection.cv_detector import CVLayoutDetector
from .pdf_layout_extractor import PdfLayoutExtractor
from models.schemas.layout_schemas import (
    LayoutExtractionResult,
    LayoutElement,
    BoundingBox,
    ElementType
)

logger = logging.getLogger(__name__)

InputDataType = Union[str, Path, bytes, Any]

class PdfStyleCVMixLayoutExtractor(BaseLayoutExtractor):
    """
    Hybrid Layout Extractor combining PDF-native style extraction with CV guidance.
    
    This extractor uses a clean architecture where:
    1. PdfLayoutExtractor handles PDF-native extraction and spatial merging
    2. CVLayoutDetector handles CV-based layout detection with PDF support
    3. This class combines both approaches for optimal results
    """

    def __init__(self, 
                 # PDF extraction parameters
                 merge_fragments: bool = True,
                 
                 # CV parameters
                 use_cv_guidance: bool = True,
                 cv_confidence_threshold: float = 0.3,
                 cv_model_name: str = "docstructbench",
                 cv_image_size: int = 1024,
                 cv_pdf_dpi: int = 200,
                 
                 # Hybrid strategy parameters
                 cv_guidance_weight: float = 0.7,
                 merge_strategy: str = "cv_guided",  # "cv_guided", "spatial_only", "cv_only"
                 
                 **kwargs):
        """
        Initialize the hybrid extractor.
        
        Args:
            merge_fragments: Enable fragment merging
            use_cv_guidance: Enable CV guidance for merging
            cv_confidence_threshold: CV detection confidence threshold
            cv_model_name: CV model to use
            cv_image_size: CV image input size
            cv_pdf_dpi: DPI for PDF to image conversion
            cv_guidance_weight: Weight for CV guidance in merging decisions
            merge_strategy: Strategy for combining approaches
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        # Store configuration
        self.merge_fragments = merge_fragments
        self.use_cv_guidance = use_cv_guidance
        self.cv_guidance_weight = cv_guidance_weight
        self.merge_strategy = merge_strategy
        
        # Initialize component extractors
        self.pdf_extractor = None
        self.cv_detector = None
        
        # Store CV parameters
        self.cv_params = {
            'confidence_threshold': cv_confidence_threshold,
            'model_name': cv_model_name,
            'image_size': cv_image_size,
            'pdf_dpi': cv_pdf_dpi
        }
        
        self.detector = self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the component detectors."""
        try:
            # Initialize PDF extractor (spatial merging only)
            self.pdf_extractor = PdfLayoutExtractor(merge_fragments=self.merge_fragments)
            logger.info("PDF-native extractor initialized")
            
            # Initialize CV detector if needed
            if self.use_cv_guidance:
                self.cv_detector = CVLayoutDetector(
                    model_name=self.cv_params['model_name'],
                    confidence_threshold=self.cv_params['confidence_threshold'],
                    image_size=self.cv_params['image_size'],
                    pdf_dpi=self.cv_params['pdf_dpi']
                )
                self.cv_detector._initialize_detector()
                logger.info("CV detector with PDF support initialized")
            
            logger.info(f"Hybrid extractor initialized with strategy: {self.merge_strategy}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid extractor: {str(e)}")
            # Fallback to PDF-only mode
            self.use_cv_guidance = False
            self.merge_strategy = "spatial_only"
            logger.warning("Falling back to PDF-only mode")
            return None
    
    def _detect_layout(self, 
                      input_data: InputDataType,
                      **kwargs) -> LayoutExtractionResult:
        """
        Core hybrid extraction method that combines PDF-native and CV approaches.
        
        Args:
            input_data: Input data (PDF file path, bytes, etc.)
            **kwargs: Additional extraction parameters
            
        Returns:
            LayoutExtractionResult with hybrid extraction results
        """
        try:
            # Validate input is PDF
            if not self._is_pdf_input(input_data):
                raise ValueError("Hybrid extractor only supports PDF input")
            
            # Choose extraction strategy
            if self.merge_strategy == "cv_only":
                return self._cv_only_extraction(input_data, **kwargs)
            elif self.merge_strategy == "spatial_only":
                return self._spatial_only_extraction(input_data, **kwargs)
            else:  # cv_guided
                return self._cv_guided_extraction(input_data, **kwargs)
                
        except Exception as e:
            logger.error(f"Hybrid extraction failed: {str(e)}")
            return LayoutExtractionResult(elements=[], metadata={"error": str(e)})
    
    def _is_pdf_input(self, input_data: Any) -> bool:
        """Check if input is a PDF."""
        if isinstance(input_data, (str, Path)):
            return str(input_data).lower().endswith('.pdf')
        # For other input types, assume they might be PDF content
        return True
    
    def _cv_only_extraction(self, input_data: InputDataType, **kwargs) -> LayoutExtractionResult:
        """Pure CV-based extraction."""
        if not self.cv_detector:
            raise RuntimeError("CV detector not available")
        
        logger.info("Performing CV-only extraction")
        result = self.cv_detector._detect_layout(input_data, **kwargs)
        
        # Update metadata
        if result.metadata:
            result.metadata['extraction_method'] = 'cv_only'
            result.metadata['hybrid_strategy'] = self.merge_strategy
        
        return result
    
    def _spatial_only_extraction(self, input_data: InputDataType, **kwargs) -> LayoutExtractionResult:
        """Pure PDF-native spatial extraction."""
        logger.info("Performing spatial-only extraction")
        result = self.pdf_extractor._detect_layout(input_data, **kwargs)
        
        # Update metadata
        if result.metadata:
            result.metadata['extraction_method'] = 'pdf_native_spatial_only'
            result.metadata['hybrid_strategy'] = self.merge_strategy
        
        return result
    
    def _cv_guided_extraction(self, input_data: InputDataType, **kwargs) -> LayoutExtractionResult:
        """CV-guided hybrid extraction - the main hybrid approach."""
        logger.info("Performing CV-guided hybrid extraction")
        
        if not self.cv_detector:
            logger.warning("CV detector not available, falling back to spatial-only")
            return self._spatial_only_extraction(input_data, **kwargs)
        
        try:
            # Step 1: Get PDF-native extraction (without merging to get raw elements)
            pdf_extractor_raw = PdfLayoutExtractor(merge_fragments=False)
            pdf_result = pdf_extractor_raw._detect_layout(input_data, **kwargs)
            raw_pdf_elements = pdf_result.elements
            
            if not raw_pdf_elements:
                logger.warning("No PDF elements extracted")
                return pdf_result
            
            # Step 2: Get CV guidance for the entire PDF
            cv_result = self.cv_detector._detect_layout(input_data, **kwargs)
            cv_elements = cv_result.elements
            
            logger.info(f"PDF extraction: {len(raw_pdf_elements)} elements")
            logger.info(f"CV detection: {len(cv_elements)} elements")
            
            # Step 3: Group elements by page for processing
            pages = self._group_elements_by_page(raw_pdf_elements)
            cv_pages = self._group_elements_by_page(cv_elements)
            
            # Step 4: Process each page with CV guidance
            merged_elements = []
            element_id = 0
            
            for page_num in pages.keys():
                page_elements = pages[page_num]
                page_cv_elements = cv_pages.get(page_num, [])
                
                logger.info(f"Processing page {page_num}: {len(page_elements)} PDF elements, {len(page_cv_elements)} CV elements")
                
                # Merge elements with CV guidance
                page_merged = self._merge_elements_with_cv_guidance(
                    page_elements, page_cv_elements
                )
                
                # Reassign IDs
                for element in page_merged:
                    element.id = element_id
                    element_id += 1
                
                merged_elements.extend(page_merged)
                
                logger.info(f"Page {page_num}: {len(page_elements)} -> {len(page_merged)} elements after CV-guided merging")
            
            # Step 5: Create final result
            metadata = {
                'extraction_method': 'hybrid_cv_guided',
                'hybrid_strategy': self.merge_strategy,
                'total_elements': len(merged_elements),
                'original_pdf_elements': len(raw_pdf_elements),
                'cv_elements': len(cv_elements),
                'cv_guidance_weight': self.cv_guidance_weight,
                'document_type': 'pdf',
                'page_count': len(pages),
                'cv_guidance_enabled': True
            }
            
            logger.info(f"CV-guided extraction complete: {len(raw_pdf_elements)} -> {len(merged_elements)} elements")
            return LayoutExtractionResult(elements=merged_elements, metadata=metadata)
            
        except Exception as e:
            logger.error(f"CV-guided extraction failed: {str(e)}")
            logger.warning("Falling back to spatial-only extraction")
            return self._spatial_only_extraction(input_data, **kwargs)
    
    def _group_elements_by_page(self, elements: List[LayoutElement]) -> Dict[int, List[LayoutElement]]:
        """Group elements by page number."""
        pages = {}
        for element in elements:
            page_num = element.metadata.get('page_number', 1) if element.metadata else 1
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(element)
        return pages
    
    def _merge_elements_with_cv_guidance(self, 
                                       pdf_elements: List[LayoutElement], 
                                       cv_elements: List[LayoutElement]) -> List[LayoutElement]:
        """
        Merge PDF elements using CV guidance.
        
        Args:
            pdf_elements: Raw PDF elements from one page
            cv_elements: CV-detected elements from the same page
            
        Returns:
            Merged elements with CV guidance
        """
        if not pdf_elements:
            return []
        
        if not cv_elements:
            # No CV guidance available, use spatial merging
            logger.debug("No CV guidance available, using spatial merging")
            return self._spatial_merge_elements(pdf_elements)
        
        # Create mapping between PDF elements and CV regions
        element_mapping = self._map_pdf_elements_to_cv_regions(pdf_elements, cv_elements)
        
        # Group PDF elements by CV regions
        cv_groups = self._group_pdf_elements_by_cv_regions(pdf_elements, element_mapping)
        
        # Merge within each CV region
        merged_elements = []
        for cv_region_id, group_elements in cv_groups.items():
            if len(group_elements) == 1:
                # Single element, no merging needed
                element = group_elements[0]
                element.metadata = element.metadata or {}
                element.metadata['merge_method'] = 'cv_guided_single'
                merged_elements.append(element)
            else:
                # Multiple elements, merge them
                merged_element = self._create_merged_element_cv_guided(group_elements, cv_region_id, element_mapping)
                merged_elements.append(merged_element)
        
        # Handle ungrouped elements (not mapped to any CV region)
        ungrouped = cv_groups.get('ungrouped', [])
        if ungrouped:
            # Use spatial merging for ungrouped elements
            spatially_merged = self._spatial_merge_elements(ungrouped)
            for elem in spatially_merged:
                elem.metadata = elem.metadata or {}
                elem.metadata['merge_method'] = 'cv_guided_spatial_fallback'
            merged_elements.extend(spatially_merged)
        
        return merged_elements
    
    def _map_pdf_elements_to_cv_regions(self, 
                                       pdf_elements: List[LayoutElement], 
                                       cv_elements: List[LayoutElement]) -> Dict[int, Dict]:
        """Map PDF elements to CV regions based on spatial overlap."""
        mapping = {}
        
        for pdf_elem in pdf_elements:
            if not pdf_elem.bbox:
                continue
            
            best_match = None
            best_overlap = 0.0
            
            for cv_elem in cv_elements:
                if not cv_elem.bbox:
                    continue
                
                overlap = self._calculate_bbox_overlap(pdf_elem.bbox, cv_elem.bbox)
                if overlap > best_overlap and overlap > 0.1:  # At least 10% overlap
                    best_overlap = overlap
                    best_match = {
                        'cv_element': cv_elem,
                        'overlap': overlap,
                        'element_type': cv_elem.element_type
                    }
            
            if best_match:
                mapping[pdf_elem.id] = best_match
        
        return mapping
    
    def _group_pdf_elements_by_cv_regions(self, 
                                         pdf_elements: List[LayoutElement], 
                                         element_mapping: Dict[int, Dict]) -> Dict[str, List[LayoutElement]]:
        """Group PDF elements by their mapped CV regions."""
        groups = {}
        
        for pdf_elem in pdf_elements:
            if pdf_elem.id in element_mapping:
                cv_elem_id = element_mapping[pdf_elem.id]['cv_element'].id
                group_key = f"cv_{cv_elem_id}"
            else:
                group_key = "ungrouped"
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(pdf_elem)
        
        return groups
    
    def _create_merged_element_cv_guided(self, 
                                       elements: List[LayoutElement], 
                                       cv_region_id: str, 
                                       element_mapping: Dict[int, Dict]) -> LayoutElement:
        """
        Create a merged element with CV guidance.
        
        Args:
            elements: PDF elements to merge
            cv_region_id: CV region identifier
            element_mapping: Mapping from PDF elements to CV regions
            
        Returns:
            Merged element with CV-guided properties
        """
        if len(elements) == 1:
            element = elements[0]
            element.metadata = element.metadata or {}
            element.metadata['merge_method'] = 'cv_guided_single'
            return element
        
        # Use the PDF extractor's merging logic but enhance with CV info
        merged = self.pdf_extractor._create_merged_element(elements)
        
        # Enhance with CV guidance information
        merged.metadata = merged.metadata or {}
        merged.metadata['merge_method'] = 'cv_guided_multi'
        merged.metadata['cv_region_id'] = cv_region_id
        merged.metadata['cv_guidance_used'] = True
        
        # Try to get CV element type for better classification
        if elements and elements[0].id in element_mapping:
            cv_element_type = element_mapping[elements[0].id]['element_type']
            # Use CV element type if it's more specific than the current type
            if merged.element_type == ElementType.TEXT and cv_element_type != ElementType.TEXT:
                merged.element_type = cv_element_type
                merged.metadata['element_type_source'] = 'cv_guidance'
        
        return merged
    
    def _spatial_merge_elements(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """Use PDF extractor's spatial merging for elements."""
        if not elements:
            return []
        
        # Use the PDF extractor's merging logic
        temp_extractor = PdfLayoutExtractor(merge_fragments=True)
        return temp_extractor._merge_fragmented_elements(elements)
    
    def _calculate_bbox_overlap(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        # Calculate intersection
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
        area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)

        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    def get_supported_formats(self) -> List[str]:
        """Get supported input formats."""
        return ['.pdf']
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about the hybrid extractor."""
        info = {
            'name': 'PdfStyleCVMixLayoutExtractor',
            'version': '1.0.0',
            'description': 'Hybrid extractor combining PDF-native style extraction with CV guidance',
            'supported_formats': self.get_supported_formats(),
            'extraction_method': f'hybrid_{self.merge_strategy}',
            'merge_strategy': self.merge_strategy,
            'cv_guidance_enabled': self.use_cv_guidance,
            'cv_guidance_weight': self.cv_guidance_weight,
            'fragment_merging': self.merge_fragments,
            'components': {
                'pdf_extractor': self.pdf_extractor.get_detector_info() if self.pdf_extractor else None,
                'cv_detector': self.cv_detector.get_detector_info() if self.cv_detector else None
            }
        }
        
        return info
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        return self._is_pdf_input(input_data)


# Unit test
if __name__ == "__main__":
    extractor = PdfStyleCVMixLayoutExtractor(
        merge_fragments=True,
        use_cv_guidance=True,
        merge_strategy="cv_guided"
    )
    
    result = extractor._detect_layout("tests/test_data/1-1 买卖合同（通用版）.pdf")
    
    import json
    json.dump(result.model_dump(), open("hybrid_extraction_result.json", "w"), indent=2, ensure_ascii=False)
    
    print(f"Hybrid extraction completed: {len(result.elements)} elements")
    print(f"Extraction method: {result.metadata.get('extraction_method')}")
    print(f"Strategy: {result.metadata.get('hybrid_strategy')}") 