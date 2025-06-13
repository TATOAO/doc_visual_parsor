"""
Hybrid PDF Style and CV Mix Layout Extractor

This module provides a hybrid implementation that combines CV-based layout detection
with PDF-native content enrichment for improved document analysis.
"""

import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
from copy import deepcopy

from ..base.base_layout_extractor import BaseLayoutExtractor
from ..visual_detection.cv_detector import CVLayoutDetector
from .pdf_layout_extractor import PdfLayoutExtractor
from models.schemas.layout_schemas import (
    LayoutExtractionResult,
    LayoutElement,
    BoundingBox,
    ElementType
)
from ..utils.layout_processing import sort_elements_by_position

logger = logging.getLogger(__name__)

InputDataType = Union[str, Path, bytes, Any]

class PdfStyleCVMixLayoutExtractor(BaseLayoutExtractor):
    """
    Hybrid Layout Extractor using CV-first approach with PDF content enrichment.
    
    This extractor uses a clean architecture where:
    1. CVLayoutDetector handles primary layout detection
    2. PdfLayoutExtractor enriches CV-detected regions with content and metadata
    """

    def __init__(self, 
                 cv_confidence_threshold: float = 0.5,
                 cv_model_name: str = "docstructbench",
                 cv_image_size: int = 1024,
                 cv_pdf_dpi: int = 150,
                 device: str = "auto"):
        """
        Initialize the hybrid extractor.
        
        Args:
            cv_confidence_threshold: Confidence threshold for CV detection
            cv_model_name: Name of the CV model to use
            cv_image_size: Input image size for CV model
            cv_pdf_dpi: DPI for PDF to image conversion
            device: Device to use for processing
        """
        super().__init__(confidence_threshold=cv_confidence_threshold, device=device)
        
        self.cv_confidence_threshold = cv_confidence_threshold
        self.cv_model_name = cv_model_name
        self.cv_image_size = cv_image_size
        self.cv_pdf_dpi = cv_pdf_dpi
        self.device = device
        self._initialize_detector()

    def _initialize_detector(self) -> None:
        # Initialize CV detector for primary layout detection
        self.cv_detector = CVLayoutDetector(
            model_name=self.cv_model_name,
            confidence_threshold=self.cv_confidence_threshold,
            image_size=self.cv_image_size,
            pdf_dpi=self.cv_pdf_dpi,
            device=self.device
        )
        
        # Initialize PDF extractor for content enrichment
        self.pdf_extractor = PdfLayoutExtractor(
            merge_fragments=False,  # We don't need PDF's merging
            device=self.device
        )
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize the component detectors."""
        try:
            self.cv_detector._initialize_detector()
            logger.info("CV detector initialized successfully")
            
            # PDF extractor doesn't need explicit initialization
            logger.info("PDF extractor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    def _detect_layout(self, 
                      input_data: InputDataType,
                      **kwargs) -> LayoutExtractionResult:
        """
        Core hybrid extraction method using CV-first approach.
        
        Args:
            input_data: Input data (PDF file path, bytes, etc.)
            **kwargs: Additional extraction parameters
            
        Returns:
            LayoutExtractionResult with hybrid extraction results
        """
        try:
            # Step 1: Get CV-based layout detection
            logger.info("Performing CV-based layout detection")
            cv_result = self.cv_detector._detect_layout(input_data, **kwargs)

            # save cv_reuslt to json for debug
            import json
            json.dump(cv_result.model_dump(), open("cv_result.json", "w"), indent=2, ensure_ascii=False)
            
            if not cv_result.elements:
                logger.warning("No elements detected by CV detector")
                return cv_result
            
            # Step 2: Get PDF content for enrichment
            logger.info("Extracting PDF content for enrichment")
            pdf_result = self.pdf_extractor._detect_layout(input_data, **kwargs)
            
            # Step 3: Process each page
            enriched_elements = []
            for page_num in self._get_page_numbers(cv_result):
                logger.info(f"Processing page {page_num}")
                
                # Get elements for this page
                page_cv_elements = self._get_page_elements(cv_result, page_num)
                page_pdf_elements = self._get_page_elements(pdf_result, page_num)
                
                # Enrich CV elements with PDF content
                page_enriched = self._enrich_cv_elements_with_pdf(
                    page_cv_elements,
                    page_pdf_elements
                )
                enriched_elements.extend(page_enriched)
            
            # Step 4: Post-process elements
            processed_elements = self._post_process_elements(enriched_elements)
            
            # Step 5: Create final result
            return self._create_final_result(processed_elements, cv_result, pdf_result)
            
        except Exception as e:
            logger.error(f"Hybrid extraction failed: {str(e)}")
            return LayoutExtractionResult(elements=[], metadata={"error": str(e)})
    
    def _get_page_numbers(self, result: LayoutExtractionResult) -> List[int]:
        """Get unique page numbers from result elements."""
        page_numbers = set()
        for element in result.elements:
            if element.metadata and 'page_number' in element.metadata:
                page_numbers.add(element.metadata['page_number'])
        return sorted(list(page_numbers))
    
    def _get_page_elements(self, 
                          result: LayoutExtractionResult, 
                          page_num: int) -> List[LayoutElement]:
        """Get elements for a specific page."""
        return [
            elem for elem in result.elements
            if elem.metadata and elem.metadata.get('page_number') == page_num
        ]
    
    def _enrich_cv_elements_with_pdf(self,
                                   cv_elements: List[LayoutElement],
                                   pdf_elements: List[LayoutElement]) -> List[LayoutElement]:
        """
        Enrich CV-detected elements with PDF content and metadata.
        
        Args:
            cv_elements: CV-detected elements for a page
            pdf_elements: PDF elements for the same page
            
        Returns:
            List of enriched elements
        """
        enriched_elements = []
        
        for cv_elem in cv_elements:
            # Find overlapping PDF elements
            overlapping_pdf = self._find_overlapping_pdf_elements(cv_elem, pdf_elements)
            
            if not overlapping_pdf:
                # Keep CV element as is if no PDF content found
                enriched_elements.append(cv_elem)
                continue
            
            # Create enriched element
            enriched = self._create_enriched_element(cv_elem, overlapping_pdf)
            enriched_elements.append(enriched)
        
        return enriched_elements
    
    def _find_overlapping_pdf_elements(self,
                                     cv_element: LayoutElement,
                                     pdf_elements: List[LayoutElement],
                                     min_overlap: float = 0.1) -> List[LayoutElement]:
        """
        Find PDF elements that overlap with a CV element.
        
        Args:
            cv_element: CV-detected element
            pdf_elements: List of PDF elements to check
            min_overlap: Minimum overlap ratio to consider
            
        Returns:
            List of overlapping PDF elements
        """
        if not cv_element.bbox:
            return []
        
        overlapping = []
        for pdf_elem in pdf_elements:
            if not pdf_elem.bbox:
                continue
            
            overlap = self._calculate_bbox_overlap(cv_element.bbox, pdf_elem.bbox)
            if overlap >= min_overlap:
                overlapping.append(pdf_elem)
        
        return overlapping
    
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
    
    def _create_enriched_element(self,
                               cv_element: LayoutElement,
                               pdf_elements: List[LayoutElement]) -> LayoutElement:
        """
        Create an enriched element combining CV layout with PDF content.
        
        Args:
            cv_element: CV-detected element
            pdf_elements: Overlapping PDF elements
            
        Returns:
            Enriched element
        """
        # Start with CV element as base
        enriched = deepcopy(cv_element)
        
        # Extract and merge text content
        text_content = self._merge_text_content(pdf_elements)
        enriched.text = text_content
        
        # Extract and merge style information
        style_info = self._merge_style_info(pdf_elements)
        enriched.metadata = enriched.metadata or {}
        enriched.metadata.update({
            'style_info': style_info,
            'source_pdf_elements': [e.id for e in pdf_elements],
            'enrichment_method': 'cv_first_pdf_enriched'
        })
        
        return enriched
    
    def _merge_text_content(self, pdf_elements: List[LayoutElement]) -> str:
        """
        Merge text content from multiple PDF elements.
        
        Args:
            pdf_elements: List of PDF elements to merge
            
        Returns:
            Merged text content
        """
        # Sort elements by position (top-to-bottom, left-to-right)
        sorted_elements = sort_elements_by_position(pdf_elements)
        
        # Merge text with appropriate spacing
        text_parts = []
        for elem in sorted_elements:
            if elem.text:
                text_parts.append(elem.text)
        
        return ' '.join(text_parts)
    
    def _merge_style_info(self, pdf_elements: List[LayoutElement]) -> Dict:
        """
        Merge style information from PDF elements.
        
        Args:
            pdf_elements: List of PDF elements to merge
            
        Returns:
            Merged style information
        """
        style_info = {}
        
        # Collect all style information
        for elem in pdf_elements:
            if elem.metadata and 'style_info' in elem.metadata:
                style_info.update(elem.metadata['style_info'])
        
        return style_info
    
    def _post_process_elements(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """
        Post-process enriched elements to improve quality.
        
        Args:
            elements: List of elements to process
            
        Returns:
            Processed elements
        """
        processed = []
        for elem in elements:
            # Clean up text content
            if elem.text:
                elem.text = self._clean_text_content(elem.text)
            
            # Validate and fix element type
            elem.element_type = self._validate_element_type(elem)
            
            # Clean up metadata
            if elem.metadata:
                elem.metadata = self._clean_metadata(elem.metadata)
            
            processed.append(elem)
        
        return processed
    
    def _clean_text_content(self, text: str) -> str:
        """Clean up text content."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _validate_element_type(self, element: LayoutElement) -> ElementType:
        """Validate and fix element type based on content and metadata."""
        # Keep CV-detected type if it's not TEXT
        if element.element_type != ElementType.TEXT:
            return element.element_type
        
        # Try to determine type from metadata
        if element.metadata and 'style_info' in element.metadata:
            style_info = element.metadata['style_info']
            
            # Check for title-like formatting
            if style_info.get('font_size', 0) > 12:  # Example threshold
                return ElementType.TITLE
            
            # Add more type detection logic here
        
        return ElementType.TEXT
    
    def _clean_metadata(self, metadata: Dict) -> Dict:
        """Clean up metadata dictionary."""
        # Remove None values
        return {k: v for k, v in metadata.items() if v is not None}
    
    def _create_final_result(self,
                           elements: List[LayoutElement],
                           cv_result: LayoutExtractionResult,
                           pdf_result: LayoutExtractionResult) -> LayoutExtractionResult:
        """
        Create final extraction result.
        
        Args:
            elements: Processed elements
            cv_result: Original CV detection result
            pdf_result: Original PDF extraction result
            
        Returns:
            Final LayoutExtractionResult
        """
        metadata = {
            'extraction_method': 'hybrid_cv_first_pdf_enriched',
            'total_elements': len(elements),
            'cv_elements': len(cv_result.elements),
            'pdf_elements': len(pdf_result.elements),
            'document_type': 'pdf',
            'page_count': len(self._get_page_numbers(cv_result))
        }
        
        return LayoutExtractionResult(elements=elements, metadata=metadata)
    
    def get_supported_formats(self) -> List[str]:
        """Get supported input formats."""
        return ['.pdf']
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about the hybrid extractor."""
        return {
            'name': 'PdfStyleCVMixLayoutExtractor',
            'version': '2.0.0',
            'description': 'CV-first hybrid extractor with PDF content enrichment',
            'supported_formats': self.get_supported_formats(),
            'extraction_method': 'hybrid_cv_first_pdf_enriched',
            'components': {
                'cv_detector': self.cv_detector.get_detector_info() if self.cv_detector else None,
                'pdf_extractor': self.pdf_extractor.get_detector_info() if self.pdf_extractor else None
            }
        }
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if isinstance(input_data, (str, Path)):
            return str(input_data).lower().endswith('.pdf')
        return False


# Unit test
# python -m models.layout_detection.layout_extraction.pdf_style_cv_mix_extractor
if __name__ == "__main__":
    extractor = PdfStyleCVMixLayoutExtractor(
        cv_confidence_threshold=0.5,
        cv_model_name="docstructbench",
        cv_image_size=1024,
        cv_pdf_dpi=150
    )
    
    result = extractor._detect_layout("tests/test_data/1-1 买卖合同（通用版）.pdf")
    
    import json
    json.dump(result.model_dump(), open("hybrid_extraction_result.json", "w"), indent=2, ensure_ascii=False)
    
    print(f"Hybrid extraction completed: {len(result.elements)} elements")
    print(f"Extraction method: {result.metadata.get('extraction_method')}")
    print(f"Page count: {result.metadata.get('page_count')}") 


    # visualize the result
    from models.layout_detection.utils.visualiza_layout_elements import visualize_pdf_layout
    visualize_pdf_layout("tests/test_data/1-1 买卖合同（通用版）.pdf", result, "hybrid_extraction_result.pdf")