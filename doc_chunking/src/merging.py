"""
Hybrid PDF Style and CV Mix Layout Extractor

This module provides a hybrid implementation that combines CV-based layout detection
with PDF-native content enrichment for improved document analysis.
"""

import logging
import os
import tempfile
import io
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
from copy import deepcopy
from PIL import Image

try:
    import fitz  # PyMuPDF for PDF support
except ImportError:
    fitz = None

from .onnx_layout_detector import ONNXDocLayoutYOLO
from .schemas import (
    LayoutExtractionResult,
    LayoutElement,
    BoundingBox,
    ElementType,
    RunInfo,
    StyleInfo,
    FontInfo,
    ParagraphFormat
)
from .utils import calculate_bbox_overlap, is_bbox_contained

logger = logging.getLogger(__name__)

# Type alias for input data
InputDataType = Union[str, Path, bytes, Any]


class PdfLayoutExtractor:
    """
    PDF Layout Extractor using PyMuPDF for content extraction.
    
    This extractor focuses on extracting text content and formatting information
    from PDF documents using PyMuPDF's native capabilities.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the PDF extractor.
        
        Args:
            device: Device to use for processing (not used for PDF extraction)
        """
        self.device = device
        
        try:
            import fitz  # PyMuPDF
            self.fitz = fitz
        except ImportError:
            raise ImportError("PyMuPDF is required for PDF processing")
    
    def extract_layout(self, input_data: InputDataType) -> LayoutExtractionResult:
        """
        Extract layout elements from PDF document.
        
        Args:
            input_data: PDF input data
            
        Returns:
            LayoutExtractionResult containing extracted elements
        """
        try:
            # Open PDF document
            if isinstance(input_data, (str, Path)):
                doc = self.fitz.open(str(input_data))
            else:
                # Handle bytes or file-like objects
                doc = self.fitz.open(stream=input_data, filetype="pdf")
            
            all_elements = []
            element_id = 0
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text blocks with formatting
                blocks = page.get_text("dict")
                
                for block in blocks.get("blocks", []):
                    if "lines" in block:  # Text block
                        for line in block["lines"]:
                            for span in line["spans"]:
                                # Create layout element for each text span
                                bbox = BoundingBox(
                                    x1=span["bbox"][0],
                                    y1=span["bbox"][1], 
                                    x2=span["bbox"][2],
                                    y2=span["bbox"][3]
                                )
                                
                                # Extract font information
                                font_info = FontInfo(
                                    name=span.get("font"),
                                    size=span.get("size"),
                                    bold=span.get("flags", 0) & 2**4 != 0,  # Bold flag
                                    italic=span.get("flags", 0) & 2**1 != 0,  # Italic flag
                                    color=f"#{span.get('color', 0):06x}" if span.get('color') else None
                                )
                                
                                # Create style information
                                style_info = StyleInfo(
                                    font=font_info,
                                    runs=[RunInfo(
                                        text=span["text"],
                                        start_index=0,
                                        end_index=len(span["text"]),
                                        font=font_info
                                    )]
                                )
                                
                                element = LayoutElement(
                                    id=element_id,
                                    element_type=ElementType.PLAIN_TEXT,  # Default type
                                    text=span["text"],
                                    bbox=bbox,
                                    confidence=1.0,  # PDF extraction is deterministic
                                    style=style_info,
                                    metadata={
                                        'page_number': page_num,
                                        'source_type': 'pdf_extraction',
                                        'block_id': block.get("number", -1),
                                        'line_id': line.get("number", -1)
                                    }
                                )
                                
                                all_elements.append(element)
                                element_id += 1
            
            doc.close()
            
            return LayoutExtractionResult(elements=all_elements)
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise
    
    def _sort_elements_by_reading_order(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """
        Sort elements by reading order, handling superscripts and subscripts.
        
        Args:
            elements: List of layout elements
            
        Returns:
            Sorted list of elements
        """
        if not elements:
            return elements
        
        # Sort by y-coordinate first, then by x-coordinate
        # Handle superscripts by giving them a slight y-offset
        def sort_key(elem):
            if not elem.bbox:
                return (0, 0)
            
            y_offset = 0
            if elem.style and elem.style.font and elem.style.font.superscript:
                y_offset = -0.1  # Slight upward adjustment for superscripts
            elif elem.style and elem.style.font and elem.style.font.subscript:
                y_offset = 0.1   # Slight downward adjustment for subscripts
            
            return (elem.bbox.y1 + y_offset, elem.bbox.x1)
        
        return sorted(elements, key=sort_key)


class PdfStyleCVMixLayoutExtractor:
    """
    Hybrid Layout Extractor using CV-first approach with PDF content enrichment.
    
    This extractor uses a clean architecture where:
    1. ONNXLayoutDetector handles primary layout detection
    2. PdfLayoutExtractor enriches CV-detected regions with content and metadata
    """

    def __init__(self, 
                 model_path: str,
                 cv_confidence_threshold: float = 0.25,
                 cv_image_size: int = 1024,
                 cv_pdf_dpi: int = 150,
                 device: str = "auto",
                 need_initialize: bool = True):
        """
        Initialize the hybrid extractor.
        
        Args:
            model_path: Path to the ONNX model file
            cv_confidence_threshold: Confidence threshold for CV detection
            cv_model_name: Name of the CV model to use
            cv_image_size: Input image size for CV model
            cv_pdf_dpi: DPI for PDF to image conversion
            device: Device to use for processing
        """
        self.cv_confidence_threshold = cv_confidence_threshold
        self.cv_image_size = cv_image_size
        self.cv_pdf_dpi = cv_pdf_dpi
        self.device = device

        # Initialize ONNX detector for primary layout detection
        self.cv_detector = ONNXDocLayoutYOLO(
            model_path=model_path,
            device=device
        )
        
        # Initialize PDF extractor for content enrichment
        self.pdf_extractor = PdfLayoutExtractor(device=device)
        
    def _pdf_to_image(self, pdf_path: str, page_num: int = 0, dpi: int = 150) -> str:
        """
        Convert a PDF page to image for ONNX analysis.
        
        Args:
            pdf_path (str): Path to the PDF file
            page_num (int): Page number to convert (0-indexed)
            dpi (int): DPI for the conversion
            
        Returns:
            str: Path to the temporary image file
        """
        if fitz is None:
            raise ImportError("PyMuPDF is required for PDF processing. Install with: pip install PyMuPDF")
        
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            
            # Calculate zoom factor based on desired DPI
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            
            # Render page as image
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_file.write(img_data)
                temp_path = tmp_file.name
            
            doc.close()
            return temp_path
            
        except Exception as e:
            raise ValueError(f"Could not convert PDF page to image: {e}")
    
    def _is_pdf_file(self, input_data: InputDataType) -> bool:
        """Check if input data is a PDF file."""
        if isinstance(input_data, (str, Path)):
            return str(input_data).lower().endswith('.pdf')
        return False
    
    def detect_layout(self, 
                      input_data: InputDataType,
                      confidence_threshold: Optional[float] = None,
                      **kwargs) -> LayoutExtractionResult:
        """
        Detect layout using hybrid CV + PDF approach.
        
        Args:
            input_data: Input data (PDF file path or bytes)
            confidence_threshold: Override default confidence threshold
            **kwargs: Additional detection parameters
            
        Returns:
            LayoutExtractionResult containing detected and enriched elements
        """
        
        # Use provided threshold or default
        threshold = confidence_threshold if confidence_threshold is not None else self.cv_confidence_threshold
        
        try:
            # Step 1: Extract PDF content using PyMuPDF
            logger.info("Extracting PDF content...")
            pdf_result = self.pdf_extractor.extract_layout(input_data)
            logger.info(f"Extracted {len(pdf_result.elements)} PDF elements")
            
            # Step 2: Detect layout using ONNX model
            logger.info("Detecting layout with ONNX model...")
            
            # Convert PDF to image if needed
            cv_input = input_data
            temp_image_path = None
            if self._is_pdf_file(input_data):
                logger.info("Converting PDF to image for ONNX processing...")
                temp_image_path = self._pdf_to_image(str(input_data))
                cv_input = temp_image_path
            
            cv_result = self.cv_detector.detect_layout(cv_input, threshold, **kwargs)
            logger.info(f"Detected {len(cv_result.elements)} CV elements")
            
            # Clean up temporary image file
            if temp_image_path and os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
            
            # Step 3: Enrich CV elements with PDF content
            logger.info("Enriching CV elements with PDF content...")
            enriched_elements = self._enrich_cv_elements_with_pdf(
                cv_elements=cv_result.elements,
                pdf_elements=pdf_result.elements
            )
            logger.info(f"Created {len(enriched_elements)} enriched elements")
            
            return LayoutExtractionResult(
                elements=enriched_elements,
                metadata={
                    'extraction_method': 'hybrid_cv_pdf',
                    'cv_elements_count': len(cv_result.elements),
                    'pdf_elements_count': len(pdf_result.elements),
                    'enriched_elements_count': len(enriched_elements)
                }
            )
            
        except Exception as e:
            logger.error(f"Hybrid layout detection failed: {str(e)}")
            raise
    
    def _enrich_cv_elements_with_pdf(self,
                                   cv_elements: List[LayoutElement],
                                   pdf_elements: List[LayoutElement]) -> List[LayoutElement]:
        """
        Enrich CV-detected elements with PDF content and formatting.
        
        Args:
            cv_elements: CV-detected layout elements
            pdf_elements: PDF-extracted content elements
            
        Returns:
            List of enriched layout elements
        """
        enriched_elements = []
        
        for cv_element in cv_elements:
            if not cv_element.bbox:
                # Skip elements without bounding boxes
                enriched_elements.append(cv_element)
                continue
            
            # Find overlapping PDF elements
            overlapping_pdf_elements = self._find_overlapping_pdf_elements(
                cv_element, pdf_elements
            )
            
            if overlapping_pdf_elements:
                # Create enriched element
                enriched_element = self._create_enriched_element(
                    cv_element, overlapping_pdf_elements
                )
                enriched_elements.append(enriched_element)
            else:
                # No overlapping PDF elements, keep original CV element
                enriched_elements.append(cv_element)
        
        return enriched_elements
    
    def _find_overlapping_pdf_elements(self,
                                     cv_element: LayoutElement,
                                     pdf_elements: List[LayoutElement]) -> List[LayoutElement]:
        """
        Find PDF elements that overlap with the CV element.
        
        Args:
            cv_element: CV-detected element
            pdf_elements: List of PDF elements to search
            
        Returns:
            List of overlapping PDF elements
        """
        overlapping_elements = []
        
        for pdf_element in pdf_elements:
            if not pdf_element.bbox:
                continue
            
            # Check if PDF element overlaps with CV element
            overlap_ratio = calculate_bbox_overlap(cv_element.bbox, pdf_element.bbox)
            
            # Use a lower threshold for overlap since we want to capture content
            if overlap_ratio > 0.1:  # 10% overlap threshold
                overlapping_elements.append(pdf_element)
        
        return overlapping_elements
    
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
        
        # Sort PDF elements using improved superscript-aware sorting
        sorted_pdf_elements = self.pdf_extractor._sort_elements_by_reading_order(pdf_elements)
        
        # Extract and merge text content and runs
        text_parts = []
        all_runs = []
        current_offset = 0
        
        for elem in sorted_pdf_elements:
            if elem.text:
                text_parts.append(elem.text)
                
                # Collect runs information if available
                if elem.style and elem.style.runs:
                    logger.debug(f"Found {len(elem.style.runs)} runs in PDF element {elem.id}")
                    # Adjust run indices based on current text position
                    for run in elem.style.runs:
                        adjusted_run = deepcopy(run)
                        if adjusted_run.start_index is not None:
                            adjusted_run.start_index += current_offset
                        if adjusted_run.end_index is not None:
                            adjusted_run.end_index += current_offset
                        all_runs.append(adjusted_run)
                        logger.debug(f"Added run with text: '{run.text[:30]}...' at offset {current_offset}")
                
                current_offset += len(elem.text)
        
        logger.info(f"Collected {len(all_runs)} total runs for enriched element")
        
        # Join text parts
        enriched.text = ''.join(text_parts)
        
        # Create enriched style information with preserved runs
        enriched_style = self._create_enriched_style(pdf_elements, all_runs)
        enriched.style = enriched_style
        
        # Update metadata
        enriched.metadata = enriched.metadata or {}
        enriched.metadata.update({
            'source_pdf_elements': [e.id for e in pdf_elements],
            'enrichment_method': 'cv_first_pdf_enriched',
            'runs_count': len(all_runs),
            'page_number': pdf_elements[-1].metadata.get('page_number') if pdf_elements else None
        })
        
        return enriched
    
    def _create_enriched_style(self, 
                              pdf_elements: List[LayoutElement], 
                              all_runs: List[RunInfo]) -> StyleInfo:
        """
        Create enriched style information from PDF elements.
        
        Args:
            pdf_elements: List of PDF elements
            all_runs: List of all text runs
            
        Returns:
            Enriched style information
        """
        if not pdf_elements:
            return None
        
        # Use the first element's style as base
        base_style = pdf_elements[0].style
        
        if not base_style:
            return None
        
        # Create enriched style with all runs
        enriched_style = StyleInfo(
            font=base_style.font,
            paragraph_format=base_style.paragraph_format,
            runs=all_runs
        )
        
        return enriched_style

# python -m doc_chunking.src.merging 
if __name__ == "__main__":
    # pdf_layout_extractor = PdfLayoutExtractor()
    # result = pdf_layout_extractor.extract_layout("3800.pdf")

    pdf_style_cv_mix_layout_extractor = PdfStyleCVMixLayoutExtractor(
        model_path="model_parameters/layout_detection/docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx"
    )
    result = pdf_style_cv_mix_layout_extractor.detect_layout("3800.pdf")
    print(result)
