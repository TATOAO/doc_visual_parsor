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
from PIL import Image, ImageDraw

try:
    import fitz  # PyMuPDF for PDF support
except ImportError:
    fitz = None

from .onnx_layout_detector import ONNXDocLayoutYOLO
from .utils import sort_elements_by_position
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
    
    def extract_layout_for_page(self, page, page_num: int, element_id_start: int, target_dpi: int = 150) -> Tuple[List[LayoutElement], int]:
        """
        Extract layout elements from a single PDF page.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            element_id_start: Starting element ID for this page
            target_dpi: Target DPI for coordinate scaling (should match CV detection DPI)
            
        Returns:
            Tuple of (list of layout elements, next element ID)
        """
        page_elements = []
        element_id = element_id_start
        
        # Calculate scaling factor from PDF coordinates (72 DPI) to target DPI
        scale_factor = target_dpi / 72.0
        
        # Extract text blocks with formatting
        blocks = page.get_text("dict")
        
        for block in blocks.get("blocks", []):
            if "lines" in block:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        # Scale coordinates from PDF coordinate system (72 DPI) to target DPI
                        scaled_bbox = [
                            span["bbox"][0] * scale_factor,
                            span["bbox"][1] * scale_factor,
                            span["bbox"][2] * scale_factor,
                            span["bbox"][3] * scale_factor
                        ]
                        
                        # Create layout element for each text span
                        bbox = BoundingBox(
                            x1=scaled_bbox[0],
                            y1=scaled_bbox[1], 
                            x2=scaled_bbox[2],
                            y2=scaled_bbox[3]
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
                        
                        page_elements.append(element)
                        element_id += 1
        
        return page_elements, element_id

    def extract_layout(self, input_data: InputDataType, max_pages: Optional[int] = None, target_dpi: int = 150) -> LayoutExtractionResult:
        """
        Extract layout elements from PDF document.
        
        Args:
            input_data: PDF input data
            max_pages: Maximum number of pages to process (None for all pages)
            target_dpi: Target DPI for coordinate scaling (should match CV detection DPI)
            
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
            num_pages = len(doc)
            if max_pages is not None:
                num_pages = min(num_pages, max_pages)
            
            for page_num in range(num_pages):
                page = doc[page_num]
                
                # Extract elements for this page
                page_elements, element_id = self.extract_layout_for_page(
                    page, page_num, element_id, target_dpi
                )
                
                all_elements.extend(page_elements)
                logger.info(f"Page {page_num + 1}: extracted {len(page_elements)} elements")
            
            doc.close()
            
            return LayoutExtractionResult(elements=all_elements)
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise

    # Backward compatibility for older scripts
    def _detect_layout(self, input_data: InputDataType, *_, **__) -> LayoutExtractionResult:
        """Alias for legacy API compatibility. Delegates to extract_layout."""
        return self.extract_layout(input_data)
    
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
                 model_path: Optional[str] = None,
                 cv_confidence_threshold: float = 0.1,  # Lowered from 0.25 to capture more detections
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

        # Resolve default model path if not provided
        resolved_model_path = model_path
        if resolved_model_path is None:
            # Default to repo model path
            default_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'model_parameters', 'layout_detection',
                'docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx'
            )
            # Fallback to relative if absolute resolution fails
            if not os.path.exists(default_path):
                default_path = 'model_parameters/layout_detection/docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx'
            resolved_model_path = default_path

        # Initialize ONNX detector for primary layout detection
        self.cv_detector = ONNXDocLayoutYOLO(
            model_path=resolved_model_path,
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
                      max_pages: Optional[int] = None,
                      **kwargs) -> LayoutExtractionResult:
        """
        Detect layout using hybrid CV + PDF approach.
        
        Args:
            input_data: Input data (PDF file path or bytes)
            confidence_threshold: Override default confidence threshold
            max_pages: Maximum number of pages to process (None for all pages)
            **kwargs: Additional detection parameters
            
        Returns:
            LayoutExtractionResult containing detected and enriched elements
        """
        
        # Use provided threshold or default
        threshold = confidence_threshold if confidence_threshold is not None else self.cv_confidence_threshold
        
        try:
            # Step 1: Extract PDF content using PyMuPDF
            logger.info("Extracting PDF content...")
            # Use same DPI as CV detection for coordinate alignment
            
            # Step 2: Detect layout using ONNX model for each page
            logger.info("Detecting layout with ONNX model...")
            
            all_cv_elements = []
            all_pdf_elements = []
            all_enriched_elements = []
            temp_image_paths = []
            
            if self._is_pdf_file(input_data):
                # Process multiple pages
                doc = fitz.open(str(input_data))
                num_pages = len(doc)
                if max_pages is not None:
                    num_pages = min(num_pages, max_pages)
                
                logger.info(f"Processing {num_pages} pages...")
                
                for page_num in range(num_pages):
                    logger.info(f"Processing page {page_num + 1}/{num_pages}...")
                    
                    # Convert PDF page to image first
                    temp_image_path = self._pdf_to_image(str(input_data), page_num)
                    temp_image_paths.append(temp_image_path)
                    
                    pdf_elements, next_id = self.pdf_extractor.extract_layout_for_page(doc[page_num], page_num, 0, 150)
                    logger.info(f"Extracted {len(pdf_elements)} PDF elements")
                    pdf_result = LayoutExtractionResult(elements=pdf_elements)
                    """
                    image = self.display_layout(temp_image_path, pdf_result)
                    image.save(f"pdf_result_{page_num}.png")
                    """

                    all_pdf_elements.extend(pdf_result.elements)

                    
                    # Run ONNX detection on this page
                    cv_result = self.cv_detector.detect_layout(temp_image_path, threshold, **kwargs)

                    # Sort CV elements by reading order (top-to-bottom, left-to-right)
                    cv_result.elements = sort_elements_by_position(cv_result.elements)

                    # display layout for debug 
                    """
                    image = self.display_layout(temp_image_path, cv_result)
                    image.save(f"cv_result_{page_num}.png")
                    """
                    
                    # Update element IDs and add page metadata
                    for element in cv_result.elements:
                        element.id = len(all_cv_elements)
                        if element.metadata is None:
                            element.metadata = {}
                        element.metadata['page_number'] = page_num
                        element.metadata['source_page'] = page_num
                    
                    all_cv_elements.extend(cv_result.elements)
                    logger.info(f"Page {page_num + 1}: detected {len(cv_result.elements)} elements")


                    enriched_elements = self._enrich_cv_elements_with_pdf(
                        cv_elements=cv_result.elements,
                        pdf_elements=pdf_elements
                    )
                    
                    # Update IDs for enriched elements to be unique across all pages
                    for enriched_element in enriched_elements:
                        enriched_element.id = len(all_enriched_elements)
                        if enriched_element.metadata is None:
                            enriched_element.metadata = {}
                        enriched_element.metadata['page_number'] = page_num
                        enriched_element.metadata['source_page'] = page_num
                        all_enriched_elements.append(enriched_element)
                    
                
                doc.close()
                    
            else:
                # Non-PDF input, process directly
                raise Exception("Non-PDF input, process directly")
            
            logger.info(f"Total detected {len(all_cv_elements)} CV elements across all pages")
            logger.info(f"Total created {len(all_enriched_elements)} enriched elements across all pages")
            
            # Clean up temporary image files
            for temp_path in temp_image_paths:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
            # Use the enriched elements from page-by-page processing
            enriched_elements = all_enriched_elements
            logger.info(f"Using {len(enriched_elements)} enriched elements from page-by-page processing")

            
            
            return LayoutExtractionResult(
                elements=enriched_elements,
                metadata={
                    'extraction_method': 'hybrid_cv_pdf',
                    'cv_elements_count': len(all_cv_elements),
                    'pdf_elements_count': len(pdf_result.elements),
                    'enriched_elements_count': len(enriched_elements),
                    'pages_processed': num_pages if self._is_pdf_file(input_data) else 1
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

    # Backward compatibility for older scripts
    def _detect_layout(self, input_data: InputDataType, *_, **kwargs) -> LayoutExtractionResult:
        """Alias for legacy API compatibility. Delegates to detect_layout."""
        return self.detect_layout(input_data, **kwargs)
    
    def display_layout(self, image_path: str, result: LayoutExtractionResult):
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Color mapping for different element types
        color_map = {
            "Title": "red",
            "Heading": "orange", 
            "Plain Text": "blue",
            "Paragraph": "green",
            "Figure": "purple",
            "Figure Caption": "pink",
            "Table": "brown",
            "Table Caption": "yellow",
            "List": "cyan",
            "Isolate Formula": "magenta",
            "Formula Caption": "lime",
            "Table Footnote": "navy",
            "Unknown": "gray",
            "Abandon": "black"
        }
        
        # Track used label positions to avoid overlaps
        used_positions = []
        
        for idx, element in enumerate(result.elements):
            # Convert BoundingBox object to tuple format expected by draw.rectangle
            if element.bbox:
                bbox_coords = (element.bbox.x1, element.bbox.y1, element.bbox.x2, element.bbox.y2)
                # Get color for this element type, default to red if not found
                color = color_map.get(element.element_type.value, "red")
                draw.rectangle(bbox_coords, outline=color, width=2)
                
                # Add text label for element type with index and confidence
                confidence = element.confidence if element.confidence is not None else 0.0
                label = f"{element.element_type.value} #{idx} ({confidence:.2f})"
                
                # Calculate text dimensions
                text_bbox = draw.textbbox((0, 0), label)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Find a good position for the label (avoid overlaps)
                base_x = element.bbox.x1
                base_y = element.bbox.y1 - text_height - 2
                
                # Try different positions to avoid overlaps
                label_x, label_y = self._find_label_position(
                    base_x, base_y, text_width, text_height, used_positions, image.size
                )
                
                # Draw background rectangle for text
                text_bg_coords = (label_x, label_y, label_x + text_width + 4, label_y + text_height + 2)
                draw.rectangle(text_bg_coords, fill=color)
                
                # Draw text label
                draw.text((label_x + 2, label_y), label, fill="white")
                
                # Record this position as used
                used_positions.append((label_x, label_y, label_x + text_width + 4, label_y + text_height + 2))
        
        return image
    
    def _find_label_position(self, base_x, base_y, text_width, text_height, used_positions, image_size):
        """Find a position for the label that doesn't overlap with existing labels."""
        # Try positions in order of preference
        positions_to_try = [
            (base_x, base_y),  # Top-left of bbox
            (base_x, base_y - text_height - 5),  # Above bbox
            (base_x + text_width + 10, base_y),  # Right of bbox
            (base_x, base_y + text_height + 5),  # Below bbox
            (base_x - text_width - 10, base_y),  # Left of bbox
        ]
        
        for pos_x, pos_y in positions_to_try:
            # Check if position is within image bounds
            if (pos_x >= 0 and pos_y >= 0 and 
                pos_x + text_width + 4 <= image_size[0] and 
                pos_y + text_height + 2 <= image_size[1]):
                
                # Check for overlaps with existing labels
                new_rect = (pos_x, pos_y, pos_x + text_width + 4, pos_y + text_height + 2)
                if not self._rectangles_overlap(new_rect, used_positions):
                    return pos_x, pos_y
        
        # If no good position found, use the base position
        return base_x, base_y
    
    def _rectangles_overlap(self, rect1, rect_list):
        """Check if rect1 overlaps with any rectangle in rect_list."""
        x1, y1, x2, y2 = rect1
        for other_rect in rect_list:
            ox1, oy1, ox2, oy2 = other_rect
            # Check if rectangles overlap
            if not (x2 <= ox1 or ox2 <= x1 or y2 <= oy1 or oy2 <= y1):
                return True
        return False

# python -m doc_chunking.src.merging 
if __name__ == "__main__":
    # pdf_layout_extractor = PdfLayoutExtractor()
    # result = pdf_layout_extractor.extract_layout("3800.pdf")

    pdf_style_cv_mix_layout_extractor = PdfStyleCVMixLayoutExtractor(
        model_path="model_parameters/layout_detection/docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx",
        cv_confidence_threshold=0.1  # Use lower threshold for better detection
    )
    result = pdf_style_cv_mix_layout_extractor.detect_layout("3800.pdf")  # Test with first 3 pages
    import json
    json.dump(result.model_dump(), open("result.json", "w"), indent=4, ensure_ascii=False)
