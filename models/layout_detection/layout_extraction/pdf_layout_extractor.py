"""
Document-native PDF Layout Extraction Module

This module provides a document-native implementation of layout extraction that works
directly with PDF document structures using PyMuPDF, extracting raw style information
without converting to images first.
"""

import logging
import tempfile
import os
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
import xml.etree.ElementTree as ET
from zipfile import ZipFile
import re

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("Please install PyMuPDF: pip install PyMuPDF")

from ..base.base_layout_extractor import BaseLayoutExtractor
from models.schemas.layout_schemas import (
    StyleInfo, 
    FontInfo, 
    ParagraphFormat, 
    RunInfo, 
    TextAlignment,
    LayoutExtractionResult,
    LayoutElement,
    BoundingBox,
    ElementType
)
from models.schemas.schemas import Section, Positions, DocumentType

logger = logging.getLogger(__name__)

import numpy as np
from PIL import Image
from typing import BinaryIO

InputDataType = Union[
    str,                    # File path as string
    Path,                   # File path as Path object
    bytes,                  # Raw file content
    np.ndarray,            # Image as numpy array
    Image.Image,           # PIL Image object
    BinaryIO,              # File-like object with read() method
    Any                    # For objects with getvalue() method (uploaded files)
]

class PdfLayoutExtractor(BaseLayoutExtractor):
    """
    Document-native Style Extractor for PDF files with spatial fragment merging.
    
    This extractor works directly with the PDF document's internal structure,
    analyzing text blocks, spans, and formatting to extract raw style information
    and merging fragmented text elements using spatial and font analysis.
    """

    def __init__(self, 
                 merge_fragments: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.merge_fragments = merge_fragments
        self.detector = self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the detector."""
        logger.info("PDF-native layout extractor initialized")
        return None
    
    def _detect_layout(self, 
                      input_data: InputDataType,
                      **kwargs) -> LayoutExtractionResult:
        """
        Core extraction method - analyzes PDF structure to extract raw style information
        and merges fragmented text elements.
        
        Args:
            input_data: Input data (file path, bytes, etc.)
            **kwargs: Additional extraction parameters
            
        Returns:
            LayoutExtractionResult containing merged and structured elements
        """
        try:
            # Load the PDF document
            doc = self._load_pdf_document(input_data)
            
            # Get page count before processing
            page_count = doc.page_count
            
            # Extract raw document structure from all pages
            raw_elements = self._extract_raw_pdf_structure(doc)
            
            # Merge fragmented elements if enabled
            if self.merge_fragments and raw_elements:
                merged_elements = self._merge_fragmented_elements(raw_elements)
            else:
                merged_elements = raw_elements
            
            # Sort elements by reading order (page, then top-to-bottom, left-to-right)
            if merged_elements:
                merged_elements = self._sort_elements_by_reading_order(merged_elements)
            
            # Clean up
            doc.close()
            
            # Create metadata
            metadata = {
                'extraction_method': 'pdf_native_with_merging' if self.merge_fragments else 'pdf_native_raw',
                'total_elements': len(merged_elements),
                'original_elements': len(raw_elements),
                'document_type': 'pdf',
                'page_count': page_count,
                'fragments_merged': self.merge_fragments
            }
            
            return LayoutExtractionResult(elements=merged_elements, metadata=metadata)
            
        except Exception as e:
            logger.error(f"Error in PDF style extraction: {str(e)}")
            # Return empty result on error
            return LayoutExtractionResult(elements=[], metadata={"error": str(e)})
    
    def _load_pdf_document(self, input_data: InputDataType) -> fitz.Document:
        """
        Load PDF document from various input types.
        
        Args:
            input_data: Input data in various formats
            
        Returns:
            PyMuPDF Document object
        """
        if isinstance(input_data, (str, Path)):
            return fitz.open(str(input_data))
        elif isinstance(input_data, bytes):
            return fitz.open(stream=input_data, filetype="pdf")
        elif hasattr(input_data, 'read'):
            # File-like object
            content = input_data.read()
            return fitz.open(stream=content, filetype="pdf")
        elif hasattr(input_data, 'getvalue'):
            # BytesIO or similar
            content = input_data.getvalue()
            return fitz.open(stream=content, filetype="pdf")
        else:
            raise ValueError(f"Unsupported input data type: {type(input_data)}")
    
    def _extract_raw_pdf_structure(self, doc: fitz.Document) -> List['LayoutElement']:
        """
        Extract raw PDF structure with style information only.
        
        Args:
            doc: PyMuPDF Document object
            
        Returns:
            List of LayoutElement objects with raw style information
        """
        elements = []
        element_id = 0
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # Extract text with detailed formatting information
            text_dict = page.get_text("dict")
            
            # Extract drawings (for underlines, etc.)
            drawings = page.get_drawings()
            
            # Process text blocks
            for block in text_dict["blocks"]:
                if "lines" not in block:  # Skip image blocks
                    continue
                
                # Process lines within the block
                for line in block["lines"]:
                    # Process spans within the line
                    for span in line["spans"]:
                        if not span["text"].strip():  # Skip empty spans
                            continue
                        
                        # Extract style information
                        style_info = self._extract_pdf_style_info(span, block, line)
                        
                        # Create bounding box from span coordinates
                        bbox = BoundingBox(
                            x1=span["bbox"][0],
                            y1=span["bbox"][1],
                            x2=span["bbox"][2],
                            y2=span["bbox"][3]
                        )
                        
                        # Determine if this might be a heading based on font characteristics
                        heading_level = self._extract_heading_level_from_pdf_style(span, style_info)
                        
                        # Check for nearby underline drawings
                        nearby_underlines = self._find_nearby_underlines(span, drawings)
                        
                        # Create metadata with raw information
                        metadata = {
                            'page_number': page_num + 1,  # 1-indexed
                            'block_number': block.get('number', 0),
                            'line_number': line.get('number', 0),
                            'span_flags': span.get('flags', 0),
                            'heading_level': heading_level,
                            'is_heading': heading_level > 0,
                            'extraction_method': 'pdf_native',
                            'raw_extraction': True,
                            'nearby_underlines': nearby_underlines,
                            'has_underlines': len(nearby_underlines) > 0,
                            'original_span_bbox': span["bbox"],  # For merging purposes
                            'line_bbox': line.get("bbox", [0, 0, 0, 0]),
                            'block_bbox': block.get("bbox", [0, 0, 0, 0])
                        }
                        
                        # Add font information to metadata
                        if style_info and style_info.primary_font:
                            font = style_info.primary_font
                            metadata.update({
                                'font_name': font.name,
                                'font_size': font.size,
                                'font_bold': font.bold,
                                'font_italic': font.italic,
                                'font_flags': span.get('flags', 0)
                            })
                        
                        # Create layout element with raw information
                        element = LayoutElement(
                            id=element_id,
                            element_type=ElementType.TEXT,  # Default type, classification happens later
                            confidence=1.0,  # Raw extraction has full confidence in style info
                            bbox=bbox,
                            text=span["text"],
                            style=style_info,
                            metadata=metadata
                        )
                        
                        elements.append(element)
                        element_id += 1
        
        return elements

    def _merge_fragmented_elements(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """
        Merge fragmented text elements into logical units using spatial analysis.
        
        Args:
            elements: List of raw text elements
            
        Returns:
            List of merged elements
        """
        if not elements:
            return elements
        
        logger.info(f"Starting spatial fragment merging for {len(elements)} elements...")
        
        # Sort elements by reading order (top to bottom, left to right)
        sorted_elements = sorted(elements, key=lambda e: (e.bbox.y1, e.bbox.x1))
        
        merged = []
        i = 0
        
        while i < len(sorted_elements):
            current = sorted_elements[i]
            merge_candidates = [current]
            
            # Look for elements to merge with current element
            j = i + 1
            while j < len(sorted_elements):
                candidate = sorted_elements[j]
                
                if self._should_merge_elements(current, candidate):
                    merge_candidates.append(candidate)
                    # Update current to be the merged representation for next comparisons
                    current = self._create_merged_element(merge_candidates)
                    sorted_elements.pop(j)  # Remove from list since it's being merged
                    continue
                
                # If we can't merge with this candidate, check if we should stop looking
                if not self._could_potentially_merge(current, candidate):
                    break
                
                j += 1
            
            # Create final merged element
            if len(merge_candidates) > 1:
                merged_element = self._create_merged_element(merge_candidates)
                merged_element.metadata['merged_from_count'] = len(merge_candidates)
                merged_element.metadata['merge_method'] = 'spatial_only'
                merged.append(merged_element)
            else:
                current.metadata['merge_method'] = 'no_merge'
                merged.append(current)
            
            i += 1
        
        logger.info(f"Spatial fragment merging complete: {len(elements)} -> {len(merged)} elements")
        return merged

    def _should_merge_elements(self, elem1: LayoutElement, elem2: LayoutElement) -> bool:
        """
        Determine if two elements should be merged based on spatial criteria.
        
        Args:
            elem1: First element
            elem2: Second element
            
        Returns:
            True if elements should be merged
        """
        # Check if they're on the same page
        if elem1.metadata.get('page_number') != elem2.metadata.get('page_number'):
            return False
        
        # Check font similarity
        if not self._fonts_similar(elem1, elem2):
            return False
        
        # Check spatial proximity
        if not self._spatially_close(elem1, elem2):
            return False
        
        # Check if they form a logical sequence
        if not self._forms_logical_sequence(elem1, elem2):
            return False
        
        return True

    def _fonts_similar(self, elem1: LayoutElement, elem2: LayoutElement) -> bool:
        """
        Check if two elements have similar font characteristics.
        
        Args:
            elem1: First element
            elem2: Second element
            
        Returns:
            True if fonts are similar enough to merge
        """
        # Get font information
        font1 = elem1.style.primary_font if elem1.style else None
        font2 = elem2.style.primary_font if elem2.style else None
        
        if not font1 or not font2:
            return True  # If we can't determine fonts, allow merging
        
        # Check font name similarity
        if font1.name and font2.name:
            if font1.name != font2.name:
                # Allow slight variations in font names
                name1_clean = re.sub(r'[^a-zA-Z]', '', font1.name.lower())
                name2_clean = re.sub(r'[^a-zA-Z]', '', font2.name.lower())
                if name1_clean != name2_clean:
                    return False
        
        # Check font size similarity (allow 10% variation)
        if font1.size and font2.size:
            size_ratio = max(font1.size, font2.size) / min(font1.size, font2.size)
            if size_ratio > 1.1:
                return False
        
        # Check font style consistency
        if font1.bold != font2.bold or font1.italic != font2.italic:
            return False
        
        return True

    def _spatially_close(self, elem1: LayoutElement, elem2: LayoutElement) -> bool:
        """
        Check if two elements are spatially close enough to be merged.
        
        Args:
            elem1: First element
            elem2: Second element
            
        Returns:
            True if elements are spatially close
        """
        bbox1, bbox2 = elem1.bbox, elem2.bbox
        
        # Calculate distances
        horizontal_gap = min(abs(bbox1.x2 - bbox2.x1), abs(bbox2.x2 - bbox1.x1))
        vertical_gap = min(abs(bbox1.y2 - bbox2.y1), abs(bbox2.y2 - bbox1.y1))
        
        # Get average font size for distance thresholds
        font_size = self._get_avg_font_size(elem1, elem2)
        
        # Check if elements are on the same line
        same_line_threshold = font_size * 0.3
        on_same_line = abs(bbox1.y1 - bbox2.y1) <= same_line_threshold
        
        if on_same_line:
            # For same-line elements, check horizontal proximity
            max_horizontal_gap = font_size * 0.5
            return horizontal_gap <= max_horizontal_gap
        
        # Check if elements are in vertical sequence (same column)
        x_center1 = (bbox1.x1 + bbox1.x2) / 2
        x_center2 = (bbox2.x1 + bbox2.x2) / 2
        same_column_threshold = font_size * 0.7
        
        if abs(x_center1 - x_center2) <= same_column_threshold:
            # For same-column elements, check vertical proximity
            max_vertical_gap = font_size * 1.5
            return vertical_gap <= max_vertical_gap
        
        return False

    def _forms_logical_sequence(self, elem1: LayoutElement, elem2: LayoutElement) -> bool:
        """
        Check if two elements form a logical text sequence.
        
        Args:
            elem1: First element
            elem2: Second element
            
        Returns:
            True if elements form a logical sequence
        """
        text1 = elem1.text.strip()
        text2 = elem2.text.strip()
        
        if not text1 or not text2:
            return True  # Allow merging of empty elements
        
        # Check for word fragments (hyphenation)
        if text1.endswith('-') and not text2[0].isupper():
            return True
        
        # Check for sentence continuation
        if not text1.endswith(('.', '!', '?', ':', ';')):
            if not text2[0].isupper():
                return True
            if text2[0].islower():
                return True
        
        # Check for number/list continuation
        if re.match(r'^\d+\.?\s*', text1) or re.match(r'^[a-zA-Z]\.?\s*', text1):
            return False  # Don't merge list items
        
        # Allow merging of short fragments
        if len(text1) < 3 or len(text2) < 3:
            return True
        
        return False

    def _could_potentially_merge(self, elem1: LayoutElement, elem2: LayoutElement) -> bool:
        """
        Quick check if elements could potentially be merged.
        
        Args:
            elem1: First element
            elem2: Second element
            
        Returns:
            True if elements could potentially be merged
        """
        bbox1, bbox2 = elem1.bbox, elem2.bbox
        font_size = self._get_avg_font_size(elem1, elem2)
        
        # If elements are too far apart, stop looking
        max_distance = font_size * 5
        
        vertical_distance = abs(bbox1.y1 - bbox2.y1)
        horizontal_distance = abs(bbox1.x1 - bbox2.x1)
        
        return vertical_distance <= max_distance and horizontal_distance <= max_distance

    def _get_avg_font_size(self, elem1: LayoutElement, elem2: LayoutElement) -> float:
        """
        Get average font size between two elements.
        
        Args:
            elem1: First element
            elem2: Second element
            
        Returns:
            Average font size or default if not available
        """
        size1 = elem1.style.primary_font.size if elem1.style and elem1.style.primary_font else 12.0
        size2 = elem2.style.primary_font.size if elem2.style and elem2.style.primary_font else 12.0
        
        return (size1 + size2) / 2

    def _create_merged_element(self, elements: List[LayoutElement]) -> LayoutElement:
        """
        Create a merged element from a list of elements.
        
        Args:
            elements: List of elements to merge
            
        Returns:
            New merged element
        """
        if len(elements) == 1:
            return elements[0]
        
        # Sort elements by position for proper text ordering
        sorted_elements = sorted(elements, key=lambda e: (e.bbox.y1, e.bbox.x1))
        
        # Merge text content
        merged_text_parts = []
        for i, elem in enumerate(sorted_elements):
            text = elem.text.strip()
            if text:
                # Handle hyphenation
                if i > 0 and sorted_elements[i-1].text.strip().endswith('-'):
                    if merged_text_parts:
                        merged_text_parts[-1] = merged_text_parts[-1].rstrip('-')
                    merged_text_parts.append(text)
                else:
                    # Add space if needed
                    if merged_text_parts and not merged_text_parts[-1].endswith(' '):
                        merged_text_parts.append(' ')
                    merged_text_parts.append(text)
        
        merged_text = ''.join(merged_text_parts).strip()
        
        # Create merged bounding box
        min_x1 = min(e.bbox.x1 for e in elements)
        min_y1 = min(e.bbox.y1 for e in elements)
        max_x2 = max(e.bbox.x2 for e in elements)
        max_y2 = max(e.bbox.y2 for e in elements)
        
        merged_bbox = BoundingBox(x1=min_x1, y1=min_y1, x2=max_x2, y2=max_y2)
        
        # Use style from the first element
        merged_style = sorted_elements[0].style
        
        # Merge metadata
        merged_metadata = sorted_elements[0].metadata.copy()
        merged_metadata.update({
            'merged_elements': len(elements),
            'original_texts': [e.text for e in elements],
            'merge_method': merged_metadata.get('merge_method', 'spatial_font_similarity')
        })
        
        # Determine element type
        element_types = [e.element_type for e in elements]
        merged_type = max(set(element_types), key=element_types.count)
        
        return LayoutElement(
            id=sorted_elements[0].id,
            element_type=merged_type,
            confidence=min(e.confidence for e in elements),
            bbox=merged_bbox,
            text=merged_text,
            style=merged_style,
            metadata=merged_metadata
        )

    def _extract_pdf_style_info(self, span: Dict[str, Any], block: Dict[str, Any], line: Dict[str, Any]) -> StyleInfo:
        """
        Extract comprehensive style information from a PDF span.
        
        Args:
            span: PyMuPDF span dictionary
            block: PyMuPDF block dictionary
            line: PyMuPDF line dictionary
            
        Returns:
            StyleInfo object with formatting details
        """
        # Extract font information
        font_info = FontInfo(
            name=span.get("font", "Unknown"),
            size=span.get("size", 0.0),
            bold=self._is_font_bold(span),
            italic=self._is_font_italic(span),
            underline=self._is_font_underlined(span),
            color=self._extract_font_color(span)
        )
        
        # Create paragraph format (approximate from line/block info)
        para_format = ParagraphFormat(
            alignment=self._determine_text_alignment(line, block),
            space_before=None,  # Not easily available in PDF spans
            space_after=None,   # Not easily available in PDF spans
        )
        
        # Create run information
        runs = [RunInfo(
            text=span["text"],
            font=font_info
        )]
        
        # Extract style name (approximate from font characteristics)
        style_name = self._generate_style_name(span, font_info)
        
        return StyleInfo(
            style_name=style_name,
            style_type='text_span',
            paragraph_format=para_format,
            runs=runs,
            primary_font=font_info
        )
    
    def _is_font_bold(self, span: Dict[str, Any]) -> bool:
        """Check if font is bold based on flags or font name."""
        flags = span.get("flags", 0)
        # Check bold flag (bit 4)
        if flags & 16:  # 2^4 = 16
            return True
        
        # Check font name for bold indicators
        font_name = span.get("font", "").lower()
        return any(indicator in font_name for indicator in ["bold", "black", "heavy"])
    
    def _is_font_italic(self, span: Dict[str, Any]) -> bool:
        """Check if font is italic based on flags or font name."""
        flags = span.get("flags", 0)
        # Check italic flag (bit 1)
        if flags & 2:  # 2^1 = 2
            return True
        
        # Check font name for italic indicators
        font_name = span.get("font", "").lower()
        return any(indicator in font_name for indicator in ["italic", "oblique"])
    
    def _is_font_underlined(self, span: Dict[str, Any]) -> bool:
        """Check if font is underlined based on flags or font name."""
        flags = span.get("flags", 0)
        # Check underline flag (bit 0)
        if flags & 1:  # Underline flag
            return True
        
        # Check font name for underline indicators
        font_name = span.get("font", "").lower()
        return "underline" in font_name
    
    def _has_font_flag(self, span: Dict[str, Any], flag_bit: int) -> bool:
        """Check if a specific font flag is set."""
        flags = span.get("flags", 0)
        return bool(flags & (1 << flag_bit))
    
    def _extract_font_color(self, span: Dict[str, Any]) -> Optional[str]:
        """Extract font color from span."""
        color = span.get("color")
        if color is not None:
            # Convert color integer to hex
            return f"#{color:06x}"
        return None
    
    def _determine_text_alignment(self, line: Dict[str, Any], block: Dict[str, Any]) -> TextAlignment:
        """Determine text alignment from line and block information."""
        # Get line and block bounding boxes
        line_bbox = line.get("bbox", [0, 0, 0, 0])
        block_bbox = block.get("bbox", [0, 0, 0, 0])
        
        if not line_bbox or not block_bbox:
            return TextAlignment.LEFT
        
        # Calculate relative positions
        line_left = line_bbox[0]
        line_right = line_bbox[2]
        line_width = line_right - line_left
        
        block_left = block_bbox[0]
        block_right = block_bbox[2]
        block_width = block_right - block_left
        
        # Calculate margins
        left_margin = line_left - block_left
        right_margin = block_right - line_right
        
        # Determine alignment based on margins and positioning
        margin_tolerance = 5.0  # pixels
        
        # Check for center alignment
        if abs(left_margin - right_margin) <= margin_tolerance:
            return TextAlignment.CENTER
        
        # Check for right alignment  
        if right_margin <= margin_tolerance and left_margin > margin_tolerance * 2:
            return TextAlignment.RIGHT
        
        # Check for justified text (line takes up most of the block width)
        if line_width / block_width > 0.9:
            return TextAlignment.JUSTIFY
        
        # Default to left alignment
        return TextAlignment.LEFT
    
    def _generate_style_name(self, span: Dict[str, Any], font_info: FontInfo) -> str:
        """Generate a descriptive style name based on font characteristics."""
        parts = []
        
        if font_info.name:
            parts.append(font_info.name)
        
        if font_info.size:
            parts.append(f"{font_info.size}pt")
        
        if font_info.bold:
            parts.append("Bold")
        
        if font_info.italic:
            parts.append("Italic")
        
        return "-".join(parts) if parts else "Unknown"
    
    def _find_nearby_underlines(self, span: Dict[str, Any], drawings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find underline drawings near a text span.
        
        Args:
            span: PyMuPDF span dictionary
            drawings: List of drawing objects from page
            
        Returns:
            List of nearby underline information
        """
        span_bbox = span.get("bbox", [0, 0, 0, 0])
        nearby_underlines = []
        
        # Define search area around the span
        search_margin = 10.0
        
        for i, drawing in enumerate(drawings):
            if drawing.get('type') == 's':  # Stroke drawing
                items = drawing.get('items', [])
                for item in items:
                    if len(item) >= 3 and item[0] == 'l':  # Line
                        start_x, start_y = item[1].x, item[1].y
                        end_x, end_y = item[2].x, item[2].y
                        
                        # Check if it's near our span (horizontally aligned underline)
                        if (span_bbox[1] - search_margin <= start_y <= span_bbox[3] + search_margin and
                            span_bbox[0] - search_margin <= start_x <= span_bbox[2] + 50):
                            
                            # Calculate approximate underscores
                            underline_length = end_x - start_x
                            num_underscores = max(int(underline_length / 8), 6)
                            
                            nearby_underlines.append({
                                'drawing_index': i,
                                'start': [start_x, start_y],
                                'end': [end_x, end_y],
                                'width': drawing.get('width', 0),
                                'length': underline_length,
                                'estimated_underscores': num_underscores,
                                'is_horizontal': abs(start_y - end_y) < 2.0
                            })
        
        return nearby_underlines

    def _extract_heading_level_from_pdf_style(self, span: Dict[str, Any], style_info: StyleInfo) -> int:
        """
        Extract heading level information from PDF span style (raw style info only).
        
        Args:
            span: PyMuPDF span dictionary
            style_info: Style information
            
        Returns:
            Heading level (0 for non-headings, 1-9 for headings) based on font characteristics
        """
        if not style_info.primary_font:
            return 0
        
        font = style_info.primary_font
        font_size = font.size or 0
        
        # Heuristic based on font size and style
        # These thresholds might need adjustment based on specific documents
        if font_size >= 18:
            return 1
        elif font_size >= 16:
            return 2
        elif font_size >= 14:
            return 3
        elif font.bold and font_size >= 12:
            return 4
        
        return 0  # Not a heading based on style
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported input formats.
        
        Returns:
            List of supported file extensions
        """
        return ['.pdf']
    
    def get_detector_info(self) -> Dict[str, Any]:
        """
        Get information about the extractor.
        
        Returns:
            Dictionary containing extractor information
        """
        features = [
            'spatial_proximity_merging',
            'font_similarity_analysis', 
            'reading_order_detection',
            'logical_sequence_validation',
            'hyphenation_handling'
        ]
        
        return {
            'name': 'PdfLayoutExtractorWithCVGuidance',
            'version': '3.0.0',
            'description': 'Document-native PDF extractor with CV-guided fragment merging using spatial, font, and visual analysis',
            'supported_formats': self.get_supported_formats(),
            'extraction_method': 'pdf-native-with-merging' if self.merge_fragments else 'pdf-native-raw',
            'requires_conversion': False,
            'fragment_merging': self.merge_fragments,
            'cv_guidance_enabled': False,
            'cv_confidence_threshold': 0.3,
            'features': features
        }

    def _sort_elements_by_reading_order(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """
        Sort elements by reading order: page number, then top-to-bottom, left-to-right.
        
        Args:
            elements: List of elements to sort
            
        Returns:
            Sorted list of elements
        """
        def sort_key(element):
            page_num = element.metadata.get('page_number', 0) if element.metadata else 0
            y_pos = element.bbox.y1 if element.bbox else 0
            x_pos = element.bbox.x1 if element.bbox else 0
            return (page_num, y_pos, x_pos)
        
        sorted_elements = sorted(elements, key=sort_key)
        logger.info(f"Sorted {len(elements)} elements by reading order")
        return sorted_elements


# unit test
# python -m models.layout_detection.layout_extraction.pdf_layout_extractor
if __name__ == "__main__":
    extractor = PdfLayoutExtractor(merge_fragments=True)
    result = extractor._detect_layout(input_data="tests/test_data/1-1 买卖合同（通用版）.pdf")
    import json 
    
    json.dump(result.model_dump(), open("pure_pdf_extraction_result.json", "w"), indent=2, ensure_ascii=False)
    from models.layout_detection.utils.visualiza_layout_elements import visualize_pdf_layout
    visualize_pdf_layout("tests/test_data/1-1 买卖合同（通用版）.pdf", result, "pure_pdf_extraction_result.pdf")
