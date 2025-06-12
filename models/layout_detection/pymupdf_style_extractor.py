#!/usr/bin/env python3
"""
PyMuPDF Style Extractor for Text with Underlines/Blank Spaces

This script demonstrates how to extract style information from PDF documents
where blank spaces are represented by underline drawings rather than underscore characters.

Example case: "6.3 检验验收标准：______ 。"
- Text appears as separate spans: "6.3 检验验收标准：" and "。"
- The blank space is represented by an underline drawing between them
"""

import fitz
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

@dataclass
class StyleInfo:
    """Style information extracted from a span"""
    font: str
    size: float
    flags: int  # Font flags (bold=16, italic=64, etc.)
    color: int
    bbox: Tuple[float, float, float, float]
    text: str
    
    @property
    def is_bold(self) -> bool:
        """Check if text is bold (flag 16)"""
        return bool(self.flags & 16)
    
    @property
    def is_italic(self) -> bool:
        """Check if text is italic (flag 64)"""
        return bool(self.flags & 64)

@dataclass
class UnderlineInfo:
    """Information about underline drawings"""
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    width: float
    color: Tuple[float, float, float]
    bbox: Tuple[float, float, float, float]

class PDFStyleExtractor:
    """Extract style information from PDF documents including underlined blank spaces"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.doc.close()
    
    def extract_text_blocks_with_style(self, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract text blocks with detailed style information
        
        Args:
            page_num: Page number (0-indexed)
            
        Returns:
            List of text blocks with style information
        """
        page = self.doc[page_num]
        text_dict = page.get_text("dict")
        
        blocks_with_style = []
        
        for block_num, block in enumerate(text_dict["blocks"]):
            if "lines" not in block:
                continue
                
            block_info = {
                'block_number': block_num,
                'bbox': block['bbox'],
                'lines': []
            }
            
            for line_num, line in enumerate(block["lines"]):
                line_info = {
                    'line_number': line_num,
                    'bbox': line['bbox'],
                    'spans': []
                }
                
                for span_num, span in enumerate(line["spans"]):
                    style_info = StyleInfo(
                        font=span['font'],
                        size=span['size'],
                        flags=span['flags'],
                        color=span['color'],
                        bbox=span['bbox'],
                        text=span['text']
                    )
                    
                    span_info = {
                        'span_number': span_num,
                        'text': span['text'],
                        'style': style_info,
                        'raw_span': span
                    }
                    
                    line_info['spans'].append(span_info)
                
                block_info['lines'].append(line_info)
            
            blocks_with_style.append(block_info)
        
        return blocks_with_style
    
    def extract_underlines(self, page_num: int) -> List[UnderlineInfo]:
        """
        Extract underline drawings from the page
        
        Args:
            page_num: Page number (0-indexed)
            
        Returns:
            List of underline information
        """
        page = self.doc[page_num]
        drawings = page.get_drawings()
        
        underlines = []
        
        for drawing in drawings:
            if drawing.get('type') == 's':  # Stroke/line drawing
                items = drawing.get('items', [])
                
                for item in items:
                    if len(item) >= 3 and item[0] == 'l':  # Line item
                        start_point = (item[1].x, item[1].y)
                        end_point = (item[2].x, item[2].y)
                        
                        # Check if it's a horizontal line (potential underline)
                        if abs(start_point[1] - end_point[1]) < 1.0:  # Horizontal tolerance
                            underline_info = UnderlineInfo(
                                start_point=start_point,
                                end_point=end_point,
                                width=drawing.get('width', 0),
                                color=drawing.get('color', (0, 0, 0)),
                                bbox=(drawing['rect'].x0, drawing['rect'].y0, 
                                     drawing['rect'].x1, drawing['rect'].y1)
                            )
                            underlines.append(underline_info)
        
        return underlines
    
    def find_text_with_blanks(self, page_num: int, 
                             proximity_threshold: float = 5.0) -> List[Dict[str, Any]]:
        """
        Find text blocks that have associated underline drawings (blank spaces)
        
        Args:
            page_num: Page number (0-indexed)
            proximity_threshold: Maximum distance to consider underline as related to text
            
        Returns:
            List of text blocks with their associated underlines
        """
        text_blocks = self.extract_text_blocks_with_style(page_num)
        underlines = self.extract_underlines(page_num)
        
        results = []
        
        for block in text_blocks:
            for line in block['lines']:
                # Find potential blanks between spans or after spans
                line_underlines = []
                
                for underline in underlines:
                    underline_y = underline.start_point[1]
                    line_bbox = line['bbox']
                    
                    # Check if underline is vertically close to the text line
                    if (line_bbox[1] - proximity_threshold <= underline_y <= 
                        line_bbox[3] + proximity_threshold):
                        
                        # Check if underline is horizontally within or near the line
                        underline_x_start = min(underline.start_point[0], underline.end_point[0])
                        underline_x_end = max(underline.start_point[0], underline.end_point[0])
                        
                        line_x_start = line_bbox[0]
                        line_x_end = line_bbox[2]
                        
                        # Check for overlap or proximity
                        if (underline_x_start <= line_x_end + proximity_threshold and 
                            underline_x_end >= line_x_start - proximity_threshold):
                            line_underlines.append(underline)
                
                if line_underlines:
                    result = {
                        'block_number': block['block_number'],
                        'line': line,
                        'underlines': line_underlines,
                        'reconstructed_text': self._reconstruct_text_with_blanks(line, line_underlines)
                    }
                    results.append(result)
        
        return results
    
    def _reconstruct_text_with_blanks(self, line: Dict[str, Any], 
                                    underlines: List[UnderlineInfo]) -> str:
        """
        Reconstruct text by inserting blank markers where underlines appear
        
        Args:
            line: Line information with spans
            underlines: Associated underlines
            
        Returns:
            Reconstructed text with blank markers
        """
        spans = line['spans']
        if not spans:
            return ""
        
        # Sort spans by x-coordinate
        sorted_spans = sorted(spans, key=lambda x: x['style'].bbox[0])
        
        # Sort underlines by x-coordinate
        sorted_underlines = sorted(underlines, key=lambda x: x.start_point[0])
        
        result_parts = []
        current_x = 0
        
        span_idx = 0
        underline_idx = 0
        
        while span_idx < len(sorted_spans) or underline_idx < len(sorted_underlines):
            next_span_x = sorted_spans[span_idx]['style'].bbox[0] if span_idx < len(sorted_spans) else float('inf')
            next_underline_x = sorted_underlines[underline_idx].start_point[0] if underline_idx < len(sorted_underlines) else float('inf')
            
            if next_span_x <= next_underline_x:
                # Add span text
                result_parts.append(sorted_spans[span_idx]['text'])
                current_x = sorted_spans[span_idx]['style'].bbox[2]
                span_idx += 1
            else:
                # Add blank marker for underline
                underline = sorted_underlines[underline_idx]
                blank_length = int((underline.end_point[0] - underline.start_point[0]) / 8)  # Approximate character width
                result_parts.append('_' * max(blank_length, 6))  # At least 6 underscores
                current_x = underline.end_point[0]
                underline_idx += 1
        
        return ''.join(result_parts)

def main():
    """Demonstrate usage with the test document"""
    pdf_path = "../../tests/test_data/1-1 买卖合同（通用版）.pdf"
    
    with PDFStyleExtractor(pdf_path) as extractor:
        # Extract text blocks with blanks from page 17
        page_num = 17
        text_with_blanks = extractor.find_text_with_blanks(page_num)
        
        print("=== TEXT BLOCKS WITH BLANK SPACES ===")
        for item in text_with_blanks:
            print(f"Block {item['block_number']}:")
            print(f"  Original spans: {[span['text'] for span in item['line']['spans']]}")
            print(f"  Reconstructed: {item['reconstructed_text']}")
            print(f"  Style info:")
            
            for span in item['line']['spans']:
                style = span['style']
                print(f"    Text: '{span['text']}'")
                print(f"    Font: {style.font}, Size: {style.size}")
                print(f"    Bold: {style.is_bold}, Italic: {style.is_italic}")
                print(f"    Color: {style.color}, BBox: {style.bbox}")
            
            print(f"  Underlines: {len(item['underlines'])}")
            for i, underline in enumerate(item['underlines']):
                print(f"    Underline {i}: {underline.start_point} -> {underline.end_point}")
            print("---")

if __name__ == "__main__":
    main() 