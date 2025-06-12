#!/usr/bin/env python3
"""
Comprehensive PyMuPDF Style Extraction Guide

This script demonstrates advanced techniques for extracting style information
from PDF documents, including handling of underlined blank spaces, font styling,
and positional information.
"""

import fitz
import json
from typing import Dict, List, Tuple, Any

class ComprehensiveStyleExtractor:
    """Advanced style extraction using PyMuPDF"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
    
    def close(self):
        """Close the document"""
        self.doc.close()
    
    def extract_font_flags(self, flags: int) -> Dict[str, bool]:
        """
        Extract font styling information from PyMuPDF flags
        
        PyMuPDF font flags:
        - 2^4 (16): Bold
        - 2^6 (64): Italic  
        - 2^3 (8): Monospaced
        - 2^5 (32): Serifed
        """
        return {
            'bold': bool(flags & 16),
            'italic': bool(flags & 64), 
            'monospaced': bool(flags & 8),
            'serifed': bool(flags & 32)
        }
    
    def get_text_styling_info(self, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract comprehensive text styling information from a page
        
        Args:
            page_num: Page number (0-indexed)
            
        Returns:
            List of text elements with complete style information
        """
        page = self.doc[page_num]
        text_dict = page.get_text("dict")
        
        styled_elements = []
        
        for block_num, block in enumerate(text_dict["blocks"]):
            if "lines" not in block:
                continue
                
            for line_num, line in enumerate(block["lines"]):
                for span_num, span in enumerate(line["spans"]):
                    font_flags = self.extract_font_flags(span['flags'])
                    
                    element = {
                        'block_id': block_num,
                        'line_id': line_num,
                        'span_id': span_num,
                        'text': span['text'],
                        'font': {
                            'name': span['font'],
                            'size': span['size'],
                            'color': span['color'],
                            'flags': span['flags'],
                            'styling': font_flags
                        },
                        'position': {
                            'bbox': span['bbox'],
                            'origin': span['origin'],
                            'baseline_y': span['origin'][1],
                            'ascender': span.get('ascender', 0),
                            'descender': span.get('descender', 0)
                        },
                        'text_direction': {
                            'wmode': line.get('wmode', 0),  # 0=horizontal, 1=vertical
                            'dir': line.get('dir', [1.0, 0.0])  # Direction vector
                        }
                    }
                    
                    styled_elements.append(element)
        
        return styled_elements
    
    def extract_drawings_info(self, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract drawing elements (lines, shapes, etc.) from a page
        
        Args:
            page_num: Page number (0-indexed)
            
        Returns:
            List of drawing elements with their properties
        """
        page = self.doc[page_num]
        drawings = page.get_drawings()
        
        drawing_elements = []
        
        for i, drawing in enumerate(drawings):
            element = {
                'drawing_id': i,
                'type': drawing.get('type', 'unknown'),
                'stroke_properties': {
                    'width': drawing.get('width', 0),
                    'color': drawing.get('color', (0, 0, 0)),
                    'opacity': drawing.get('stroke_opacity', 1.0),
                    'line_cap': drawing.get('lineCap', (0, 0, 0)),
                    'line_join': drawing.get('lineJoin', 0),
                    'dashes': drawing.get('dashes', '[] 0')
                },
                'fill_properties': {
                    'fill': drawing.get('fill', None),
                    'opacity': drawing.get('fill_opacity', None),
                    'even_odd': drawing.get('even_odd', None)
                },
                'geometry': {
                    'bbox': (drawing['rect'].x0, drawing['rect'].y0, 
                            drawing['rect'].x1, drawing['rect'].y1),
                    'items': []
                }
            }
            
            # Extract geometric items
            for item in drawing.get('items', []):
                if len(item) >= 3:
                    item_type = item[0]
                    if item_type == 'l':  # Line
                        element['geometry']['items'].append({
                            'type': 'line',
                            'start': (item[1].x, item[1].y),
                            'end': (item[2].x, item[2].y)
                        })
                    elif item_type == 'c':  # Curve
                        element['geometry']['items'].append({
                            'type': 'curve',
                            'points': [(p.x, p.y) for p in item[1:]]
                        })
            
            drawing_elements.append(element)
        
        return drawing_elements
    
    def find_underlined_text(self, page_num: int, 
                           vertical_tolerance: float = 3.0,
                           horizontal_tolerance: float = 10.0) -> List[Dict[str, Any]]:
        """
        Find text elements that have associated underline drawings
        
        Args:
            page_num: Page number (0-indexed)
            vertical_tolerance: Vertical distance tolerance for matching
            horizontal_tolerance: Horizontal distance tolerance for matching
            
        Returns:
            List of text elements with their associated underlines
        """
        text_elements = self.get_text_styling_info(page_num)
        drawings = self.extract_drawings_info(page_num)
        
        # Filter for horizontal lines (potential underlines)
        potential_underlines = []
        for drawing in drawings:
            if drawing['type'] == 's':  # Stroke
                for item in drawing['geometry']['items']:
                    if (item['type'] == 'line' and 
                        abs(item['start'][1] - item['end'][1]) < 1.0):  # Horizontal line
                        potential_underlines.append({
                            'drawing': drawing,
                            'line': item,
                            'y_position': item['start'][1],
                            'x_start': min(item['start'][0], item['end'][0]),
                            'x_end': max(item['start'][0], item['end'][0])
                        })
        
        # Match text with underlines
        underlined_text = []
        
        for text_elem in text_elements:
            text_bbox = text_elem['position']['bbox']
            text_bottom = text_bbox[3]  # Bottom edge of text
            text_left = text_bbox[0]
            text_right = text_bbox[2]
            
            # Find matching underlines
            matching_underlines = []
            for underline in potential_underlines:
                # Check vertical proximity
                if abs(underline['y_position'] - text_bottom) <= vertical_tolerance:
                    # Check horizontal overlap or proximity
                    if not (underline['x_end'] < text_left - horizontal_tolerance or 
                           underline['x_start'] > text_right + horizontal_tolerance):
                        matching_underlines.append(underline)
            
            if matching_underlines:
                underlined_text.append({
                    'text_element': text_elem,
                    'underlines': matching_underlines
                })
        
        return underlined_text
    
    def reconstruct_text_with_blanks(self, page_num: int, 
                                   block_num: int = None) -> List[Dict[str, Any]]:
        """
        Reconstruct text from a page or specific block, inserting blank markers 
        where underlines appear between text spans
        
        Args:
            page_num: Page number (0-indexed)
            block_num: Optional specific block number to process
            
        Returns:
            List of reconstructed text blocks with style information
        """
        page = self.doc[page_num]
        text_dict = page.get_text("dict")
        drawings = self.extract_drawings_info(page_num)
        
        # Filter for horizontal lines
        horizontal_lines = []
        for drawing in drawings:
            if drawing['type'] == 's':
                for item in drawing['geometry']['items']:
                    if (item['type'] == 'line' and 
                        abs(item['start'][1] - item['end'][1]) < 1.0):
                        horizontal_lines.append({
                            'y': item['start'][1],
                            'x_start': min(item['start'][0], item['end'][0]),
                            'x_end': max(item['start'][0], item['end'][0]),
                            'length': abs(item['end'][0] - item['start'][0])
                        })
        
        reconstructed_blocks = []
        
        blocks_to_process = [text_dict["blocks"][block_num]] if block_num is not None else text_dict["blocks"]
        
        for block_idx, block in enumerate(blocks_to_process):
            if "lines" not in block:
                continue
                
            actual_block_num = block_num if block_num is not None else block_idx
            
            for line in block["lines"]:
                # Collect all spans in this line
                spans = []
                for span in line["spans"]:
                    spans.append({
                        'text': span['text'],
                        'x_start': span['bbox'][0],
                        'x_end': span['bbox'][2],
                        'y_center': (span['bbox'][1] + span['bbox'][3]) / 2,
                        'style': {
                            'font': span['font'],
                            'size': span['size'],
                            'flags': span['flags'],
                            'styling': self.extract_font_flags(span['flags'])
                        }
                    })
                
                # Sort spans by x position
                spans.sort(key=lambda x: x['x_start'])
                
                # Find relevant underlines for this line
                line_y = (line['bbox'][1] + line['bbox'][3]) / 2
                relevant_underlines = []
                
                for underline in horizontal_lines:
                    if abs(underline['y'] - line_y) <= 10:  # Within 10 points vertically
                        relevant_underlines.append(underline)
                
                # Reconstruct text with blanks
                reconstructed_parts = []
                
                for i, span in enumerate(spans):
                    reconstructed_parts.append({
                        'type': 'text',
                        'content': span['text'],
                        'style': span['style']
                    })
                    
                    # Check for underlines between this span and the next
                    span_end = span['x_end']
                    next_span_start = spans[i+1]['x_start'] if i+1 < len(spans) else float('inf')
                    
                    for underline in relevant_underlines:
                        if (span_end <= underline['x_start'] <= next_span_start or
                            span_end <= underline['x_end'] <= next_span_start):
                            
                            # Estimate number of characters for the blank
                            avg_char_width = 8  # Approximate character width in points
                            num_chars = max(int(underline['length'] / avg_char_width), 3)
                            
                            reconstructed_parts.append({
                                'type': 'blank',
                                'content': '_' * num_chars,
                                'style': span['style'],  # Use same style as preceding text
                                'underline_info': underline
                            })
                            break
                
                reconstructed_blocks.append({
                    'block_number': actual_block_num,
                    'original_spans': [s['text'] for s in spans],
                    'reconstructed_parts': reconstructed_parts,
                    'full_text': ''.join(part['content'] for part in reconstructed_parts)
                })
        
        return reconstructed_blocks

def main():
    """Demonstrate comprehensive style extraction"""
    pdf_path = "../../tests/test_data/1-1 买卖合同（通用版）.pdf"
    
    extractor = ComprehensiveStyleExtractor(pdf_path)
    
    try:
        page_num = 17
        print("=== COMPREHENSIVE STYLE EXTRACTION DEMO ===\n")
        
        # 1. Extract specific block (your example)
        print("1. RECONSTRUCTING BLOCK 12 (Your Example):")
        reconstructed = extractor.reconstruct_text_with_blanks(page_num, block_num=12)
        
        for block in reconstructed:
            print(f"Block {block['block_number']}:")
            print(f"  Original spans: {block['original_spans']}")
            print(f"  Reconstructed: '{block['full_text']}'")
            print(f"  Parts breakdown:")
            for part in block['reconstructed_parts']:
                print(f"    {part['type']}: '{part['content']}' "
                      f"[{part['style']['font']}, {part['style']['size']}pt, "
                      f"Bold: {part['style']['styling']['bold']}]")
        
        print("\n2. FINDING ALL UNDERLINED TEXT ON PAGE:")
        underlined = extractor.find_underlined_text(page_num)
        
        for item in underlined[:3]:  # Show first 3 examples
            text_elem = item['text_element']
            print(f"Text: '{text_elem['text']}'")
            print(f"  Font: {text_elem['font']['name']}, Size: {text_elem['font']['size']}")
            print(f"  Style: Bold={text_elem['font']['styling']['bold']}, "
                  f"Italic={text_elem['font']['styling']['italic']}")
            print(f"  Underlines: {len(item['underlines'])}")
            print("---")
        
        print("\n3. FONT FLAG ANALYSIS:")
        text_elements = extractor.get_text_styling_info(page_num)
        
        # Group by font flags
        flag_groups = {}
        for elem in text_elements:
            flags = elem['font']['flags']
            if flags not in flag_groups:
                flag_groups[flags] = []
            flag_groups[flags].append(elem)
        
        print("Font flag distribution:")
        for flags, elements in sorted(flag_groups.items()):
            if elements:  # Only show non-empty groups
                styling = extractor.extract_font_flags(flags)
                print(f"  Flags {flags}: {len(elements)} elements")
                print(f"    Style: {styling}")
                print(f"    Example: '{elements[0]['text'][:30]}...'")
    
    finally:
        extractor.close()

if __name__ == "__main__":
    main() 