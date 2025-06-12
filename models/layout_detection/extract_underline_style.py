#!/usr/bin/env python3
"""Extract style information from text with underlined blank spaces using PyMuPDF"""

import fitz
import json

def extract_style_info_with_underlines(pdf_path, page_num):
    """Extract style information from text blocks and correlate with underline drawings"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Get text blocks
    text_dict = page.get_text("dict")
    blocks = text_dict["blocks"]
    
    # Get drawings (underlines)
    drawings = page.get_drawings()
    
    print("=== ANALYZING BLOCK 12 (Your specific example) ===")
    block_12 = blocks[12]
    
    print("Block 12 text spans:")
    for line_num, line in enumerate(block_12['lines']):
        print(f"Line {line_num}:")
        for span_num, span in enumerate(line['spans']):
            print(f"  Span {span_num}: '{span['text']}'")
            print(f"    Font: {span['font']}, Size: {span['size']}")
            print(f"    Bold: {bool(span['flags'] & 16)}, Italic: {bool(span['flags'] & 64)}")
            print(f"    BBox: {span['bbox']}")
    
    print("\n=== NEARBY UNDERLINES ===")
    block_bbox = block_12['bbox']
    nearby_underlines = []
    
    for i, drawing in enumerate(drawings):
        if drawing.get('type') == 's':  # Stroke drawing
            items = drawing.get('items', [])
            for item in items:
                if len(item) >= 3 and item[0] == 'l':  # Line
                    start_x, start_y = item[1].x, item[1].y
                    end_x, end_y = item[2].x, item[2].y
                    
                    # Check if it's near our block vertically
                    if (block_bbox[1] - 10 <= start_y <= block_bbox[3] + 10 and
                        block_bbox[0] - 10 <= start_x <= block_bbox[2] + 50):
                        nearby_underlines.append({
                            'drawing_index': i,
                            'start': (start_x, start_y),
                            'end': (end_x, end_y),
                            'width': drawing.get('width', 0)
                        })
                        print(f"  Underline {i}: ({start_x:.1f}, {start_y:.1f}) -> ({end_x:.1f}, {end_y:.1f})")
    
    print("\n=== RECONSTRUCTED TEXT ===")
    # Reconstruct the text with underlines
    spans = []
    for line in block_12['lines']:
        for span in line['spans']:
            spans.append({
                'text': span['text'],
                'x_start': span['bbox'][0],
                'x_end': span['bbox'][2]
            })
    
    # Sort by x position
    spans.sort(key=lambda x: x['x_start'])
    
    # Insert underlines in proper positions
    reconstructed_parts = []
    for i, span in enumerate(spans):
        reconstructed_parts.append(span['text'])
        
        # Check if there's an underline after this span
        span_end_x = span['x_end']
        next_span_start_x = spans[i+1]['x_start'] if i+1 < len(spans) else float('inf')
        
        for underline in nearby_underlines:
            underline_start_x = underline['start'][0]
            underline_end_x = underline['end'][0]
            
            if (span_end_x <= underline_start_x < next_span_start_x):
                # Calculate approximate number of underscores
                underline_length = underline_end_x - underline_start_x
                num_underscores = max(int(underline_length / 8), 6)
                reconstructed_parts.append('_' * num_underscores)
                break
    
    reconstructed_text = ''.join(reconstructed_parts)
    print(f"Reconstructed: '{reconstructed_text}'")
    
    doc.close()
    return reconstructed_text

if __name__ == "__main__":
    pdf_path = "../../tests/test_data/1-1 买卖合同（通用版）.pdf"
    result = extract_style_info_with_underlines(pdf_path, 17) 