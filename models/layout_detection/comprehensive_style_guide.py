#!/usr/bin/env python3
"""
Comprehensive PyMuPDF Style Extraction Guide

This demonstrates how to extract style information from PDF text blocks
that contain underlined blank spaces, like "6.3 检验验收标准：______。"
"""

import fitz
from typing import Dict, List, Any

def extract_font_styling(flags: int) -> Dict[str, bool]:
    """Extract font styling from PyMuPDF flags"""
    return {
        'bold': bool(flags & 16),
        'italic': bool(flags & 64),
        'monospaced': bool(flags & 8),
        'serifed': bool(flags & 32)
    }

def comprehensive_style_extraction(pdf_path: str, page_num: int):
    """Comprehensive style extraction example"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Get text and drawings
    text_dict = page.get_text("dict")
    blocks = text_dict["blocks"]
    drawings = page.get_drawings()
    
    print("=== COMPREHENSIVE STYLE EXTRACTION ===")
    
    # Example 1: Your specific case - Block 12
    print("\n1. ANALYZING YOUR SPECIFIC EXAMPLE (Block 12):")
    block_12 = blocks[12]
    
    print("Text Analysis:")
    all_spans = []
    for line in block_12['lines']:
        for span in line['spans']:
            styling = extract_font_styling(span['flags'])
            span_info = {
                'text': span['text'],
                'font': span['font'],
                'size': span['size'],
                'styling': styling,
                'bbox': span['bbox'],
                'x_start': span['bbox'][0],
                'x_end': span['bbox'][2]
            }
            all_spans.append(span_info)
            
            print(f"  '{span['text']}':")
            print(f"    Font: {span['font']}, Size: {span['size']}pt")
            print(f"    Styling: {styling}")
            print(f"    Position: {span['bbox']}")
    
    # Find underlines near this block
    print("\nUnderline Analysis:")
    block_bbox = block_12['bbox']
    
    for i, drawing in enumerate(drawings):
        if drawing.get('type') == 's':  # Stroke
            for item in drawing.get('items', []):
                if len(item) >= 3 and item[0] == 'l':  # Line
                    start_x, start_y = item[1].x, item[1].y
                    end_x, end_y = item[2].x, item[2].y
                    
                    # Check if near our block
                    if (block_bbox[1] - 10 <= start_y <= block_bbox[3] + 10 and
                        block_bbox[0] - 10 <= start_x <= block_bbox[2] + 50):
                        print(f"  Underline: ({start_x:.1f}, {start_y:.1f}) -> ({end_x:.1f}, {end_y:.1f})")
                        print(f"    Length: {abs(end_x - start_x):.1f} points")
                        print(f"    Width: {drawing.get('width', 0)} points")
    
    # Reconstruct with blanks
    print("\nReconstructed Text:")
    all_spans.sort(key=lambda x: x['x_start'])
    
    reconstructed = []
    for i, span in enumerate(all_spans):
        reconstructed.append(span['text'])
        
        # Check for underlines after this span
        span_end = span['x_end']
        next_start = all_spans[i+1]['x_start'] if i+1 < len(all_spans) else float('inf')
        
        # Look for underlines in the gap
        for drawing in drawings:
            if drawing.get('type') == 's':
                for item in drawing.get('items', []):
                    if len(item) >= 3 and item[0] == 'l':
                        start_x = item[1].x
                        end_x = item[2].x
                        
                        if span_end <= start_x < next_start:
                            underline_length = abs(end_x - start_x)
                            num_underscores = max(int(underline_length / 8), 6)
                            reconstructed.append('_' * num_underscores)
                            break
    
    final_text = ''.join(reconstructed)
    print(f"Final: '{final_text}'")
    
    # Example 2: Advanced style analysis
    print("\n2. ADVANCED STYLE ANALYSIS:")
    
    # Analyze all text elements
    style_stats = {}
    for block in blocks:
        if 'lines' not in block:
            continue
        for line in block['lines']:
            for span in line['spans']:
                key = (span['font'], span['size'], span['flags'])
                if key not in style_stats:
                    style_stats[key] = {'count': 0, 'examples': []}
                style_stats[key]['count'] += 1
                if len(style_stats[key]['examples']) < 3:
                    style_stats[key]['examples'].append(span['text'][:20])
    
    print("Font Style Distribution:")
    for (font, size, flags), stats in sorted(style_stats.items(), key=lambda x: x[1]['count'], reverse=True):
        styling = extract_font_styling(flags)
        print(f"  {font} {size}pt (flags: {flags}): {stats['count']} occurrences")
        print(f"    Styling: {styling}")
        print(f"    Examples: {stats['examples']}")
    
    # Example 3: Finding patterns
    print("\n3. PATTERN DETECTION:")
    
    # Find all text with underlines
    underlined_patterns = []
    
    for block_num, block in enumerate(blocks):
        if 'lines' not in block:
            continue
            
        for line in block['lines']:
            line_spans = [span for span in line['spans']]
            if len(line_spans) >= 2:  # Multiple spans might indicate gaps
                
                # Check for drawings between spans
                for drawing in drawings:
                    if drawing.get('type') == 's':
                        for item in drawing.get('items', []):
                            if len(item) >= 3 and item[0] == 'l':
                                start_x, start_y = item[1].x, item[1].y
                                
                                # Check if this line is within the text line
                                line_bbox = line['bbox']
                                if (line_bbox[1] - 5 <= start_y <= line_bbox[3] + 5):
                                    pattern = {
                                        'block': block_num,
                                        'spans': [s['text'] for s in line_spans],
                                        'underline_pos': (start_x, start_y),
                                        'style': extract_font_styling(line_spans[0]['flags'])
                                    }
                                    underlined_patterns.append(pattern)
                                    break
    
    print(f"Found {len(underlined_patterns)} underlined text patterns:")
    for pattern in underlined_patterns[:5]:  # Show first 5
        print(f"  Block {pattern['block']}: {pattern['spans']}")
        print(f"    Style: {pattern['style']}")
    
    doc.close()

if __name__ == "__main__":
    pdf_path = "../../tests/test_data/1-1 买卖合同（通用版）.pdf"
    comprehensive_style_extraction(pdf_path, 17) 