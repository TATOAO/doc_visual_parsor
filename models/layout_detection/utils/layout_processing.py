"""
Layout Processing Utilities

This module provides common utilities for processing layout elements, such as
sorting, filtering, and other layout-related operations.
"""

from typing import List, Dict, Tuple
from models.schemas.layout_schemas import LayoutElement, BoundingBox

def sort_elements_by_position(elements: List[LayoutElement]) -> List[LayoutElement]:
    """
    Sort elements by natural reading order (top-to-bottom, left-to-right).
    Uses a sophisticated algorithm that:
    1. First sorts by page number
    2. Groups elements into lines based on vertical overlap within each page
    3. Sorts elements within each line from left to right
    4. Handles elements that span multiple lines
    """
    if not elements:
        return elements

    # Group elements by page number
    page_groups = {}
    for elem in elements:
        if not elem.bbox:
            continue
        page_num = elem.metadata.get('page_number', 0) if elem.metadata else 0
        if page_num not in page_groups:
            page_groups[page_num] = []
        page_groups[page_num].append(elem)

    # Sort pages and process each page
    sorted_elements = []
    for page_num in sorted(page_groups.keys()):
        page_elements = page_groups[page_num]
        
        # First, sort all elements by their vertical position (top to bottom)
        page_elements = sorted(page_elements, key=lambda e: e.bbox.y1 if e.bbox else 0)

        # Group elements into lines based on vertical overlap
        lines = []
        current_line = []
        current_line_y = None
        line_height_threshold = 0.5  # Threshold for considering elements in the same line

        for elem in page_elements:
            if not elem.bbox:
                continue

            # If this is the first element or it's significantly below the current line
            if current_line_y is None or elem.bbox.y1 > current_line_y + line_height_threshold:
                if current_line:
                    lines.append(current_line)
                current_line = [elem]
                current_line_y = elem.bbox.y1
            else:
                current_line.append(elem)
                # Update line height to account for elements that might be taller
                current_line_y = min(current_line_y, elem.bbox.y1)

        # Add the last line if it exists
        if current_line:
            lines.append(current_line)

        # Sort elements within each line from left to right
        for i in range(len(lines)):
            lines[i] = sorted(lines[i], key=lambda e: e.bbox.x1 if e.bbox else 0)

        # Add the sorted elements from this page to the final result
        sorted_elements.extend([elem for line in lines for elem in line])

    return sorted_elements

def filter_redundant_boxes(elements: List[LayoutElement], overlap_threshold: float = 0.9) -> List[LayoutElement]:
    """
    Filter out redundant boxes that have significant overlap.
    
    Args:
        elements: List of layout elements
        overlap_threshold: Threshold for considering boxes redundant (0.0 to 1.0)
        
    Returns:
        Filtered list of elements with redundant boxes removed
    """
    def calculate_area_overlap(box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate area overlap ratio between two boxes."""
        # Calculate intersection area
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate area of smaller box
        box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        smaller_area = min(box1_area, box2_area)
        
        return intersection_area / smaller_area if smaller_area > 0 else 0.0
    
    # Sort elements by area (larger to smaller)
    sorted_elements = sorted(
        elements,
        key=lambda e: (e.bbox.x2 - e.bbox.x1) * (e.bbox.y2 - e.bbox.y1),
        reverse=True
    )
    
    filtered_elements = []
    for element in sorted_elements:
        # Check if this element overlaps significantly with any already accepted element
        is_redundant = False
        for accepted in filtered_elements:
            if calculate_area_overlap(element.bbox, accepted.bbox) > overlap_threshold:
                is_redundant = True
                break
        
        if not is_redundant:
            filtered_elements.append(element)
    
    return filtered_elements 