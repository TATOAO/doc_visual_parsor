"""
Layout Processing Utilities

This module provides common utilities for processing layout elements, such as
sorting, filtering, and other layout-related operations.
"""

from typing import List, Dict, Tuple
from .schemas import LayoutElement, BoundingBox


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


def calculate_bbox_overlap(box1: BoundingBox, box2: BoundingBox) -> float:
    """
    Calculate the overlap ratio between two bounding boxes.
    
    Args:
        box1: First bounding box
        box2: Second bounding box
        
    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    # Calculate intersection area
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
        
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate area of each box
    area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
    area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
    
    # Return intersection as ratio of smaller area
    smaller_area = min(area1, area2)
    return intersection_area / smaller_area if smaller_area > 0 else 0.0


def is_bbox_contained(inner_bbox: BoundingBox, outer_bbox: BoundingBox) -> bool:
    """
    Check if inner_bbox is contained within outer_bbox.
    
    Args:
        inner_bbox: Inner bounding box
        outer_bbox: Outer bounding box
        
    Returns:
        True if inner_bbox is contained within outer_bbox
    """
    return (inner_bbox.x1 >= outer_bbox.x1 and 
            inner_bbox.y1 >= outer_bbox.y1 and
            inner_bbox.x2 <= outer_bbox.x2 and 
            inner_bbox.y2 <= outer_bbox.y2)


def merge_overlapping_elements(elements: List[LayoutElement], 
                              overlap_threshold: float = 0.5) -> List[LayoutElement]:
    """
    Merge elements that have significant overlap.
    
    Args:
        elements: List of layout elements
        overlap_threshold: Threshold for merging elements
        
    Returns:
        List of merged elements
    """
    if not elements:
        return elements
    
    # Sort elements by area (larger first)
    sorted_elements = sorted(
        elements,
        key=lambda e: e.bbox.area if e.bbox else 0,
        reverse=True
    )
    
    merged_elements = []
    used_indices = set()
    
    for i, element in enumerate(sorted_elements):
        if i in used_indices:
            continue
            
        # Find overlapping elements
        overlapping_indices = [i]
        for j, other_element in enumerate(sorted_elements):
            if j <= i or j in used_indices:
                continue
                
            if element.bbox and other_element.bbox:
                overlap = calculate_bbox_overlap(element.bbox, other_element.bbox)
                if overlap > overlap_threshold:
                    overlapping_indices.append(j)
        
        # Merge overlapping elements
        if len(overlapping_indices) > 1:
            # Create merged element
            overlapping_elements = [sorted_elements[idx] for idx in overlapping_indices]
            merged_element = _create_merged_element(overlapping_elements)
            merged_elements.append(merged_element)
            
            # Mark indices as used
            used_indices.update(overlapping_indices)
        else:
            # No overlap, keep original element
            merged_elements.append(element)
            used_indices.add(i)
    
    return merged_elements


def _create_merged_element(elements: List[LayoutElement]) -> LayoutElement:
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
        text = elem.text.strip() if elem.text else ""
        if text:
            # Handle hyphenation
            if i > 0 and sorted_elements[i-1].text and sorted_elements[i-1].text.strip().endswith('-'):
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
    min_x1 = min(e.bbox.x1 for e in elements if e.bbox)
    min_y1 = min(e.bbox.y1 for e in elements if e.bbox)
    max_x2 = max(e.bbox.x2 for e in elements if e.bbox)
    max_y2 = max(e.bbox.y2 for e in elements if e.bbox)
    
    merged_bbox = BoundingBox(x1=min_x1, y1=min_y1, x2=max_x2, y2=max_y2)
    
    # Use style from the first element
    merged_style = sorted_elements[0].style
    
    # Merge metadata
    merged_metadata = sorted_elements[0].metadata.copy() if sorted_elements[0].metadata else {}
    merged_metadata.update({
        'merged_elements': len(elements),
        'merged_from_ids': [e.id for e in elements],
        'merge_method': 'overlap_based'
    })
    
    # Use the element type from the element with highest confidence
    best_element = max(elements, key=lambda e: e.confidence or 0)
    
    return LayoutElement(
        id=sorted_elements[0].id,  # Keep first element's ID
        element_type=best_element.element_type,
        text=merged_text,
        bbox=merged_bbox,
        confidence=max(e.confidence or 0 for e in elements),
        style=merged_style,
        metadata=merged_metadata
    )
