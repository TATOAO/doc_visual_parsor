"""
PDF Layout Element Visualization Utility

This module provides visualization tools for PDF layout elements, allowing users to
see bounding boxes, element types, and merging results overlaid on the original PDF.
"""

import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
from doc_chunking.schemas.layout_schemas import LayoutExtractionResult, LayoutElement, BoundingBox

logger = logging.getLogger(__name__)


class PdfLayoutVisualizer:
    """
    Visualizer for PDF layout elements with bounding box highlighting.
    """
    
    # Color scheme for different element types and statuses
    COLORS = {
        'text': (0, 0, 1),           # Blue
        'title': (1, 0, 0),          # Red
        'heading': (1, 0.5, 0),      # Orange
        'table': (0, 1, 0),          # Green
        'figure': (1, 0, 1),         # Magenta
        'merged': (0.8, 0.2, 0.8),   # Purple for merged elements
        'original': (0.3, 0.3, 0.3), # Gray for original fragments
        'default': (0, 0, 0)         # Black default
    }
    
    def __init__(self, 
                 line_width: float = 1.5,
                 font_size: int = 8,
                 show_labels: bool = True,
                 show_element_ids: bool = True):
        """
        Initialize the visualizer.
        
        Args:
            line_width: Width of bounding box lines
            font_size: Font size for labels
            show_labels: Whether to show element type labels
            show_element_ids: Whether to show element IDs
        """
        self.line_width = line_width
        self.font_size = font_size
        self.show_labels = show_labels
        self.show_element_ids = show_element_ids
    
    def visualize_layout_elements(self,
                                  pdf_path: Union[str, Path],
                                  layout_result: LayoutExtractionResult,
                                  output_path: Union[str, Path],
                                  title_suffix: str = "") -> bool:
        """
        Create a visualization PDF with layout elements highlighted.
        
        Args:
            pdf_path: Path to the original PDF file
            layout_result: Layout extraction results
            output_path: Path for the output visualization PDF
            title_suffix: Suffix to add to visualization title
            
        Returns:
            True if visualization was successful
        """
        try:
            # Open the original PDF
            doc = fitz.open(str(pdf_path))
            
            # Group elements by page
            page_elements = self._group_elements_by_page(layout_result.elements)
            
            # Process each page
            for page_num in range(doc.page_count):
                page = doc[page_num]
                elements = page_elements.get(page_num + 1, [])  # 1-indexed
                
                if elements:
                    self._draw_elements_on_page(page, elements, title_suffix)
            
            # Save the visualization
            doc.save(str(output_path))
            doc.close()
            
            logger.info(f"Visualization saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return False
    
    def compare_before_after(self,
                           pdf_path: Union[str, Path],
                           before_result: LayoutExtractionResult,
                           after_result: LayoutExtractionResult,
                           output_dir: Union[str, Path]) -> bool:
        """
        Create side-by-side comparison visualizations.
        
        Args:
            pdf_path: Path to the original PDF file
            before_result: Layout results before merging
            after_result: Layout results after merging
            output_dir: Directory to save comparison files
            
        Returns:
            True if comparison was successful
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # Create before visualization
            before_path = output_dir / "layout_before_merging.pdf"
            self.visualize_layout_elements(
                pdf_path, 
                before_result, 
                before_path,
                f"BEFORE Merging ({before_result.element_count} elements)"
            )
            
            # Create after visualization
            after_path = output_dir / "layout_after_merging.pdf"
            self.visualize_layout_elements(
                pdf_path, 
                after_result, 
                after_path,
                f"AFTER Merging ({after_result.element_count} elements)"
            )
            
            # Create detailed comparison
            comparison_path = output_dir / "layout_comparison_detailed.pdf"
            self._create_detailed_comparison(
                pdf_path, before_result, after_result, comparison_path
            )
            
            logger.info(f"Comparison visualizations saved to: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating comparison: {str(e)}")
            return False
    
    def _group_elements_by_page(self, elements: List[LayoutElement]) -> Dict[int, List[LayoutElement]]:
        """
        Group layout elements by page number.
        
        Args:
            elements: List of layout elements
            
        Returns:
            Dictionary mapping page numbers to element lists
        """
        page_elements = {}
        for element in elements:
            page_num = element.metadata.get('page_number', 1)
            if page_num not in page_elements:
                page_elements[page_num] = []
            page_elements[page_num].append(element)
        
        return page_elements
    
    def _draw_elements_on_page(self, 
                              page: fitz.Page, 
                              elements: List[LayoutElement],
                              title_suffix: str = "") -> None:
        """
        Draw layout elements on a PDF page.
        
        Args:
            page: PyMuPDF page object
            elements: List of elements to draw
            title_suffix: Suffix for page title
        """
        # Add title at the top of the page
        if title_suffix:
            title_rect = fitz.Rect(10, 10, page.rect.width - 10, 30)
            page.insert_textbox(
                title_rect,
                f"Layout Elements - {title_suffix}",
                fontsize=12,
                color=(0, 0, 0),
                fontname="helv"
            )
        
        # Draw each element
        for i, element in enumerate(elements):
            self._draw_single_element(page, element, i)
    
    def _draw_single_element(self, 
                           page: fitz.Page, 
                           element: LayoutElement, 
                           index: int) -> None:
        """
        Draw a single layout element on the page.
        
        Args:
            page: PyMuPDF page object
            element: Layout element to draw
            index: Element index for coloring
        """
        bbox = element.bbox
        
        # Create rectangle for bounding box
        rect = fitz.Rect(bbox.x1, bbox.y1, bbox.x2, bbox.y2)
        
        # Determine color based on element properties
        color = self._get_element_color(element)
        
        # Draw bounding box
        page.draw_rect(rect, color=color, width=self.line_width)
        
        # Add small index label in top-left corner
        index_label = f"[{index}]"
        index_rect = fitz.Rect(
            rect.x0 + 2,  # Small offset from left
            rect.y0 + 2,  # Small offset from top
            rect.x0 + 25,  # Slightly wider for better visibility
            rect.y0 + 15  # Slightly taller for better visibility
        )
        
        # Add semi-transparent background for index label
        page.draw_rect(index_rect, color=(1, 1, 1), fill=(1, 1, 1), width=0.5)
        
        # Add index label text with improved visibility
        try:
            page.insert_text(
                (index_rect.x0 + 2, index_rect.y0 + 10),  # Position text within the box
                index_label,
                fontsize=8,  # Slightly larger font
                color=color,
                fontname="helv",
                render_mode=0  # Normal rendering mode
            )
        except Exception as e:
            logger.warning(f"Failed to insert index label: {str(e)}")
        
        # Add element information if enabled
        if self.show_labels or self.show_element_ids:
            self._add_element_label(page, element, rect, color, index)
    
    def _get_element_color(self, element: LayoutElement) -> Tuple[float, float, float]:
        """
        Get color for an element based on its properties.
        
        Args:
            element: Layout element
            
        Returns:
            RGB color tuple (values 0-1)
        """
        # Check if element was merged
        if element.metadata.get('merged_elements', 0) > 1:
            return self.COLORS['merged']
        
        # Color by element type
        element_type = element.element_type.lower()
        return self.COLORS.get(element_type, self.COLORS['default'])
    
    def _add_element_label(self, 
                          page: fitz.Page, 
                          element: LayoutElement, 
                          rect: fitz.Rect,
                          color: Tuple[float, float, float],
                          index: int) -> None:
        """
        Add label to an element.
        
        Args:
            page: PyMuPDF page object
            element: Layout element
            rect: Element bounding rectangle
            color: Element color
            index: Element index
        """
        # Prepare label text
        label_parts = []
        
        if self.show_element_ids:
            label_parts.append(f"ID:{element.id}")
        
        if self.show_labels:
            label_parts.append(f"Type:{element.element_type}")
            
            # Add merge info if available
            merged_count = element.metadata.get('merged_elements', 0)
            if merged_count > 1:
                label_parts.append(f"Merged:{merged_count}")
        
        if not label_parts:
            return
        
        label_text = " | ".join(label_parts)
        
        # Position label outside the bounding box (above if possible)
        label_y = max(rect.y0 - 5, 5)  # Above the box, or at top of page
        label_rect = fitz.Rect(
            rect.x0, 
            label_y - self.font_size - 2,
            min(rect.x0 + len(label_text) * self.font_size * 0.6, page.rect.width - 5),
            label_y
        )
        
        # Add semi-transparent background for label
        page.draw_rect(label_rect, color=(1, 1, 1), fill=(1, 1, 1), width=0.5)
        
        # Add label text
        try:
            page.insert_textbox(
                label_rect,
                label_text,
                fontsize=self.font_size,
                color=color,
                fontname="helv"
            )
        except Exception:
            # Fallback if text insertion fails
            pass
    
    def _create_detailed_comparison(self,
                                  pdf_path: Union[str, Path],
                                  before_result: LayoutExtractionResult,
                                  after_result: LayoutExtractionResult,
                                  output_path: Union[str, Path]) -> None:
        """
        Create a detailed comparison showing merged elements.
        
        Args:
            pdf_path: Original PDF path
            before_result: Results before merging
            after_result: Results after merging
            output_path: Output path for comparison
        """
        doc = fitz.open(str(pdf_path))
        
        # Group elements by page
        before_pages = self._group_elements_by_page(before_result.elements)
        after_pages = self._group_elements_by_page(after_result.elements)
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_number = page_num + 1
            
            before_elements = before_pages.get(page_number, [])
            after_elements = after_pages.get(page_number, [])
            
            # Add comparison title
            reduction = len(before_elements) - len(after_elements)
            title = f"Page {page_number}: {len(before_elements)} → {len(after_elements)} elements (-{reduction})"
            
            title_rect = fitz.Rect(10, 10, page.rect.width - 10, 30)
            page.insert_textbox(
                title_rect,
                title,
                fontsize=12,
                color=(0, 0, 0),
                fontname="helv"
            )
            
            # Draw original fragments in light gray
            for element in before_elements:
                rect = fitz.Rect(element.bbox.x1, element.bbox.y1, element.bbox.x2, element.bbox.y2)
                page.draw_rect(rect, color=(0.7, 0.7, 0.7), width=0.5)
            
            # Draw merged elements with highlighting
            for element in after_elements:
                rect = fitz.Rect(element.bbox.x1, element.bbox.y1, element.bbox.x2, element.bbox.y2)
                
                merged_count = element.metadata.get('merged_elements', 0)
                if merged_count > 1:
                    # Highlight merged elements
                    page.draw_rect(rect, color=self.COLORS['merged'], width=2.0)
                    
                    # Add merge count label
                    label_rect = fitz.Rect(rect.x0, rect.y0 - 15, rect.x0 + 50, rect.y0)
                    page.insert_textbox(
                        label_rect,
                        f"×{merged_count}",
                        fontsize=10,
                        color=self.COLORS['merged'],
                        fontname="helv"
                    )
                else:
                    # Single elements
                    page.draw_rect(rect, color=(0, 0, 1), width=1.0)
        
        doc.save(str(output_path))
        doc.close()


def visualize_pdf_layout(pdf_path: Union[str, Path],
                        layout_result: LayoutExtractionResult,
                        output_path: Union[str, Path],
                        **kwargs) -> bool:
    """
    Convenience function to create a layout visualization.
    
    Args:
        pdf_path: Path to original PDF
        layout_result: Layout extraction results
        output_path: Output path for visualization
        **kwargs: Additional arguments for PdfLayoutVisualizer
        
    Returns:
        True if successful
    """
    visualizer = PdfLayoutVisualizer(**kwargs)
    return visualizer.visualize_layout_elements(pdf_path, layout_result, output_path)


def compare_layout_results(pdf_path: Union[str, Path],
                          before_result: LayoutExtractionResult,
                          after_result: LayoutExtractionResult,
                          output_dir: Union[str, Path],
                          **kwargs) -> bool:
    """
    Convenience function to compare layout results before and after processing.
    
    Args:
        pdf_path: Path to original PDF
        before_result: Results before processing
        after_result: Results after processing
        output_dir: Directory for output files
        **kwargs: Additional arguments for PdfLayoutVisualizer
        
    Returns:
        True if successful
    """
    visualizer = PdfLayoutVisualizer(**kwargs)
    return visualizer.compare_before_after(pdf_path, before_result, after_result, output_dir)


# Example usage
# python -m models.layout_detection.utils.visualiza_layout_elements
if __name__ == "__main__":
    # This would be used in conjunction with the PDF extractor
    from doc_chunking.layout_detection.layout_extraction.pdf_layout_extractor import PdfLayoutExtractor
    from doc_chunking.layout_detection.layout_extraction.pdf_style_cv_mix_extractor import PdfStyleCVMixLayoutExtractor
    
    # Test file path
    test_pdf = "tests/test_data/1-1 买卖合同（通用版）.pdf"
    
    if Path(test_pdf).exists():
        # Extract with merging
        extractor_with_merge = PdfLayoutExtractor(merge_fragments=True)
        result_after = extractor_with_merge._detect_layout(test_pdf)

        reulst_mix = PdfStyleCVMixLayoutExtractor(merge_fragments=True)._detect_layout(test_pdf)
        
        # Create comparison visualization
        output_dir = Path("visualization_output")
        output_dir.mkdir(exist_ok=True)
        
        success = compare_layout_results(
            test_pdf,
            result_after,
            reulst_mix,
            output_dir,
            show_labels=True,
            show_element_ids=True
        )
        
        if success:
            print("✅ Visualization completed successfully!")
            print(f"Check the files in: {output_dir}")
        else:
            print("❌ Visualization failed!")
    else:
        print(f"Test file not found: {test_pdf}")
