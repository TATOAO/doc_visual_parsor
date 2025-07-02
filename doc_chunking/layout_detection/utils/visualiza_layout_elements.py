"""
PDF Layout Element Visualization Utility

This module provides visualization tools for PDF layout elements, allowing users to
see bounding boxes, element types, and merging results overlaid on the original PDF.
"""

import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import io
import tempfile
from doc_chunking.schemas.layout_schemas import LayoutExtractionResult, LayoutElement, BoundingBox, ElementType

logger = logging.getLogger(__name__)

# Type alias for input data - consistent with cv_detector.py
InputDataType = Union[str, Path, bytes, io.BytesIO, io.BufferedReader, Any]

def _load_pdf_document(input_data: InputDataType) -> fitz.Document:
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
                 show_element_ids: bool = True,
                 colors: Optional[Dict[ElementType, str]] = None,
                 figsize: tuple = (12, 16),
                 dpi: int = 150):
        """
        Initialize the visualizer.
        
        Args:
            line_width: Width of bounding box lines
            font_size: Font size for labels
            show_labels: Whether to show element type labels
            show_element_ids: Whether to show element IDs
            colors: Custom color mapping for element types
            figsize: Figure size for visualization
            dpi: DPI for rendering
        """
        self.line_width = line_width
        self.font_size = font_size
        self.show_labels = show_labels
        self.show_element_ids = show_element_ids
        self.colors = colors or self.COLORS
        self.figsize = figsize
        self.dpi = dpi
    
    def visualize_layout_elements(self, 
                                pdf_input: InputDataType,
                                layout_result: LayoutExtractionResult, 
                                output_path: Union[str, Path]) -> bool:
        """
        Create a visualization of layout elements overlaid on PDF pages.
        
        Args:
            pdf_input: Path to original PDF file or file object
            layout_result: Layout extraction results
            output_path: Output path for the visualization PDF
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load PDF document
            doc = _load_pdf_document(pdf_input)
            
            # Group elements by page
            page_elements = {}
            for element in layout_result.elements:
                page_num = element.metadata.get('page_number', 1)
                if page_num not in page_elements:
                    page_elements[page_num] = []
                page_elements[page_num].append(element)
            
            # Create visualization
            with PdfPages(output_path) as pdf:
                for page_num in range(1, doc.page_count + 1):
                    elements = page_elements.get(page_num, [])
                    
                    fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
                    
                    # Get page dimensions
                    page = doc[page_num - 1]  # Convert to 0-indexed
                    page_rect = page.rect
                    
                    # Set up the plot
                    ax.set_xlim(0, page_rect.width)
                    ax.set_ylim(0, page_rect.height)
                    ax.invert_yaxis()  # PDF coordinates start from top
                    ax.set_aspect('equal')
                    ax.set_title(f'Layout Analysis - Page {page_num}')
                    
                    # Draw layout elements
                    for element in elements:
                        if element.bbox:
                            self._draw_element(ax, element)
                    
                    # Add legend
                    self._add_legend(ax, elements)
                    
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
            
            doc.close()
            return True
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return False
    
    def compare_before_after(self, 
                           pdf_input: InputDataType,
                           before_result: LayoutExtractionResult,
                           after_result: LayoutExtractionResult,
                           output_dir: Union[str, Path]) -> bool:
        """
        Create a side-by-side comparison of before and after layout results.
        
        Args:
            pdf_input: Path to original PDF file or file object
            before_result: Layout results before processing
            after_result: Layout results after processing  
            output_dir: Directory to save comparison files
            
        Returns:
            True if successful
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            before_path = output_dir / "before_layout.pdf"
            after_path = output_dir / "after_layout.pdf"
            comparison_path = output_dir / "comparison.pdf"
            
            # Create individual visualizations
            self.visualize_layout_elements(pdf_input, before_result, before_path)
            self.visualize_layout_elements(pdf_input, after_result, after_path)
            
            # Create side-by-side comparison
            self._create_side_by_side_comparison(
                pdf_input, before_result, after_result, comparison_path
            )
            
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
    
    def _draw_element(self, ax: plt.Axes, element: LayoutElement) -> None:
        """
        Draw a single layout element on the plot.
        
        Args:
            ax: Matplotlib Axes object
            element: Layout element to draw
        """
        bbox = element.bbox
        
        # Create rectangle for bounding box
        rect = patches.Rectangle(
            (bbox.x1, bbox.y1),
            bbox.x2 - bbox.x1,
            bbox.y2 - bbox.y1,
            linewidth=self.line_width,
            edgecolor=self.colors.get(element.element_type, self.COLORS['default']),
            facecolor='none'
        )
        
        # Draw bounding box
        ax.add_patch(rect)
        
        # Add element information if enabled
        if self.show_labels or self.show_element_ids:
            self._add_element_label(ax, element, rect)
    
    def _add_element_label(self, ax: plt.Axes, element: LayoutElement, rect: patches.Rectangle) -> None:
        """
        Add label to an element.
        
        Args:
            ax: Matplotlib Axes object
            element: Layout element
            rect: Element bounding rectangle
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
        label_y = max(rect.get_y() - 5, 5)  # Above the box, or at top of page
        label_rect = patches.Rectangle(
            (rect.get_x(), label_y - self.font_size - 2),
            rect.get_width(),
            self.font_size + 4,
            linewidth=0.5,
            edgecolor='none',
            facecolor='white',
            alpha=0.7
        )
        
        # Add label text
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            label_y,
            label_text,
            ha='center',
            va='center',
            fontsize=self.font_size,
            color=self.colors.get(element.element_type, self.COLORS['default']),
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
        )
        
        # Add label rectangle
        ax.add_patch(label_rect)
    
    def _add_legend(self, ax: plt.Axes, elements: List[LayoutElement]) -> None:
        """
        Add a legend to the plot.
        
        Args:
            ax: Matplotlib Axes object
            elements: List of layout elements
        """
        handles = [
            patches.Patch(facecolor=self.colors.get(element.element_type, self.COLORS['default']), label=element.element_type)
            for element in elements
        ]
        ax.legend(handles=handles, loc='upper right')
    
    def _create_side_by_side_comparison(self, 
                                      pdf_input: InputDataType,
                                      before_result: LayoutExtractionResult,
                                      after_result: LayoutExtractionResult,
                                      output_path: Union[str, Path]) -> None:
        """Create a side-by-side comparison visualization."""
        doc = _load_pdf_document(pdf_input)
        
        # Group elements by page for both results
        before_pages = self._group_elements_by_page(before_result.elements)
        after_pages = self._group_elements_by_page(after_result.elements)
        
        with PdfPages(output_path) as pdf:
            for page_num in range(1, doc.page_count + 1):
                before_elements = before_pages.get(page_num, [])
                after_elements = after_pages.get(page_num, [])
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 16))
                
                # Get page dimensions
                page = doc[page_num - 1]
                page_rect = page.rect
                
                # Set up both plots
                for ax in [ax1, ax2]:
                    ax.set_xlim(0, page_rect.width)
                    ax.set_ylim(0, page_rect.height)
                    ax.invert_yaxis()
                    ax.set_aspect('equal')
                
                ax1.set_title(f'Before - Page {page_num} ({len(before_elements)} elements)')
                ax2.set_title(f'After - Page {page_num} ({len(after_elements)} elements)')
                
                # Draw elements
                for element in before_elements:
                    if element.bbox:
                        self._draw_element(ax1, element)
                
                for element in after_elements:
                    if element.bbox:
                        self._draw_element(ax2, element)
                
                # Add legend to the right plot
                self._add_legend(ax2, after_elements)
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        
        doc.close()


def visualize_pdf_layout(pdf_input: InputDataType,
                        layout_result: LayoutExtractionResult,
                        output_path: Union[str, Path],
                        **kwargs) -> bool:
    """
    Convenience function to create a layout visualization.
    
    Args:
        pdf_input: Path to original PDF or file object
        layout_result: Layout extraction results
        output_path: Output path for visualization
        **kwargs: Additional arguments for PdfLayoutVisualizer
        
    Returns:
        True if successful
    """
    visualizer = PdfLayoutVisualizer(**kwargs)
    return visualizer.visualize_layout_elements(pdf_input, layout_result, output_path)


def compare_layout_results(pdf_input: InputDataType,
                          before_result: LayoutExtractionResult,
                          after_result: LayoutExtractionResult,
                          output_dir: Union[str, Path],
                          **kwargs) -> bool:
    """
    Convenience function to create a comparison visualization.
    
    Args:
        pdf_input: Path to original PDF or file object
        before_result: Layout results before processing
        after_result: Layout results after processing
        output_dir: Directory to save comparison files
        **kwargs: Additional arguments for PdfLayoutVisualizer
        
    Returns:
        True if successful
    """
    visualizer = PdfLayoutVisualizer(**kwargs)
    return visualizer.compare_before_after(pdf_input, before_result, after_result, output_dir)


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
