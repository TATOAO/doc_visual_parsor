"""
Abstract Base Class for Document Layout Detection

This module defines the abstract interface for all layout detection implementations.
Different detection methods (CV-based, rule-based, etc.) should inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
import numpy as np
from dataclasses import dataclass
from enum import Enum

class ElementType(Enum):
    """Standardized element types across all detection methods."""
    TEXT = "Text"
    TITLE = "Title"
    HEADING = "Heading"
    FIGURE = "Figure"
    FIGURE_CAPTION = "Figure Caption"
    TABLE = "Table"
    TABLE_CAPTION = "Table Caption"
    HEADER = "Header"
    FOOTER = "Footer"
    REFERENCE = "Reference"
    EQUATION = "Equation"
    LIST = "List"
    PARAGRAPH = "Paragraph"
    UNKNOWN = "Unknown"

class TextAlignment(Enum):
    """Text alignment options."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFY = "justify"
    DISTRIBUTE = "distribute"
    UNKNOWN = "unknown"

@dataclass
class FontInfo:
    """Font information for text runs."""
    name: Optional[str] = None
    size: Optional[float] = None  # in points
    bold: Optional[bool] = None
    italic: Optional[bool] = None
    underline: Optional[bool] = None
    color: Optional[str] = None  # hex color code
    highlight: Optional[str] = None  # highlight color
    strikethrough: Optional[bool] = None
    superscript: Optional[bool] = None
    subscript: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'name': self.name,
            'size': self.size,
            'bold': self.bold,
            'italic': self.italic,
            'underline': self.underline,
            'color': self.color,
            'highlight': self.highlight,
            'strikethrough': self.strikethrough,
            'superscript': self.superscript,
            'subscript': self.subscript
        }

@dataclass 
class ParagraphFormat:
    """Paragraph formatting information."""
    alignment: Optional[TextAlignment] = None
    left_indent: Optional[float] = None  # in points
    right_indent: Optional[float] = None  # in points
    first_line_indent: Optional[float] = None  # in points
    space_before: Optional[float] = None  # in points
    space_after: Optional[float] = None  # in points
    line_spacing: Optional[float] = None  # line spacing multiplier
    line_spacing_rule: Optional[str] = None  # single, multiple, exact, at_least
    keep_together: Optional[bool] = None
    keep_with_next: Optional[bool] = None
    page_break_before: Optional[bool] = None
    widow_control: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'alignment': self.alignment.value if self.alignment else None,
            'left_indent': self.left_indent,
            'right_indent': self.right_indent,
            'first_line_indent': self.first_line_indent,
            'space_before': self.space_before,
            'space_after': self.space_after,
            'line_spacing': self.line_spacing,
            'line_spacing_rule': self.line_spacing_rule,
            'keep_together': self.keep_together,
            'keep_with_next': self.keep_with_next,
            'page_break_before': self.page_break_before,
            'widow_control': self.widow_control
        }

@dataclass
class RunInfo:
    """Information about a text run with specific formatting."""
    text: str
    font: FontInfo
    start_index: Optional[int] = None  # character start index within element
    end_index: Optional[int] = None    # character end index within element
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'text': self.text,
            'font': self.font.to_dict(),
            'start_index': self.start_index,
            'end_index': self.end_index
        }

@dataclass
class StyleInfo:
    """Comprehensive style information for layout elements."""
    # Style identification
    style_name: Optional[str] = None
    style_type: Optional[str] = None  # paragraph, character, table, etc.
    builtin: Optional[bool] = None  # whether it's a built-in style
    
    # Paragraph-level formatting
    paragraph_format: Optional[ParagraphFormat] = None
    
    # Text runs with individual formatting
    runs: Optional[List[RunInfo]] = None
    
    # Dominant/primary font info (for backward compatibility)
    primary_font: Optional[FontInfo] = None
    
    # Table-specific styling (if applicable)
    table_style: Optional[Dict[str, Any]] = None
    
    # List-specific styling (if applicable)
    list_style: Optional[Dict[str, Any]] = None
    
    # Custom properties
    custom_properties: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            'style_name': self.style_name,
            'style_type': self.style_type,
            'builtin': self.builtin,
        }
        
        if self.paragraph_format:
            result['paragraph_format'] = self.paragraph_format.to_dict()
        
        if self.runs:
            result['runs'] = [run.to_dict() for run in self.runs]
        
        if self.primary_font:
            result['primary_font'] = self.primary_font.to_dict()
        
        if self.table_style:
            result['table_style'] = self.table_style
        
        if self.list_style:
            result['list_style'] = self.list_style
        
        if self.custom_properties:
            result['custom_properties'] = self.custom_properties
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StyleInfo':
        """Create StyleInfo from dictionary."""
        # Parse paragraph format
        paragraph_format = None
        if 'paragraph_format' in data and data['paragraph_format']:
            pf_data = data['paragraph_format']
            paragraph_format = ParagraphFormat(
                alignment=TextAlignment(pf_data['alignment']) if pf_data.get('alignment') else None,
                left_indent=pf_data.get('left_indent'),
                right_indent=pf_data.get('right_indent'),
                first_line_indent=pf_data.get('first_line_indent'),
                space_before=pf_data.get('space_before'),
                space_after=pf_data.get('space_after'),
                line_spacing=pf_data.get('line_spacing'),
                line_spacing_rule=pf_data.get('line_spacing_rule'),
                keep_together=pf_data.get('keep_together'),
                keep_with_next=pf_data.get('keep_with_next'),
                page_break_before=pf_data.get('page_break_before'),
                widow_control=pf_data.get('widow_control')
            )
        
        # Parse runs
        runs = None
        if 'runs' in data and data['runs']:
            runs = []
            for run_data in data['runs']:
                font_info = FontInfo(**run_data.get('font', {}))
                run = RunInfo(
                    text=run_data.get('text', ''),
                    font=font_info,
                    start_index=run_data.get('start_index'),
                    end_index=run_data.get('end_index')
                )
                runs.append(run)
        
        # Parse primary font
        primary_font = None
        if 'primary_font' in data and data['primary_font']:
            primary_font = FontInfo(**data['primary_font'])
        
        return cls(
            style_name=data.get('style_name'),
            style_type=data.get('style_type'),
            builtin=data.get('builtin'),
            paragraph_format=paragraph_format,
            runs=runs,
            primary_font=primary_font,
            table_style=data.get('table_style'),
            list_style=data.get('list_style'),
            custom_properties=data.get('custom_properties')
        )

@dataclass
class BoundingBox:
    """Standardized bounding box representation."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        return self.width * self.height

@dataclass
class LayoutElement:
    """Standardized layout element representation."""
    id: int
    element_type: ElementType
    confidence: float
    bbox: Optional[BoundingBox] = None
    text: Optional[str] = None
    style: Optional[StyleInfo] = None  # Enhanced from Dict to StyleInfo
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert element to dictionary format."""
        result = {
            'id': self.id,
            'type': self.element_type.value,
            'confidence': self.confidence,
        }
        
        if self.bbox:
            result['bbox'] = {
                'x1': self.bbox.x1,
                'y1': self.bbox.y1,
                'x2': self.bbox.x2,
                'y2': self.bbox.y2,
                'width': self.bbox.width,
                'height': self.bbox.height
            }
        
        if self.text:
            result['text'] = self.text
            
        if self.style:
            result['style'] = self.style.to_dict()
            
        if self.metadata:
            result['metadata'] = self.metadata
            
        return result

class LayoutDetectionResult:
    """Container for layout detection results."""
    
    def __init__(self, elements: List[LayoutElement]):
        self.elements = elements
        
    def get_elements(self) -> List[LayoutElement]:
        """Get all detected elements."""
        return self.elements
    
    def get_elements_dict(self) -> List[Dict[str, Any]]:
        """Get elements as list of dictionaries (for backward compatibility)."""
        return [element.to_dict() for element in self.elements]
    
    def filter_by_confidence(self, min_confidence: float = 0.5) -> 'LayoutDetectionResult':
        """Filter results by minimum confidence threshold."""
        filtered_elements = [elem for elem in self.elements if elem.confidence >= min_confidence]
        return LayoutDetectionResult(filtered_elements)
    
    def filter_by_type(self, element_types: List[ElementType]) -> 'LayoutDetectionResult':
        """Filter results by element types."""
        filtered_elements = [elem for elem in self.elements if elem.element_type in element_types]
        return LayoutDetectionResult(filtered_elements)
    
    def sort_by_reading_order(self) -> 'LayoutDetectionResult':
        """Sort elements by reading order (top-to-bottom, left-to-right)."""
        if not self.elements or not all(elem.bbox for elem in self.elements):
            return self
        
        # Sort by y-coordinate first (top to bottom), then by x-coordinate (left to right)
        sorted_elements = sorted(self.elements, key=lambda elem: (elem.bbox.y1, elem.bbox.x1))
        return LayoutDetectionResult(sorted_elements)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected elements."""
        stats = {
            'total_elements': len(self.elements),
            'element_types': {},
            'confidence_stats': {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0
            }
        }
        
        if not self.elements:
            return stats
        
        # Count by type
        for element in self.elements:
            element_type = element.element_type.value
            stats['element_types'][element_type] = stats['element_types'].get(element_type, 0) + 1
        
        # Confidence statistics
        confidences = [elem.confidence for elem in self.elements]
        stats['confidence_stats'] = {
            'min': min(confidences),
            'max': max(confidences),
            'mean': sum(confidences) / len(confidences)
        }
        
        return stats

class BaseLayoutDetector(ABC):
    """
    Abstract base class for document layout detection.
    
    All layout detection implementations should inherit from this class
    and implement the required abstract methods.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.5,
                 device: str = "auto",
                 **kwargs):
        """
        Initialize the layout detector.
        
        Args:
            confidence_threshold: Minimum confidence for detections
            device: Device to use for computation
            **kwargs: Additional detector-specific parameters
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.is_initialized = False
        
    @abstractmethod
    def _initialize_detector(self) -> None:
        """Initialize the detector (load models, setup resources, etc.)."""
        pass
    
    @abstractmethod
    def _detect_layout(self, 
                      input_data: Any,
                      confidence_threshold: Optional[float] = None,
                      **kwargs) -> LayoutDetectionResult:
        """
        Core detection method to be implemented by subclasses.
        
        Args:
            input_data: Input data (image, document path, etc.)
            confidence_threshold: Override default confidence threshold
            **kwargs: Additional detection parameters
            
        Returns:
            LayoutDetectionResult containing detected elements
        """
        pass
    
    def detect(self, 
               input_data: Any,
               confidence_threshold: Optional[float] = None,
               **kwargs) -> LayoutDetectionResult:
        """
        Public interface for layout detection.
        
        Args:
            input_data: Input data (image, document path, etc.)
            confidence_threshold: Override default confidence threshold
            **kwargs: Additional detection parameters
            
        Returns:
            LayoutDetectionResult containing detected elements
        """
        if not self.is_initialized:
            self._initialize_detector()
            self.is_initialized = True
        
        return self._detect_layout(input_data, confidence_threshold, **kwargs)
    
    def detect_batch(self, 
                    input_data_list: List[Any],
                    confidence_threshold: Optional[float] = None,
                    **kwargs) -> List[LayoutDetectionResult]:
        """
        Detect layout in multiple inputs.
        
        Args:
            input_data_list: List of input data
            confidence_threshold: Override default confidence threshold
            **kwargs: Additional detection parameters
            
        Returns:
            List of LayoutDetectionResult objects
        """
        results = []
        for input_data in input_data_list:
            try:
                result = self.detect(input_data, confidence_threshold, **kwargs)
                results.append(result)
            except Exception as e:
                # Create empty result for failed detection
                print(f"Failed to process input {input_data}: {str(e)}")
                results.append(LayoutDetectionResult([]))
        
        return results
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported input formats.
        
        Returns:
            List of supported file extensions or format descriptions
        """
        pass
    
    @abstractmethod
    def get_detector_info(self) -> Dict[str, Any]:
        """
        Get information about the detector.
        
        Returns:
            Dictionary containing detector information
        """
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate if input data is supported by this detector.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if input is supported, False otherwise
        """
        # Default implementation - subclasses can override
        return True
    
    def visualize(self, 
                  input_data: Any,
                  result: Optional[LayoutDetectionResult] = None,
                  save_path: Optional[str] = None,
                  **kwargs) -> Optional[np.ndarray]:
        """
        Visualize detection results (optional implementation).
        
        Args:
            input_data: Original input data
            result: Detection result (if None, will run detection)
            save_path: Path to save visualization
            **kwargs: Additional visualization parameters
            
        Returns:
            Visualized image as numpy array (if applicable)
        """
        # Default implementation - subclasses can override
        print("Visualization not implemented for this detector type")
        return None 