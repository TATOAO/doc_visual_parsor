"""
Pydantic schemas for DOCX document layout detection and processing.

This module provides Pydantic BaseModel equivalents of the dataclasses
defined in base_detector.py, offering better validation, serialization,
and type checking capabilities.
"""

from pydantic import BaseModel, Field, field_validator, computed_field
from typing import List, Optional, Dict, Any, Tuple, Union
from enum import Enum


class ElementType(str, Enum):
    """Standardized element types across all detection methods."""
    PLAIN_TEXT = "Plain Text"
    TITLE = "Title"
    HEADING = "Heading"
    FIGURE = "Figure"
    FIGURE_CAPTION = "Figure Caption"
    TABLE = "Table"
    TABLE_CAPTION = "Table Caption"
    TABLE_FOOTNOTE = "Table Footnote"
    ISOLATE_FORMULA = "Isolate Formula"
    FORMULA_CAPTION = "Formula Caption"
    LIST = "List"
    PARAGRAPH = "Paragraph"
    UNKNOWN = "Unknown"
    ABANDON = "Abandon"


class TextAlignment(str, Enum):
    """Text alignment options."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFY = "justify"
    DISTRIBUTE = "distribute"
    UNKNOWN = "unknown"


class FontInfo(BaseModel):
    """Font information for text runs."""
    name: Optional[str] = Field(None, description="Font name")
    size: Optional[float] = Field(None, description="Font size in points", ge=0)
    bold: Optional[bool] = Field(None, description="Whether text is bold")
    italic: Optional[bool] = Field(None, description="Whether text is italic")
    underline: Optional[bool] = Field(None, description="Whether text is underlined")
    color: Optional[str] = Field(None, description="Font color as hex code", pattern="^#[0-9A-Fa-f]{6}$")
    highlight: Optional[str] = Field(None, description="Highlight color as hex code", pattern="^#[0-9A-Fa-f]{6}$")
    strikethrough: Optional[bool] = Field(None, description="Whether text has strikethrough")
    superscript: Optional[bool] = Field(None, description="Whether text is superscript")
    subscript: Optional[bool] = Field(None, description="Whether text is subscript")
    
    @field_validator('color', 'highlight', mode='before')
    @classmethod
    def validate_color(cls, v):
        """Validate and normalize color values."""
        if v is None:
            return v
        if isinstance(v, str) and not v.startswith('#'):
            return f"#{v}"
        return v
    
    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "ignore"


class ParagraphFormat(BaseModel):
    """Paragraph formatting information."""
    alignment: Optional[TextAlignment] = Field(None, description="Text alignment")
    left_indent: Optional[float] = Field(None, description="Left indent in points", ge=0)
    right_indent: Optional[float] = Field(None, description="Right indent in points", ge=0)
    first_line_indent: Optional[float] = Field(None, description="First line indent in points")
    space_before: Optional[float] = Field(None, description="Space before paragraph in points", ge=0)
    space_after: Optional[float] = Field(None, description="Space after paragraph in points", ge=0)
    line_spacing: Optional[float] = Field(None, description="Line spacing multiplier", gt=0)
    line_spacing_rule: Optional[str] = Field(None, description="Line spacing rule")
    keep_together: Optional[bool] = Field(None, description="Keep paragraph together")
    keep_with_next: Optional[bool] = Field(None, description="Keep with next paragraph")
    page_break_before: Optional[bool] = Field(None, description="Page break before paragraph")
    widow_control: Optional[bool] = Field(None, description="Widow control enabled")
    
    @field_validator('line_spacing_rule')
    @classmethod
    def validate_line_spacing_rule(cls, v):
        """Validate line spacing rule values."""
        if v is None:
            return v
        valid_rules = ['single', 'multiple', 'exact', 'at_least']
        if v.lower() not in valid_rules:
            raise ValueError(f"Line spacing rule must be one of: {valid_rules}")
        return v.lower()
    
    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "ignore"


class RunInfo(BaseModel):
    """Information about a text run with specific formatting."""
    text: str = Field(..., description="Text content of the run")
    font: FontInfo = Field(..., description="Font information for the run")
    start_index: Optional[int] = Field(None, description="Character start index within element", ge=0)
    end_index: Optional[int] = Field(None, description="Character end index within element", ge=0)
    
    @field_validator('end_index')
    @classmethod
    def validate_end_index(cls, v, info):
        """Ensure end_index is greater than start_index."""
        if v is not None and hasattr(info, 'data') and 'start_index' in info.data:
            start_index = info.data.get('start_index')
            if start_index is not None and v <= start_index:
                raise ValueError("end_index must be greater than start_index")
        return v
    
    @computed_field
    @property
    def text_length(self) -> int:
        """Get the length of the text content."""
        return len(self.text)
    
    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "ignore"


class StyleInfo(BaseModel):
    """Comprehensive style information for layout elements."""
    # Style identification
    style_name: Optional[str] = Field(None, description="Name of the style")
    style_type: Optional[str] = Field(None, description="Type of style (paragraph, character, table, etc.)")
    builtin: Optional[bool] = Field(None, description="Whether it's a built-in style")
    
    # Paragraph-level formatting
    paragraph_format: Optional[ParagraphFormat] = Field(None, description="Paragraph formatting information")
    
    # Text runs with individual formatting
    runs: Optional[List[RunInfo]] = Field([], description="List of text runs with formatting")
    
    # Dominant/primary font info (for backward compatibility)
    primary_font: Optional[FontInfo] = Field(None, description="Primary font information")
    
    # Table-specific styling (if applicable)
    table_style: Optional[Dict[str, Any]] = Field(None, description="Table-specific styling information")
    
    # List-specific styling (if applicable)
    list_style: Optional[Dict[str, Any]] = Field(None, description="List-specific styling information")
    
    # Custom properties
    custom_properties: Optional[Dict[str, Any]] = Field(None, description="Custom style properties")
    
    @field_validator('style_type')
    @classmethod
    def validate_style_type(cls, v):
        """Validate style type values."""
        if v is None:
            return v
        valid_types = ['paragraph', 'character', 'table', 'list', 'numbering']
        if v.lower() not in valid_types:
            # Don't raise error, just log warning for flexibility
            pass
        return v.lower() if v else v
    
    @computed_field
    @property
    def has_formatting(self) -> bool:
        """Check if the style has any formatting information."""
        return any([
            self.paragraph_format is not None,
            self.runs is not None and len(self.runs) > 0,
            self.primary_font is not None,
            self.table_style is not None,
            self.list_style is not None
        ])
    
    @computed_field
    @property
    def run_count(self) -> int:
        """Get the number of text runs."""
        return len(self.runs) if self.runs else 0
    
    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "ignore"


class BoundingBox(BaseModel):
    """Standardized bounding box representation."""
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate")
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")
    
    @field_validator('x2')
    @classmethod
    def validate_x2(cls, v, info):
        """Ensure x2 is greater than x1."""
        if hasattr(info, 'data') and 'x1' in info.data and v <= info.data['x1']:
            raise ValueError("x2 must be greater than x1")
        return v
    
    @field_validator('y2')
    @classmethod
    def validate_y2(cls, v, info):
        """Ensure y2 is greater than y1."""
        if hasattr(info, 'data') and 'y1' in info.data and v <= info.data['y1']:
            raise ValueError("y2 must be greater than y1")
        return v
    
    @computed_field
    @property
    def width(self) -> float:
        """Calculate width of the bounding box."""
        return self.x2 - self.x1
    
    @computed_field
    @property
    def height(self) -> float:
        """Calculate height of the bounding box."""
        return self.y2 - self.y1
    
    @computed_field
    @property
    def center(self) -> Tuple[float, float]:
        """Calculate center point of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @computed_field
    @property
    def area(self) -> float:
        """Calculate area of the bounding box."""
        return self.width * self.height
    
    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "ignore"


class LayoutElement(BaseModel):
    """Standardized layout element representation."""
    id: int = Field(..., description="Unique identifier for the element", ge=0)
    element_type: ElementType = Field(..., description="Type of the layout element")
    confidence: float = Field(..., description="Confidence score for the detection", ge=0.0, le=1.0)
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box coordinates")
    text: Optional[str] = Field(None, description="Text content of the element")
    style: Optional[StyleInfo] = Field(StyleInfo(runs=[]), description="Style information")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        """Validate and clean text content."""
        if v is not None:
            # Strip leading/trailing whitespace but preserve internal formatting
            return v.strip() if v.strip() else None
        return v
    
    @computed_field
    @property
    def has_text(self) -> bool:
        """Check if element has text content."""
        return self.text is not None and len(self.text.strip()) > 0
    
    @computed_field
    @property
    def has_bbox(self) -> bool:
        """Check if element has bounding box."""
        return self.bbox is not None
    
    @computed_field
    @property
    def has_style(self) -> bool:
        """Check if element has style information."""
        return self.style is not None
    
    @computed_field  
    @property
    def text_length(self) -> int:
        """Get length of text content."""
        return len(self.text) if self.text else 0
    
    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "ignore"


class LayoutExtractionResult(BaseModel):
    """Container for layout extraction results."""
    elements: List[LayoutElement] = Field(..., description="List of detected layout elements")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Result metadata")
    
    @computed_field
    @property
    def element_count(self) -> int:
        """Get total number of elements."""
        return len(self.elements)
    
    @computed_field
    @property
    def element_type_counts(self) -> Dict[str, int]:
        """Get count of each element type."""
        counts = {}
        for element in self.elements:
            element_type = element.element_type.value
            counts[element_type] = counts.get(element_type, 0) + 1
        return counts
    
    @computed_field
    @property
    def confidence_stats(self) -> Dict[str, float]:
        """Get confidence statistics."""
        if not self.elements:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'median': 0.0}
        
        confidences = [elem.confidence for elem in self.elements]
        confidences.sort()
        n = len(confidences)
        
        return {
            'min': min(confidences),
            'max': max(confidences),
            'mean': sum(confidences) / n,
            'median': confidences[n // 2] if n % 2 == 1 else (confidences[n // 2 - 1] + confidences[n // 2]) / 2
        }
    
    def filter_by_confidence(self, min_confidence: float = 0.5) -> 'LayoutExtractionResult':
        """Filter results by minimum confidence threshold."""
        filtered_elements = [elem for elem in self.elements if elem.confidence >= min_confidence]
        return LayoutExtractionResult(
            elements=filtered_elements,
            metadata=self.metadata
        )
    
    def filter_by_type(self, element_types: List[ElementType]) -> 'LayoutExtractionResult':
        """Filter results by element types."""
        filtered_elements = [elem for elem in self.elements if elem.element_type in element_types]
        return LayoutExtractionResult(
            elements=filtered_elements,
            metadata=self.metadata
        )
    
    def sort_by_reading_order(self) -> 'LayoutExtractionResult':
        """Sort elements by reading order (top-to-bottom, left-to-right)."""
        if not self.elements or not all(elem.bbox for elem in self.elements):
            return self
        
        # Sort by y-coordinate first (top to bottom), then by x-coordinate (left to right)
        sorted_elements = sorted(self.elements, key=lambda elem: (elem.bbox.y1, elem.bbox.x1))
        return LayoutExtractionResult(
            elements=sorted_elements,
            metadata=self.metadata
        )
    
    def get_elements_by_type(self, element_type: ElementType) -> List[LayoutElement]:
        """Get all elements of a specific type."""
        return [elem for elem in self.elements if elem.element_type == element_type]
    
    def get_text_content(self, separator: str = "\n") -> str:
        """Get all text content concatenated."""
        text_parts = [elem.text for elem in self.elements if elem.has_text]
        return separator.join(text_parts)
    
    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "ignore"


# Export all models for easy import
__all__ = [
    "ElementType",
    "TextAlignment", 
    "FontInfo",
    "ParagraphFormat",
    "RunInfo",
    "StyleInfo",
    "BoundingBox",
    "LayoutElement",
    "LayoutExtractionResult"
]
