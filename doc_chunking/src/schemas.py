"""
Pydantic schemas for document layout detection and processing.

This module provides Pydantic BaseModel equivalents for layout elements,
offering better validation, serialization, and type checking capabilities.
"""

from pydantic import BaseModel, Field, field_validator, computed_field
from typing import List, Optional, Dict, Any, Tuple, Union
from enum import Enum
import hashlib


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
    """Information about a text run (continuous text with same formatting)."""
    text: str = Field(..., description="The text content of this run")
    start_index: Optional[int] = Field(None, description="Start index in the full text", ge=0)
    end_index: Optional[int] = Field(None, description="End index in the full text", ge=0)
    font: Optional[FontInfo] = Field(None, description="Font information for this run")
    
    @field_validator('end_index')
    @classmethod
    def validate_end_index(cls, v, info):
        """Validate that end_index is greater than start_index."""
        if v is not None and 'start_index' in info.data and info.data['start_index'] is not None:
            if v <= info.data['start_index']:
                raise ValueError("end_index must be greater than start_index")
        return v
    
    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "ignore"


class StyleInfo(BaseModel):
    """Style information for a layout element."""
    font: Optional[FontInfo] = Field(None, description="Primary font information")
    paragraph_format: Optional[ParagraphFormat] = Field(None, description="Paragraph formatting")
    runs: Optional[List[RunInfo]] = Field(None, description="Text runs with individual formatting")
    
    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "ignore"


class BoundingBox(BaseModel):
    """Bounding box coordinates for a layout element."""
    x1: float = Field(..., description="Left coordinate", ge=0)
    y1: float = Field(..., description="Top coordinate", ge=0)
    x2: float = Field(..., description="Right coordinate", ge=0)
    y2: float = Field(..., description="Bottom coordinate", ge=0)
    
    @field_validator('x2', 'y2')
    @classmethod
    def validate_coordinates(cls, v, info):
        """Validate that x2 > x1 and y2 > y1."""
        if 'x1' in info.data and info.data['x1'] is not None and v <= info.data['x1']:
            raise ValueError("x2 must be greater than x1")
        if 'y1' in info.data and info.data['y1'] is not None and v <= info.data['y1']:
            raise ValueError("y2 must be greater than y1")
        return v
    
    @computed_field
    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.x2 - self.x1
    
    @computed_field
    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.y2 - self.y1
    
    @computed_field
    @property
    def area(self) -> float:
        """Area of the bounding box."""
        return self.width * self.height
    
    @computed_field
    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if this bounding box intersects with another."""
        return not (self.x2 <= other.x1 or other.x2 <= self.x1 or 
                   self.y2 <= other.y1 or other.y2 <= self.y1)
    
    def contains(self, other: 'BoundingBox') -> bool:
        """Check if this bounding box contains another."""
        return (self.x1 <= other.x1 and self.y1 <= other.y1 and 
                self.x2 >= other.x2 and self.y2 >= other.y2)
    
    def intersection_area(self, other: 'BoundingBox') -> float:
        """Calculate intersection area with another bounding box."""
        if not self.intersects(other):
            return 0.0
        
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        return (x2 - x1) * (y2 - y1)
    
    def union_area(self, other: 'BoundingBox') -> float:
        """Calculate union area with another bounding box."""
        return self.area + other.area - self.intersection_area(other)
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union (IoU) with another bounding box."""
        intersection = self.intersection_area(other)
        union = self.union_area(other)
        return intersection / union if union > 0 else 0.0
    
    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "ignore"


class LayoutElement(BaseModel):
    """A single layout element detected in a document."""
    id: int = Field(..., description="Unique identifier for this element")
    element_type: ElementType = Field(..., description="Type of layout element")
    text: Optional[str] = Field(None, description="Text content of the element")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box coordinates")
    confidence: Optional[float] = Field(None, description="Detection confidence score", ge=0, le=1)
    style: Optional[StyleInfo] = Field(None, description="Style information")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @computed_field
    @property
    def has_text(self) -> bool:
        """Check if this element has text content."""
        return self.text is not None and len(self.text.strip()) > 0
    
    @computed_field
    @property
    def has_bbox(self) -> bool:
        """Check if this element has bounding box information."""
        return self.bbox is not None
    
    @computed_field
    @property
    def text_length(self) -> int:
        """Get the length of text content."""
        return len(self.text) if self.text else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return self.model_dump()
    
    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "ignore"


class LayoutExtractionResult(BaseModel):
    """Container for layout extraction results."""
    elements: List[LayoutElement] = Field(default_factory=list, description="List of detected layout elements")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the extraction")
    
    @computed_field
    @property
    def element_count(self) -> int:
        """Get the number of detected elements."""
        return len(self.elements)
    
    @computed_field
    @property
    def has_elements(self) -> bool:
        """Check if any elements were detected."""
        return len(self.elements) > 0
    
    def get_elements_by_type(self, element_type: ElementType) -> List[LayoutElement]:
        """Get all elements of a specific type."""
        return [elem for elem in self.elements if elem.element_type == element_type]
    
    def get_text_content(self, separator: str = "\n") -> str:
        """Get all text content concatenated."""
        text_parts = [elem.text for elem in self.elements if elem.has_text]
        return separator.join(text_parts)
    
    def filter_by_confidence(self, min_confidence: float = 0.5) -> 'LayoutExtractionResult':
        """Filter results by minimum confidence threshold."""
        filtered_elements = [elem for elem in self.elements 
                           if elem.confidence is None or elem.confidence >= min_confidence]
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
