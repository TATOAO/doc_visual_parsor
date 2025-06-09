from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Union, Dict, Any
from typing_extensions import Self
from enum import Enum


class DocumentType(str, Enum):
    """Document type enumeration"""
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    TEXT = "text"


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: float = Field(description="Left coordinate")
    y1: float = Field(description="Top coordinate") 
    x2: float = Field(description="Right coordinate")
    y2: float = Field(description="Bottom coordinate")
    page_number: Optional[int] = Field(description="Page number", default=None)
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1


class TextPosition(BaseModel):
    """Simple text-based position using character indices"""
    start: int = Field(description="Start character index")
    end: int = Field(description="End character index")


class PDFPosition(BaseModel):
    """PDF-specific position with page and coordinates"""
    page_number: int = Field(description="Page number (0-indexed)")
    bounding_box: Optional[BoundingBox] = Field(description="Bounding box coordinates", default=None)
    text_start: Optional[int] = Field(description="Character start index within page", default=None)
    text_end: Optional[int] = Field(description="Character end index within page", default=None)


class DOCXPosition(BaseModel):
    """DOCX-specific position using paragraph and character indices"""
    paragraph_index: int = Field(description="Paragraph index")
    character_start: Optional[int] = Field(description="Character start within paragraph", default=None)
    character_end: Optional[int] = Field(description="Character end within paragraph", default=None)
    section_index: Optional[int] = Field(description="Section index", default=None)


class HTMLPosition(BaseModel):
    """HTML-specific position using DOM elements"""
    element_selector: str = Field(description="CSS selector or XPath to element")
    text_start: Optional[int] = Field(description="Character start within element", default=None)
    text_end: Optional[int] = Field(description="Character end within element", default=None)


class Positions(BaseModel):
    """Universal position class that can handle different document types"""
    document_type: DocumentType = Field(description="Type of document")
    
    # Different position types - only one should be set based on document_type
    pdf_position: Optional[PDFPosition] = Field(description="PDF-specific position", default=None)
    docx_position: Optional[DOCXPosition] = Field(description="DOCX-specific position", default=None)
    html_position: Optional[HTMLPosition] = Field(description="HTML-specific position", default=None)


    # this must be included no matter what type of document it is
    text_position: Optional[TextPosition] = Field(description="Text-based position", default=None)
    
    # Common metadata
    metadata: Dict[str, Any] = Field(description="Additional position metadata", default_factory=dict)
    
    @classmethod
    def from_text(cls, start: int, end: int, **metadata) -> "Positions":
        """Create a text position"""
        return cls(
            document_type=DocumentType.TEXT,
            text_position=TextPosition(start=start, end=end),
            metadata=metadata
        )
    
    @classmethod
    def from_pdf(cls, page_number: int, bounding_box: Optional[BoundingBox] = None, 
                 text_start: Optional[int] = None, text_end: Optional[int] = None, **metadata) -> "Positions":
        """Create a PDF position"""
        return cls(
            document_type=DocumentType.PDF,
            pdf_position=PDFPosition(
                page_number=page_number,
                bounding_box=bounding_box,
                text_start=text_start,
                text_end=text_end
            ),
            metadata=metadata
        )
    
    @classmethod
    def from_docx(cls, paragraph_index: int, character_start: Optional[int] = None,
                  character_end: Optional[int] = None, section_index: Optional[int] = None, **metadata) -> "Positions":
        """Create a DOCX position"""
        return cls(
            document_type=DocumentType.DOCX,
            docx_position=DOCXPosition(
                paragraph_index=paragraph_index,
                character_start=character_start,
                character_end=character_end,
                section_index=section_index
            ),
            metadata=metadata
        )
    
    @classmethod
    def from_html(cls, element_selector: str, text_start: Optional[int] = None,
                  text_end: Optional[int] = None, **metadata) -> "Positions":
        """Create an HTML position"""
        return cls(
            document_type=DocumentType.HTML,
            html_position=HTMLPosition(
                element_selector=element_selector,
                text_start=text_start,
                text_end=text_end
            ),
            metadata=metadata
        )



class Section(BaseModel):
    title: str = Field(description="The title of the section", default="")
    content: str = Field(description="The content of the section", default="")
    level: int = Field(description="The level of the section", default=0)
    
    # Enhanced position fields using the new Position system
    title_position: Positions = Field(description="Position of the section title", default=Positions.from_text(-1, -1))
    content_position: Positions = Field(description="Position of the section content", default=Positions.from_text(-1, -1))

    sub_sections: List[Self] = Field(description="The sub sections of the section", default=[])
    parent_section: Optional[Self] = Field(description="The parent section of the section", default=None)

    title_parsed: str = Field(description="The parsed title of the section", default="")
    content_parsed: str = Field(description="The parsed content of the section", default="")


def get_section_tree(section: Self) -> Self:
    """
    Get the section tree of the section
    """
    return section
