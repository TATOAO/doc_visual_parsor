from pydantic import BaseModel, Field, computed_field
from typing import List, Optional, Dict, Any, Union
from typing_extensions import Self
from pathlib import Path
from enum import Enum
import hashlib
from doc_chunking.schemas.layout_schemas import BoundingBox

import numpy as np
from PIL import Image
from typing import BinaryIO


class DocumentType(str, Enum):
    """Document type enumeration"""
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    TEXT = "text"

InputDataType = Union[
    str,                    # File path as string
    Path,                   # File path as Path object
    bytes,                  # Raw file content
    np.ndarray,            # Image as numpy array
    Image.Image,           # PIL Image object
    BinaryIO,              # File-like object with read() method
    Any                    # For objects with getvalue() method (uploaded files)
]



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
    """
    DOCX-specific position using paragraph and character indices

    DOCX files (unlike PDFs) are flow-based, not fixed-layout. So no page number.
    """
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
    element_id: int = Field(description="The element id of the section", default=None)

    sub_sections: List[Self] = Field(description="The sub sections of the section", default=[])
    parent_section: Optional[Self] = Field(description="The parent section of the section", default=None)

    @computed_field
    @property
    def section_hash(self) -> str:
        """
        Get the hash of the section based on title_parsed and content_parsed only
        """
        # Combine title_parsed and content_parsed for hashing
        combined_content = f"{self.title}|{self.content}"
        
        # Generate hash from the combined content
        return hashlib.sha256(combined_content.encode('utf-8')).hexdigest()


def get_section_tree(section: Self) -> Self:
    """
    Get the section tree of the section
    """
    return section



# python -m models.utils.schemas
if __name__ == "__main__":
    section = Section(title="test", content="test")
    print('test0', section.section_hash)

    section.title_parsed = "test"
    section.content_parsed = "test"
    print('test1', section.section_hash)

    section.title_parsed = "test2"
    section.content_parsed = "test2"
    print('test2', section.section_hash)
