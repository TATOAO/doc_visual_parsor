"""
Examples of using the new Position system for different document types
"""

from models.schemas.schemas import Position, BoundingBox, Section

# Example 1: Text position (simple case)
text_position = Position.from_text(
    start=100, 
    end=200,
    confidence=0.95
)
print("Text position:", text_position)

# Example 2: PDF position with bounding box
pdf_bbox = BoundingBox(x1=50.0, y1=100.0, x2=500.0, y2=150.0, page_number=1)
pdf_position = Position.from_pdf(
    page_number=1,
    bounding_box=pdf_bbox,
    text_start=0,
    text_end=50,
    font_size=14,
    is_heading=True
)
print("PDF position:", pdf_position)

# Example 3: DOCX position
docx_position = Position.from_docx(
    paragraph_index=5,
    character_start=10,
    character_end=45,
    section_index=2,
    style="Heading 1"
)
print("DOCX position:", docx_position)

# Example 4: HTML position
html_position = Position.from_html(
    element_selector="div.content > h2:nth-child(3)",
    text_start=0,
    text_end=25,
    tag_name="h2"
)
print("HTML position:", html_position)

# Example 5: Using positions in Section
section = Section(
    title="Chapter 1: Introduction",
    content="This is the introductory chapter...",
    level=1,
    title_position=Position.from_pdf(
        page_number=0,
        bounding_box=BoundingBox(x1=72.0, y1=100.0, x2=400.0, y2=130.0),
        font_size=18,
        is_title=True
    ),
    content_position=Position.from_pdf(
        page_number=0,
        bounding_box=BoundingBox(x1=72.0, y1=140.0, x2=500.0, y2=400.0),
        font_size=12
    )
)
print("Section with positions:", section)

# Example 6: Backward compatibility with legacy PositionIndex
from models.schemas.schemas import PositionIndex

legacy_pos = PositionIndex(start=100, end=200)
new_pos = legacy_pos.to_position()
print("Converted legacy position:", new_pos) 