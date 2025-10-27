from doc_chunking.schemas import LayoutExtractionResult, ElementType
from pydantic import BaseModel, Field, computed_field
from typing import List, Optional, Self
import hashlib


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



def simple_chunking(layout_extraction_result: LayoutExtractionResult) -> List[Section]:
    """
    Simple chunking function that:
    - Skips all "ABANDON" type elements
    - Connects all "broken" text (PLAIN_TEXT, PARAGRAPH)
    - Uses only "TITLE" elements as separators
    - Combines continuous titles into the next title
    - Returns a plain list of sections
    """
    sections = []
    current_content = []
    current_title = ""
    current_level = 0
    current_element_id = 0
    pending_titles = []  # Store continuous titles
    
    for element in layout_extraction_result.elements:
        print(element)
        # Skip ABANDON elements
        if element.element_type == ElementType.ABANDON:
            continue
            
        # If we encounter a TITLE, handle continuous titles
        if element.element_type == ElementType.TITLE:
            # If we have accumulated content, create a section first
            if current_content:
                content_text = " ".join(current_content)
                section = Section(
                    title=current_title,
                    content=content_text,
                    level=current_level,
                    element_id=current_element_id,
                    parent_section=None,
                    sub_sections=[]
                )
                sections.append(section)
                current_content = []
            
            # Add this title to pending titles
            if element.text and element.text.strip():
                pending_titles.append(element.text.strip())
            
            # Update current section info
            current_level = getattr(element, 'level', 0)
            current_element_id = element.id
            
        # Accumulate text content for current section
        elif element.element_type in [ElementType.PLAIN_TEXT, ElementType.PARAGRAPH]:
            if element.text and element.text.strip():
                # If we have pending titles, combine them with the first content
                if pending_titles:
                    current_title = " ".join(pending_titles)
                    pending_titles = []
                
                current_content.append(element.text.strip())
        
        # Figure
        elif element.element_type == ElementType.FIGURE:


            Section(title="", content=element.text, level=0, element_id=element.id, parent_section=None, sub_sections=[])
        # Table
        elif element.element_type == ElementType.TABLE:

            if element.text and element.text.strip():
                # If we have pending titles, combine them with the first content
                if pending_titles:
                    current_title = " ".join(pending_titles)
                    pending_titles = []
                
                current_content.append(element.text.strip())

            # todo parsing table content
            # Section(title="", content=element.text, level=0, element_id=element.id, parent_section=None, sub_sections=[])
        # List
        elif element.element_type == ElementType.LIST:
            Section(title="", content=element.text, level=0, element_id=element.id, parent_section=None, sub_sections=[])
        # Equation
        elif element.element_type == ElementType.ISOLATE_FORMULA:
            Section(title="", content=element.text, level=0, element_id=element.id, parent_section=None, sub_sections=[])

        # Handle any other element type that has text
        else:
            if element.text and element.text.strip():
                # If we have pending titles, combine them with the first content
                if pending_titles:
                    current_title = " ".join(pending_titles)
                    pending_titles = []
                
                current_content.append(element.text.strip())

    # Create final section if there's remaining content
    if current_content:
        content_text = " ".join(current_content)
        # Use pending titles if no content was processed
        if pending_titles and not current_title:
            current_title = " ".join(pending_titles)
        
        section = Section(
            title=current_title,
            content=content_text,
            level=current_level,
            element_id=current_element_id,
            parent_section=None,
            sub_sections=[]
        )
        sections.append(section)
    
    return sections


# python -m doc_chunking.rule_base_chunker
if __name__ == "__main__":
    from doc_chunking.schemas import LayoutExtractionResult, LayoutElement
    from doc_chunking.merging import PdfStyleCVMixLayoutExtractor

    extractor = PdfStyleCVMixLayoutExtractor(
        model_path="model_parameters/layout_detection/docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx",
        cv_confidence_threshold=0.1,  # Lower threshold for better detection
        cv_image_size=1024,
        cv_pdf_dpi=150,
        device="auto"
    )

    # layout_extraction_result = extractor.detect_layout("3800.pdf", max_pages=50)
    layout_extraction_result = extractor.detect_layout("trouble_handle_number_27th_page_of_3800.pdf", max_pages=50)
    # layout_extraction_result = LayoutExtractionResult(
    #     elements=[
    #         LayoutElement(element_type=ElementType.TITLE, text="Title", level=0, id=1),
    #         LayoutElement(element_type=ElementType.PLAIN_TEXT, text="Plain Text", level=0, id=2),
    #         LayoutElement(element_type=ElementType.PLAIN_TEXT, text="Plain Text", level=0, id=2),
    #         LayoutElement(element_type=ElementType.PLAIN_TEXT, text="Plain Text", level=0, id=2),
    #         LayoutElement(element_type=ElementType.TITLE, text="Title", level=0, id=1),
    #         LayoutElement(element_type=ElementType.TITLE, text="Title", level=0, id=1),
    #         LayoutElement(element_type=ElementType.TITLE, text="Title", level=0, id=1),
    #         LayoutElement(element_type=ElementType.PLAIN_TEXT, text="Plain Text", level=0, id=2),
    #         LayoutElement(element_type=ElementType.PLAIN_TEXT, text="Plain Text", level=0, id=2),
    #         LayoutElement(element_type=ElementType.TITLE, text="Title", level=0, id=1),
    #         LayoutElement(element_type=ElementType.PLAIN_TEXT, text="Plain Text", level=0, id=2),
    #     ]
    # )
    for section in simple_chunking(layout_extraction_result):
        print(f"title: {section.title}, content: {section.content}")