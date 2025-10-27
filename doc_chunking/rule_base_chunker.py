from doc_chunking.schemas import LayoutExtractionResult, ElementType, LayoutElement
from typing import List
from doc_chunking.schemas import Section


def _extract_page_number(element: LayoutElement) -> int:
    """Extract page number from element metadata."""
    if element.metadata and 'page_number' in element.metadata:
        return element.metadata['page_number']
    return 0  # Default page number if not found


def simple_chunking(layout_extraction_result: LayoutExtractionResult) -> List[Section]:
    """
    Simple chunking function that:
    - Skips all "ABANDON" type elements
    - Connects all "broken" text (PLAIN_TEXT, PARAGRAPH)
    - Uses only "TITLE" elements as separators
    - Combines continuous titles into the next title
    - Tracks page numbers for each section
    - Returns a plain list of sections
    """
    sections = []
    current_content = []
    current_title = ""
    current_level = 0
    current_element_id = 0
    current_page_numbers = set()  # Track page numbers for current section
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
                    page_number=list(current_page_numbers),
                    parent_section=None,
                    sub_sections=[]
                )
                sections.append(section)
                current_content = []
                current_page_numbers = set()  # Reset page numbers for new section
            
            # Add this title to pending titles
            if element.text and element.text.strip():
                pending_titles.append(element.text.strip())
            
            # Update current section info
            current_level = getattr(element, 'level', 0)
            current_element_id = element.id
            # Add page number for the title
            current_page_numbers.add(_extract_page_number(element))
            
        # Accumulate text content for current section
        elif element.element_type in [ElementType.PLAIN_TEXT, ElementType.PARAGRAPH]:
            if element.text and element.text.strip():
                # If we have pending titles, combine them with the first content
                if pending_titles:
                    current_title = " ".join(pending_titles)
                    pending_titles = []
                
                current_content.append(element.text.strip())
                # Add page number for the text content
                current_page_numbers.add(_extract_page_number(element))
        
        # Figure
        elif element.element_type == ElementType.FIGURE:
            if element.text and element.text.strip():
                # If we have pending titles, combine them with the first content
                if pending_titles:
                    current_title = " ".join(pending_titles)
                    pending_titles = []
                
                current_content.append(element.text.strip())
                # Add page number for the figure
                current_page_numbers.add(_extract_page_number(element))
        
        # Table
        elif element.element_type == ElementType.TABLE:
            if element.text and element.text.strip():
                # If we have pending titles, combine them with the first content
                if pending_titles:
                    current_title = " ".join(pending_titles)
                    pending_titles = []
                
                current_content.append(element.text.strip())
                # Add page number for the table
                current_page_numbers.add(_extract_page_number(element))
            # todo parsing table content
        
        # List
        elif element.element_type == ElementType.LIST:
            if element.text and element.text.strip():
                # If we have pending titles, combine them with the first content
                if pending_titles:
                    current_title = " ".join(pending_titles)
                    pending_titles = []
                
                current_content.append(element.text.strip())
                # Add page number for the list
                current_page_numbers.add(_extract_page_number(element))
        
        # Equation
        elif element.element_type == ElementType.ISOLATE_FORMULA:
            if element.text and element.text.strip():
                # If we have pending titles, combine them with the first content
                if pending_titles:
                    current_title = " ".join(pending_titles)
                    pending_titles = []
                
                current_content.append(element.text.strip())
                # Add page number for the equation
                current_page_numbers.add(_extract_page_number(element))

        # Handle any other element type that has text
        else:
            if element.text and element.text.strip():
                # If we have pending titles, combine them with the first content
                if pending_titles:
                    current_title = " ".join(pending_titles)
                    pending_titles = []
                
                current_content.append(element.text.strip())
                # Add page number for the other element
                current_page_numbers.add(_extract_page_number(element))

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
            page_number=list(current_page_numbers),
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
        print(f"title: {section.title}, content: {section.content}, page_numbers: {section.page_number}")