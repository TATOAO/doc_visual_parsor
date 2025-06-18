from typing import List, Optional
from models.schemas.layout_schemas import LayoutExtractionResult, LayoutElement, ElementType
from pydantic import BaseModel


def display_layout(layout_extraction_result: LayoutExtractionResult, 
                   exclude_types: List[ElementType] = []):
    """
    Display the layout in a human-readable format.
    Using special tags to indicate the element type, position, style, text, etc.
    
    Args:
        layout_extraction_result: The layout extraction result.
        
    Returns:
        A list of strings, each representing a line of the layout.
    
    """
    display_result = []

    for element in layout_extraction_result.elements:
        if element.element_type in exclude_types:
            continue
        line = str(DisplayLine.from_layout_element(element))
        display_result.append(line)


    return display_result

class DisplayLine(BaseModel):
    page_number: Optional[int] = None
    element_type: str
    element_id: int
    element_text: str
    element_bbox: str
    font_name: str
    font_size: float
    font_color: Optional[str] = None
    font_italic: Optional[bool] = None
    font_underline: Optional[bool] = None
    font_bold: Optional[bool] = None
    alignment: Optional[str] = None

    @classmethod
    def from_layout_element(cls, layout_element: LayoutElement) -> "DisplayLine":
        """
        example input: LayoutElement(id=0, element_type=<ElementType.TEXT: 'Text'>, confidence=0.9144884347915649, bbox=BoundingBox(x1=223.87587890625, y1=108.01868408203124, x2=370.85455078125, y2=129.90099609375, width=146.978671875, height=21.882312011718753, center=(297.36521484375, 118.95984008789063), area=3216.233157036782), text='第一章买卖合同', style=StyleInfo(style_name='SimSun-18.0pt', style_type='text_span', builtin=None, paragraph_format=ParagraphFormat(alignment=<TextAlignment.LEFT: 'left'>, left_indent=None, right_indent=None, first_line_indent=None, space_before=None, space_after=None, line_spacing=None, line_spacing_rule=None, keep_together=None, keep_with_next=None, page_break_before=None, widow_control=None), runs=[RunInfo(text='第一章', font=FontInfo(name='SimSun', size=18.0, bold=False, italic=False, underline=False, color='#000000', highlight=None, strikethrough=None, superscript=None, subscript=None), start_index=None, end_index=None, text_length=3), RunInfo(text='买卖合同', font=FontInfo(name='SimSun', size=18.0, bold=False, italic=False, underline=False, color='#000000', highlight=None, strikethrough=None, superscript=None, subscript=None), start_index=None, end_index=None, text_length=4)], primary_font=FontInfo(name='SimSun', size=18.0, bold=False, italic=False, underline=False, color='#000000', highlight=None, strikethrough=None, superscript=None, subscript=None), table_style=None, list_style=None, custom_properties=None, has_formatting=True, run_count=2), metadata={'model_class_id': 0, 'detection_method': 'cv_yolo', 'source_type': 'pdf_page', 'page_number': 1, 'scale_factor': 0.48, 'source_pdf_elements': [0, 1], 'enrichment_method': 'cv_first_pdf_enriched', 'runs_count': 2}, has_text=True, has_bbox=True, has_style=True, text_length=7)
        
        """

        bbox = layout_element.bbox
        bbox_str = f"[{int(bbox.x1)},{int(bbox.y1)},{int(bbox.x2)},{int(bbox.y2)}]"


        return cls(
            page_number=layout_element.metadata.get('page_number', None),
            element_type=layout_element.element_type,
            element_id=layout_element.id,
            element_text=layout_element.text,
            element_bbox=bbox_str,
            font_name=layout_element.style.runs[0].font.name if layout_element.style.runs else None,
            font_size=round(layout_element.style.runs[0].font.size, 1) if layout_element.style.runs else None,
            font_color=layout_element.style.runs[0].font.color if layout_element.style.runs[0].font.color != '#000000' else None,
            font_italic=layout_element.style.runs[0].font.italic if layout_element.style.runs else None,
            font_underline=layout_element.style.runs[0].font.underline,
            font_bold=layout_element.style.runs[0].font.bold if layout_element.style.runs else None,
            alignment=layout_element.style.paragraph_format.alignment.value if layout_element.style.paragraph_format else None
        )

    
    def __str__(self) -> str:
        return f"[id:{self.element_id}]" + \
            (f"[page:{self.page_number}]" if self.page_number else "") + \
            f"[type:{self.element_type}]" + \
            (f"[pos:{self.element_bbox}]" if False else "") + \
            f"[{self.font_name} {self.font_size}pt]" + \
            (f"[color:{self.font_color}]" if self.font_color else "") + \
            (f"[italic:{self.font_italic}]" if self.font_italic else "") + \
            (f"[bold:{self.font_bold}]" if self.font_bold else "") + \
            (f"[underline:{self.font_underline}]" if self.font_underline else "") + \
            (f"[alignment:{self.alignment}]" if self.alignment else "") + \
            f"{self.element_text}"



# python -m models.layout_structuring.title_structure_builder_llm.layout_displayer
if __name__ == "__main__":
    import json
    with open("./layout_detection_result_with_runs.json", "r") as f:
        layout_data = json.load(f)
    layout_extraction_result = LayoutExtractionResult(
        elements=layout_data["elements"], 
        metadata=layout_data["metadata"]
    )

    display_result = display_layout(layout_extraction_result, exclude_types=[])

    with open("./layout_displayer.txt", "w") as f:
        f.write("\n".join(display_result))
