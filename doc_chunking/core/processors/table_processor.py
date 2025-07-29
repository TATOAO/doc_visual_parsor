
import json
from typing import List
from processor_pipeline import AsyncProcessor
from doc_chunking.schemas.layout_schemas import LayoutElement, TableElement, TableCell, BoundingBox, StyleInfo, FontInfo, RunInfo, ElementType
from doc_chunking.schemas import Section
from doc_chunking.utils.logging_config import get_logger
from typing import AsyncGenerator, Any
import pdfplumber

logger = get_logger(__name__)

class TableProcessor(AsyncProcessor):
    meta = {
        "name": "TableProcessor",
        "input_type": List[LayoutElement],
        "output_type": List[LayoutElement],
    }

    def _char_in_bbox(self, char: dict, bbox: tuple) -> bool:
        """Check if a character is within a bounding box."""
        v_mid = (char["top"] + char["bottom"]) / 2
        h_mid = (char["x0"] + char["x1"]) / 2
        x0, top, x1, bottom = bbox
        return bool(
            (h_mid >= x0) and (h_mid < x1) and (v_mid >= top) and (v_mid < bottom)
        )

    def _extract_cell_text_and_style(self, cell_bbox: tuple, chars: List[dict]) -> tuple[str, StyleInfo]:
        """Extract text and style information from a cell."""
        # Filter characters that are in this cell
        cell_chars = [char for char in chars if self._char_in_bbox(char, cell_bbox)]
        
        if not cell_chars:
            return "", StyleInfo(runs=[])
        
        # Sort characters by reading order (top to bottom, left to right)
        cell_chars.sort(key=lambda c: (c["top"], c["x0"]))
        
        # Extract text
        text = "".join(char["text"] for char in cell_chars)
        
        # Extract style information
        runs = []
        current_run = None
        
        for char in cell_chars:
            # Create font info for this character
            font_info = FontInfo(
                name=char.get("fontname", "Unknown"),
                size=char.get("size", 0.0),
                bold=char.get("fontname", "").lower().find("bold") != -1,
                italic=char.get("fontname", "").lower().find("italic") != -1,
                underline=False,  # PDF doesn't easily provide this info
                color=char.get("non_stroking_color", "#000000")
            )
            
            # Check if this character has the same style as the current run
            if (current_run is None or 
                current_run.font.name != font_info.name or
                current_run.font.size != font_info.size or
                current_run.font.bold != font_info.bold or
                current_run.font.italic != font_info.italic):
                
                # Start a new run
                if current_run is not None:
                    runs.append(current_run)
                
                current_run = RunInfo(
                    text=char["text"],
                    font=font_info,
                    start_index=len("".join(r.text for r in runs)),
                    end_index=len("".join(r.text for r in runs)) + len(char["text"])
                )
            else:
                # Extend current run
                current_run.text += char["text"]
                current_run.end_index = len("".join(r.text for r in runs)) + len(current_run.text)
        
        # Add the last run
        if current_run is not None:
            runs.append(current_run)
        
        # Create style info
        style_info = StyleInfo(
            runs=runs,
            primary_font=runs[0].font if runs else FontInfo(name="Unknown", size=0.0)
        )
        
        return text, style_info

    def _reconstruct_table_into_table_element(self, table: pdfplumber.Table) -> TableElement:
        """Convert a pdfplumber Table into a TableElement."""
        chars = table.page.chars
        
        # Create table bounding box
        table_bbox = BoundingBox(
            x1=table.bbox[0],
            y1=table.bbox[1],
            x2=table.bbox[2],
            y2=table.bbox[3]
        )
        
        # Process all cells
        all_cells = []
        table_rows = []
        
        for row in table.rows:
            row_cells = []
            for cell_bbox in row.cells:
                if cell_bbox is None:
                    # Create empty cell
                    empty_cell = TableCell(
                        bbox=BoundingBox(x1=0, y1=0, x2=0, y2=0),
                        text="",
                        style=StyleInfo(runs=[])
                    )
                    row_cells.append(empty_cell)
                else:
                    # Extract text and style for this cell
                    cell_text, cell_style = self._extract_cell_text_and_style(cell_bbox, chars)
                    
                    # Create cell bounding box
                    cell_bbox_obj = BoundingBox(
                        x1=cell_bbox[0],
                        y1=cell_bbox[1],
                        x2=cell_bbox[2],
                        y2=cell_bbox[3]
                    )
                    
                    # Create table cell
                    table_cell = TableCell(
                        bbox=cell_bbox_obj,
                        text=cell_text,
                        style=cell_style
                    )
                    
                    row_cells.append(table_cell)
                    all_cells.append(table_cell)
            
            table_rows.append(row_cells)
        
        # Create table element
        table_element = TableElement(
            bbox=table_bbox,
            cells=all_cells,
            rows=table_rows,
            metadata=None  # Will be set by the calling function
        )
        
        return table_element


    async def process(self, layout_list_generator: AsyncGenerator[List[LayoutElement], None]) -> AsyncGenerator[List[LayoutElement], None]:
        """
        Process layout elements and convert table elements to proper TableElement objects.

        depends: 
            PageImageLayoutProcessor():
                self.session['plumber_pages'] = plumber_pages
                self.session['plumber_page_map'] = { f'plumber_page_{i}': page for i, page in enumerate(plumber_pages.pages) }
        """
        async for page_layout_list in layout_list_generator:
            if not page_layout_list:
                yield page_layout_list
                continue
                
            page_num = page_layout_list[0].metadata.page_number - 1
            
            # Get the corresponding pdfplumber page
            plumber_page = self.session['plumber_page_map'][f'plumber_page_{page_num}']

            # Find all tables in the page
            table_list = plumber_page.find_tables()

            for element in page_layout_list:
                if element.element_type == ElementType.TABLE:
                    # match the table with the element
                    for table in table_list:
                        # Convert table.bbox tuple to BoundingBox object
                        table_bbox = BoundingBox(
                            x1=table.bbox[0],
                            y1=table.bbox[1],
                            x2=table.bbox[2],
                            y2=table.bbox[3]
                        )
                        if element.bbox.contains(table_bbox):
                            table_element = self._reconstruct_table_into_table_element(table)
                            element.metadata.table_element = table_element
                            continue
                            
                    

            yield page_layout_list


# python -m doc_chunking.core.processors.table_processor
if __name__ == "__main__":
    from processor_pipeline import AsyncPipeline
    from doc_chunking.core.processors.page_chunker import PdfPageImageSplitterProcessor
    from doc_chunking.core.processors.page_image_layout_processor import PageImageLayoutProcessor
    import asyncio
    async def main():
        pipeline = AsyncPipeline([
            PdfPageImageSplitterProcessor(), 
            PageImageLayoutProcessor(),
            TableProcessor()
        ])
        async for result in pipeline.astream('tests/test_data/table_test.pdf'):
            print(result)

    asyncio.run(main())