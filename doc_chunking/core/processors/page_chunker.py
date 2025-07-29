import pdfplumber
import asyncio
import fitz
import io
import fitz  # PyMuPDF
from processor_pipeline import AsyncProcessor
from pathlib import Path
from typing import Any, AsyncGenerator, Union, List, Tuple
from doc_chunking.schemas import FileInputData
from doc_chunking.layout_detection.layout_extraction.pdf_layout_extractor import PdfLayoutExtractor
from doc_chunking.schemas.layout_schemas import LayoutElement
from PIL import Image
from doc_chunking.utils.logging_config import get_logger

logger = get_logger(__name__)


class PdfPageImageSplitterProcessor(AsyncProcessor):
    meta = {
        "name": "PdfPageImageSplitter",
        "description": "Split a page into images.",
        "input_type": FileInputData,
        "output_type": Tuple[Image, List[LayoutElement]],
    }
    def __init__(self, **kwargs):
        self.pdf_dpi = 150
        super().__init__(**kwargs)

    async def process(self, input_data: FileInputData) -> AsyncGenerator[Tuple[Image, List[LayoutElement]], None]:
        """
        Chunk a page into sections.

        Args:
            input_data: The input data to process.

        Returns:
            An async generator that yields the chunked sections.
        """

        async for item in input_data:

            if isinstance(item, str):
                file = open(item, 'rb')
            elif isinstance(item, bytes):
                file = io.BytesIO(item)
            elif isinstance(item, Path):
                file = open(item, 'rb')
            else:
                raise ValueError(f"Unsupported file type: {type(item)}")

            file_content = file.read()
            
            class MockUploadedFile:
                def __init__(self, content, content_type, name):
                    self.content = content
                    self.type = content_type
                    self.name = name
                
                def getvalue(self):
                    return self.content

            mock_file = MockUploadedFile(file_content, 'application/pdf', "file.pdf")


            if hasattr(mock_file, 'getvalue'):
                pdf_content = mock_file.getvalue()
            elif hasattr(mock_file, 'read'):
                pdf_content = mock_file.read()
            else:
                pdf_content = mock_file

            plumber_pages = pdfplumber.open(io.BytesIO(pdf_content))
            self.session['plumber_pages'] = plumber_pages
            self.session['plumber_page_map'] = { f'plumber_page_{i}': page for i, page in enumerate(plumber_pages.pages) }

            # load the pdf
            pdf_doc = fitz.open(stream=pdf_content, filetype="pdf")

            extractor = PdfLayoutExtractor(merge_fragments=True)

            for page_num in range(len(pdf_doc)):

                # logger.debug(f"Processing page {page_num + 1}")
                
                # Get page
                page = pdf_doc[page_num]

                # Calculate zoom factor based on desired DPI
                # Default PDF DPI is 72, so zoom = desired_dpi / 72
                ###### Step 1: Convert to image with high resolution ######
                zoom = self.pdf_dpi / 72.0
                self.session['zoom_factor'] = zoom
                mat = fitz.Matrix(zoom, zoom)
                
                # Render page as image
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to numpy array
                img = Image.open(io.BytesIO(img_data))

                logger.info(f"Pymupdf processed page {page_num + 1} into image")

                # return np.array(img), 1.0 / zoom  # Return image and inverse scale factor
                ###### Step 2:  Extract the layout ######
                page_layout = extractor._extract_raw_pdf_structure_from_page(page=page, page_number=page_num + 1, element_start_id=0)

                logger.info(f"Pymupdf processed page {page_num + 1} into layout")
                scale_factor = zoom
                # Scale coordinates back to PDF space
                for element in page_layout:
                    if element.bbox:
                        element.bbox.x1 *= scale_factor
                        element.bbox.y1 *= scale_factor
                        element.bbox.x2 *= scale_factor
                        element.bbox.y2 *= scale_factor

                await asyncio.sleep(0.01)
                yield img, page_layout


# python -m doc_chunking.core.processors.page_chunker
if __name__ == "__main__":
    from processor_pipeline import AsyncPipeline
    async def main():
        pipeline = AsyncPipeline([PdfPageImageSplitterProcessor()])
        result = await pipeline.run('tests/test_data/1-1 买卖合同（通用版）.pdf')
        print(result)

    import asyncio
    asyncio.run(main())