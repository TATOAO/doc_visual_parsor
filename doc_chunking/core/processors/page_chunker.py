from processor_pipeline import AsyncProcessor
from typing import Any, AsyncGenerator, Union, List, Tuple
from doc_chunking.schemas.files import FileInputData
from doc_chunking.layout_detection.layout_extraction.pdf_layout_extractor import PdfLayoutExtractor
from doc_chunking.schemas.layout_schemas import LayoutElement
import asyncio
from PIL import Image
import fitz
import io
import fitz  # PyMuPDF

class PdfPageImageSplitter(AsyncProcessor):
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
            file = open(item, 'rb')

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
            mat = fitz.Matrix(zoom, zoom)
            
            # Render page as image
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to numpy array
            img = Image.open(io.BytesIO(img_data))

            # return np.array(img), 1.0 / zoom  # Return image and inverse scale factor
            ###### Step 2:  Extract the layout ######
            page_layout = extractor._extract_raw_pdf_structure_from_page(page=page, page_number=page_num + 1, element_start_id=0)
            scale_factor = 1.0 / zoom
            # Scale coordinates back to PDF space
            for element in page_layout:
                if element.bbox:
                    element.bbox.x1 *= scale_factor
                    element.bbox.y1 *= scale_factor
                    element.bbox.x2 *= scale_factor
                    element.bbox.y2 *= scale_factor

            import time
            print('......', time.time())
            await asyncio.sleep(0.01)
            yield img, page_layout


# python -m doc_chunking.core.processors.page_chunker
if __name__ == "__main__":
    from processor_pipeline import AsyncPipeline
    async def main():
        pipeline = AsyncPipeline([PdfPageImageSplitter()])
        result = await pipeline.run('/Users/tatoaoliang/Downloads/Work/doc_chunking/tests/test_data/1-1 买卖合同（通用版）.pdf')
        # print(result)

    import asyncio
    asyncio.run(main())