import asyncio
from PIL import Image
from processor_pipeline import AsyncProcessor
from typing import Any, AsyncGenerator, Union, List, Tuple
from doc_chunking.schemas.layout_schemas import LayoutElement
from PIL import Image
from .page_chunker import PdfPageImageSplitter
from doc_chunking.layout_detection.visual_detection.cv_detector import CVLayoutDetector
from doc_chunking.layout_detection.layout_extraction.pdf_style_cv_mix_extractor import PdfStyleCVMixLayoutExtractor

class PageImageLayoutProcessor(AsyncProcessor):
    meta = {
        "name": "PageImageLayoutProcessor",
        "input_type": Tuple[Image, List[LayoutElement]],
        "output_type": List[LayoutElement],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detector = CVLayoutDetector()
        self.detector._initialize_detector()

        self.merger = PdfStyleCVMixLayoutExtractor(need_initialize=False)

    async def process(self, input_data: AsyncGenerator[Tuple[Image, List[LayoutElement]], None]) -> AsyncGenerator[List[LayoutElement], None]:

        async for item in input_data:
            img, layout = item

            # detect layout
            layout_result = self.detector._detect_layout(input_data=img)

            # merge layout
            enriched_layout = self.merger._enrich_cv_elements_with_pdf(cv_elements=layout_result.elements, pdf_elements=layout)

            await asyncio.sleep(0.001)

            yield enriched_layout


# python -m doc_chunking.core.processors.page_image_layout_processor
if __name__ == "__main__":
    from processor_pipeline import AsyncPipeline
    async def main():
        pipeline = AsyncPipeline([PdfPageImageSplitter(), PageImageLayoutProcessor()])
        # result = await pipeline.run('/Users/tatoaoliang/Downloads/Work/doc_chunking/tests/test_data/1-1 买卖合同（通用版）.pdf')
        input_data = '/Users/tatoaoliang/Downloads/Work/doc_chunking/tests/test_data/1-1 买卖合同（通用版）.pdf'
        import time
        async for item in pipeline.astream(input_data=input_data):
            start_time = time.time()
            print(item)
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds")

    
    import asyncio
    asyncio.run(main())