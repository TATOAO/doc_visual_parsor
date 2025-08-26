import asyncio
from PIL import Image
from processor_pipeline.new_core import AsyncProcessor
from typing import Any, AsyncGenerator, Union, List, Tuple
from doc_chunking.schemas.layout_schemas import LayoutElement
from doc_chunking.schemas.schemas import FileLayoutElementCollection
from PIL import Image
from doc_chunking.new_core.page_chunker import PdfPageImageSplitterProcessor
from doc_chunking.layout_detection.visual_detection.cv_detector import CVLayoutDetector
from doc_chunking.layout_detection.layout_extraction.pdf_style_cv_mix_extractor import PdfStyleCVMixLayoutExtractor
from doc_chunking.utils.logging_config import get_logger
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="INFO")

class PageImageLayoutProcessor(AsyncProcessor):
    meta = {
        "name": "PageImageLayoutProcessor",
        "input_type": Tuple[Image, List[LayoutElement]],
        "output_type": List[LayoutElement],
        "output_strategy": "ordered",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detector = CVLayoutDetector()
        self.detector._initialize_detector()

        self.merger = PdfStyleCVMixLayoutExtractor(need_initialize=False)

    async def process(self, input_data: Tuple[Image, FileLayoutElementCollection], *args, **kwargs) -> FileLayoutElementCollection:
        img, file_layout_element_collection = input_data
        # List[LayoutElement]]
        layout = file_layout_element_collection.elements

        # detect layout
        layout_result = self.detector._detect_layout(input_data=img)

        # merge layout
        enriched_layout = self.merger._enrich_cv_elements_with_pdf(cv_elements=layout_result.elements, pdf_elements=layout)

        file_layout_element_collection.elements = enriched_layout

        return file_layout_element_collection


# python -m doc_chunking.core.processors.page_image_layout_processor
if __name__ == "__main__":
    from processor_pipeline import AsyncPipeline
    async def main():
        pipeline = AsyncPipeline([
            PdfPageImageSplitterProcessor(), 
            PageImageLayoutProcessor()])
        # result = await pipeline.run('/Users/tatoaoliang/Downloads/Work/doc_chunking/tests/test_data/1-1 买卖合同（通用版）.pdf')
        input_data = '/Users/tatoaoliang/Downloads/Work/doc_chunking/tests/test_data/1-1 买卖合同（通用版）.pdf'
        import time
        # async for item in pipeline.astream(input_data=input_data):
        #     start_time = time.time()
        #     print(item)
        #     for layout in item:
        #         result.append(layout.model_dump(mode='json'))
        #     end_time = time.time()
        #     print(f"Time taken: {end_time - start_time} seconds")

        # import json
        # with open('result.json', 'w') as f:
        #     json.dump(result, f, ensure_ascii=False, indent=4)

        result = await pipeline.run(input_data=input_data)

        import json
        with open('result_run.json', 'w') as f:
            json_result = []
            for layouts in result:
                for l in layouts:
                    json_result.append(l.model_dump(mode='json'))
            json.dump(json_result, f, ensure_ascii=False, indent=4)

        return result


    
    import asyncio
    asyncio.run(main())