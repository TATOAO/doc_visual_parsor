import asyncio
from PIL import Image
from processor_pipeline import AsyncProcessor
from typing import Any, AsyncGenerator, Union, List, Tuple
from doc_chunking.schemas import LayoutElement
from PIL import Image
from .page_chunker import PdfPageImageSplitterProcessor
from doc_chunking.onnx_layout_detector import ONNXDocLayoutYOLO
from doc_chunking.merging import PdfStyleCVMixLayoutExtractor
from doc_chunking.utils.logging_config import get_logger

logger = get_logger(__name__)

class PageImageLayoutProcessor(AsyncProcessor):
    meta = {
        "name": "PageImageLayoutProcessor",
        "input_type": Tuple[Image, List[LayoutElement]],
        "output_type": List[LayoutElement],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detector = ONNXDocLayoutYOLO(model_path='model_parameters/layout_detection/docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx')

        self.merger = PdfStyleCVMixLayoutExtractor(need_initialize=False)

    async def process(self, input_data: AsyncGenerator[Tuple[Image, List[LayoutElement]], None]) -> AsyncGenerator[List[LayoutElement], None]:

        index = 0
        async for item in input_data:
            img, layout = item

            # detect layout
            layout_result = self.detector.detect_layout(input_data=img)
            logger.info(f"ONNXDocLayoutYOLO detected layout {index}")

            # merge layout
            enriched_layout = self.merger.merge_layout(cv_elements=layout_result.elements, pdf_elements=layout)
            logger.info(f"PdfStyleCVMixLayoutExtractor merged layout {index}")

            await asyncio.sleep(0.001)

            yield enriched_layout


# python -m doc_chunking.core.processors.page_image_layout_processor
if __name__ == "__main__":
    from processor_pipeline import AsyncPipeline
    async def main():
        pipeline = AsyncPipeline([
            PdfPageImageSplitterProcessor(), 
            PageImageLayoutProcessor()])
        # result = await pipeline.run('./智能设备.pdf')
        # input_data = '/Users/tatoaoliang/Downloads/Work/doc_chunking/tests/test_data/1-1 买卖合同（通用版）.pdf'
        input_data = './智能设备.pdf'
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