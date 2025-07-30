import asyncio
from typing import Any, List, AsyncGenerator, Tuple
from processor_pipeline import AsyncProcessor
from .page_chunker import PdfPageImageSplitterProcessor
from .page_image_layout_processor import PageImageLayoutProcessor
from doc_chunking.schemas.layout_schemas import LayoutElement
from doc_chunking.layout_structuring.title_structure_builder_llm.layout_displayer import DisplayLine
from doc_chunking.utils.logging_config import get_logger

logger = get_logger(__name__)

class BboxNLPProcessor(AsyncProcessor):
    meta = {
        "name": "BboxNLPProcessor",
        "input_type": List[LayoutElement],
        "output_type": Tuple[str, LayoutElement],
    }

    async def process(self, input_data: AsyncGenerator[List[LayoutElement], None]) -> AsyncGenerator[Tuple[str, LayoutElement], None]:
        index = 0
        async for elements in input_data:
            logger.debug(f"BboxNLPProcessor processed element {index}")
            for element in elements:
                yield str(DisplayLine.from_layout_element(element)), element
                index += 1
            await asyncio.sleep(0.001)
            

# python -m doc_chunking.core.processors.bbox_nlp_processor
if __name__ == "__main__":
    from processor_pipeline import AsyncPipeline
    async def main():
        pipeline = AsyncPipeline([
            PdfPageImageSplitterProcessor(), 
            PageImageLayoutProcessor(), 
            BboxNLPProcessor()
            ]
        )
        async for nlp, laytout_element in pipeline.astream(input_data='/Users/tatoaoliang/Downloads/Work/doc_chunking/tests/test_data/1-1 买卖合同（通用版）.pdf'):
            print(nlp)

        # result = await pipeline.run('/Users/tatoaoliang/Downloads/Work/doc_chunking/tests/test_data/1-1 买卖合同（通用版）.pdf')
        # print(result)
        # return result


    import asyncio
    asyncio.run(main())