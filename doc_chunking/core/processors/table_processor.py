
import json
from typing import List
from processor_pipeline import AsyncProcessor
from doc_chunking.schemas.layout_schemas import LayoutElement
from doc_chunking.schemas import Section
from doc_chunking.utils.logging_config import get_logger
from typing import AsyncGenerator, Any

logger = get_logger(__name__)

class TableProcessor(AsyncProcessor):
    meta = {
        "name": "TableProcessor",
        "input_type": List[LayoutElement],
        "output_type": List[LayoutElement],
    }

    async def process(self, layout_list_generator: AsyncGenerator[List[LayoutElement], None]) -> AsyncGenerator[List[LayoutElement], None]:
        async for page_layout_list in layout_list_generator:
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