import asyncio
from typing import Any, List, AsyncGenerator, Tuple
from processor_pipeline.core import AsyncProcessor
from doc_chunking.core.page_chunker import PdfPageImageSplitterProcessor
from doc_chunking.core.page_image_layout_processor import PageImageLayoutProcessor
from doc_chunking.schemas.layout_schemas import LayoutElement
from doc_chunking.layout_structuring.title_structure_builder_llm.layout_displayer import DisplayLine
from doc_chunking.utils.logging_config import get_logger
from loguru import logger
import sys

logger.remove()
logger.add('logs/bbox_nlp_processor.log', level="DEBUG")

class BboxNLPProcessor(AsyncProcessor):
    meta = {
        "name": "BboxNLPProcessor",
        "input_type": List[LayoutElement],
        "output_type": Tuple[str, LayoutElement],
        "output_strategy": "ordered",
    }

    async def process(self, element: LayoutElement, *args, **kwargs) -> Tuple[str, LayoutElement]:
        yield (str(DisplayLine.from_layout_element(element)), element)
            

# python -m doc_chunking.new_core.bbox_nlp_processor
if __name__ == "__main__":
    from processor_pipeline.core import GraphBase
    from processor_pipeline.core.graph_model import Node, Edge
    async def main():
        graph = GraphBase(
            nodes=[
                Node(
                    processor_class_name="PdfPageImageSplitterProcessor",
                    processor_unique_name="PdfPageImageSplitterProcessor_1",
                ),
                Node(
                    processor_class_name="PageImageLayoutProcessor",
                    processor_unique_name="PageImageLayoutProcessor_1",
                ),
                Node(
                    processor_class_name="BboxNLPProcessor",
                    processor_unique_name="BboxNLPProcessor_1",
                ),
            ],
            edges=[
                Edge(
                    source_node_unique_name="PdfPageImageSplitterProcessor_1",
                    target_node_unique_name="PageImageLayoutProcessor_1",
                    edge_unique_name="PdfPageImageSplitterProcessor_1_to_PageImageLayoutProcessor_1",
                ),
                Edge(
                    source_node_unique_name="PageImageLayoutProcessor_1",
                    target_node_unique_name="BboxNLPProcessor_1",
                    edge_unique_name="PageImageLayoutProcessor_1_to_BboxNLPProcessor_1",
                ),
            ],
        )


        await graph.initialize()

        async for nlp, laytout_element in graph.astream(data=['tests/test_data/1-1 买卖合同（通用版）.pdf']):
            print(nlp)

        # result = await pipeline.run('/Users/tatoaoliang/Downloads/Work/doc_chunking/tests/test_data/1-1 买卖合同（通用版）.pdf')
        # print(result)
        # return result


    import asyncio
    asyncio.run(main())