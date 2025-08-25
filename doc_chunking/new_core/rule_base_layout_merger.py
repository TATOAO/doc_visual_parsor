from processor_pipeline.core import AsyncProcessor
from typing import Any, AsyncGenerator, Union, List, Tuple
from doc_chunking.schemas.layout_schemas import LayoutElement, ElementType
from doc_chunking.schemas.schemas import FileLayoutElementCollection
from processor_pipeline.core.pipe import BufferPipe
from PIL import Image
from loguru import logger
import sys
from asyncio import Queue


class RuleBaseLayoutMerger(AsyncProcessor):
    meta = {
        "name": "RuleBaseLayoutMerger",
        "input_type": Tuple[Image, FileLayoutElementCollection],
        "output_type": FileLayoutElementCollection,
        "output_strategy": "ordered"
    }


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.core_pipe = BufferPipe(maxsize=10)
    
    async def check_if_break(self, layout_element: LayoutElement, message_id: str):
        pass


    async def process(self, input_data: FileLayoutElementCollection, message_id = None, *args, **kwargs) -> FileLayoutElementCollection:
        """
        Input layout with continues 3 elements
        """

        layout_elements = input_data.elements

        for layout_element in layout_elements:
            if layout_element.element_type == ElementType.ABANDON:
                continue
            
            if layout_element.element_type == ElementType.TEXT:
                await self.core_pipe.put(layout_element)
            

            can_break = await self.check_if_break()
            if can_break:
                yield self.core_pipe.get()
                










if __name__ == "__main__":
    from processor_pipeline import GraphBase
    from doc_chunking.new_core.page_chunker import PdfPageImageSplitterProcessor
    from doc_chunking.new_core.page_image_layout_processor import PageImageLayoutProcessor
    from processor_pipeline.core.graph import Node, Edge
    async def main():
        pipeline = GraphBase(nodes=[
            Node(processor_class_name="PdfPageImageSplitterProcessor",
                processor_unique_name="pdf_page_image_splitter_processor",
            ),
            Node(processor_class_name="PageImageLayoutProcessor",
                processor_unique_name="page_image_layout_processor",
            ),
            Node(processor_class_name="RuleBaseLayoutMerger",
                processor_unique_name="rule_base_layout_merger",
            )
        ], edges=[
            Edge(
                source_node_pipe_id="pdf_page_image_splitter_processor", 
                target_node_pipe_id="page_image_layout_processor", 
                edge_unique_name="page_image_layout_processor_edge"
            ),
            Edge(
                source_node_pipe_id="page_image_layout_processor", 
                target_node_pipe_id="rule_base_layout_merger", 
                edge_unique_name="rule_base_layout_merger_edge"
            )
        ])

        input_data = "/Users/tatoaoliang/Downloads/Work/doc_chunking/tests/test_data/1-1 买卖合同（通用版）.pdf"
        await pipeline.execute(input_data=input_data)

    import asyncio
    asyncio.run(main())