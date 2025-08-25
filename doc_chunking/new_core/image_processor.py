from processor_pipeline.new_core import AsyncProcessor
from doc_chunking.schemas.layout_schemas import LayoutElement, ElementType
from PIL import Image
from typing import AsyncGenerator, Tuple, List
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="DEBUG")


class FigureTableImageExtractor(AsyncProcessor):
    meta = {
        "name": "FigureTableImageExtractor",
        "description": "Extract figure and table image from pdf page with description.",
        "output_strategy": "asap"
    }
    def __init__(self, **kwargs):
        pass

    async def process(self, data:Tuple[Image, List[LayoutElement]], *args, **kwargs) -> AsyncGenerator[Tuple[Image, List[LayoutElement]], None]:
        """
        Extract image from pdf page. If the element is a figure, return the image. If the element is a table, return the cropped image.

        Args:
            data: The input data to process.

        Returns:
            An async generator that yields the extracted image.
        """
        image, layout = data
        for element in layout:
            if element.element_type == ElementType.FIGURE:
                image = image.crop(element.bbox.to_tuple())
                image.save(f"figure_{element.id}.png")

            elif element.element_type == ElementType.TABLE:
                image = image.crop(element.bbox.to_tuple())
                image.save(f"table_{element.id}.png")
            else:
                continue




if __name__ == "__main__":
    from doc_chunking.new_core.page_chunker import PdfPageImageSplitterProcessor
    from doc_chunking.new_core.page_image_layout_processor import PageImageLayoutProcessor


    from processor_pipeline.core.graph import Graph, Node


    async def main():
        nodes = [
            Node(
                processor_class_name=PdfPageImageSplitterProcessor().meta["name"],
                processor_unique_name=PdfPageImageSplitterProcessor().meta["name"]
            ),
            Node(
                processor_class_name=PageImageLayoutProcessor().meta["name"],
                processor_unique_name=PageImageLayoutProcessor().meta["name"]
            ),
            Node(
                processor_class_name=FigureTableImageExtractor().meta["name"],
                processor_unique_name=FigureTableImageExtractor().meta["name"]
            )
        ]

        edges = [
            Edge(
                source_node_unique_name=PdfPageImageSplitterProcessor().meta["name"],
                target_node_unique_name=PageImageLayoutProcessor().meta["name"]
            ),
            Edge(
                source_node_unique_name=PageImageLayoutProcessor().meta["name"],
                target_node_unique_name=FigureTableImageExtractor().meta["name"]
            )
        ]

        pipeline = Graph(nodes=nodes)
        async for item in pipeline.astream(input_data='/Users/tatoaoliang/Downloads/Work/doc_chunking/tests/test_data/1-1 买卖合同（通用版）.pdf'):
            pass
