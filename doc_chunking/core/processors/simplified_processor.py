from typing import AsyncGenerator
from processor_pipeline import AsyncProcessor
from doc_chunking.merging import PdfStyleCVMixLayoutExtractor
from doc_chunking.schemas import Section
from doc_chunking.layout_structuring.title_structure_builder_llm.flatten_sections_generator import _flatten_section_tree
from doc_chunking.layout_structuring.title_structure_builder_llm.structurer_llm import stream_title_structure_builder_llm_with_plain_text
from doc_chunking.layout_structuring.title_structure_builder_llm.layout_displayer import display_layout
from doc_chunking.layout_structuring.title_structure_builder_llm.section_reconstructor import section_reconstructor
from doc_chunking.core.processors.doc_to_pdf_processor import convert_doc_to_pdf
from doc_chunking.utils.helper import detect_file_type
from loguru import logger
import tempfile
import os




async def simplified_processor(pdf_path: str, model = None) -> str:

    if model is None:
        pdf_style_cv_mix_layout_extractor = PdfStyleCVMixLayoutExtractor(
            model_path="model_parameters/layout_detection/docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx",
            cv_confidence_threshold=0.1  # Use lower threshold for better detection
        )
    else:
        pdf_style_cv_mix_layout_extractor = model

    temp_pdf_path = None
    try:
        file_type = detect_file_type(pdf_path)
        effective_pdf_path = pdf_path

        if file_type in ("doc", "docx"):
            pdf_bytes = convert_doc_to_pdf(pdf_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                tmp_pdf.write(pdf_bytes)
                temp_pdf_path = tmp_pdf.name
            effective_pdf_path = temp_pdf_path

        result = pdf_style_cv_mix_layout_extractor.detect_layout(effective_pdf_path)
        all_text = display_layout(result)
        next_title_structure = ""
        async for chunk in stream_title_structure_builder_llm_with_plain_text(next_title_structure, all_text):
            next_title_structure += chunk

        logger.info(f"next_title_structure: {next_title_structure}")

        sections = section_reconstructor(next_title_structure, result)
        flattened_sections = _flatten_section_tree(sections)

        return flattened_sections
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.unlink(temp_pdf_path)
            except Exception:
                pass


class SimplifiedProcessor(AsyncProcessor):
    meta = {
        "name": "SimplifiedProcessor",
        "input_type": str,
        "output_type": Section,
    }
    def __init__(self, model = None, **kwargs):
        super().__init__(**kwargs)
        if model is None:
            self.model = PdfStyleCVMixLayoutExtractor(
                    model_path="model_parameters/layout_detection/docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx",
                    cv_confidence_threshold=0.1  # Use lower threshold for better detection
                )
        else:
            self.model = model

    async def process(self, chunk_generator: AsyncGenerator[str, None]) -> AsyncGenerator[Section, None]:
        async for chunk in chunk_generator:
            result = await simplified_processor(chunk, self.model)
            yield result


# python -m doc_chunking.core.processors.simplified_processor
if __name__ == "__main__":
    import asyncio
    # asyncio.run(simplified_processor("./智能设备.pdf"))
    result = asyncio.run(simplified_processor("/Users/tatoao_mini/Downloads/劳动合同(1).docx"))
    print(result)