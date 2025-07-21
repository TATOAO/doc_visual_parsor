from typing import AsyncGenerator
from processor_pipeline import AsyncProcessor
from doc_chunking.schemas import FileInputData
from doc_chunking.utils.logging_config import get_logger
from pathlib import Path
from io import BytesIO
import tempfile
import subprocess
from doc_chunking.utils.helper import detect_file_type

logger = get_logger(__name__)


def convert_doc_to_pdf(file_content: FileInputData) -> bytes:
    # Accepts FileInputData.file_content as bytes or str (path)
    if isinstance(file_content, str):
        docx_path = Path(file_content)
    elif isinstance(file_content, bytes):
        # Save bytes to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_docx:
            tmp_docx.write(file_content)
            docx_path = Path(tmp_docx.name)
    elif isinstance(file_content, Path):
        docx_path = file_content
    else:
        raise ValueError("Unsupported file_content type for DOCX to PDF conversion.")

    # Create a temp file for the output PDF
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        # Use LibreOffice to convert DOCX to PDF
        subprocess.run([
            'soffice', '--headless', '--convert-to', 'pdf', '--outdir', str(output_dir), str(docx_path)
        ], check=True)
        # Find the output PDF
        pdf_path = output_dir / (docx_path.stem + '.pdf')
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
    return pdf_bytes


class WordToPdfProcessor(AsyncProcessor):
    meta = {
        "name": "WordToPdfProcessor",
        "input_type": FileInputData,
        "output_type": FileInputData,
    }

    async def process(self, data: AsyncGenerator[FileInputData, None]) -> AsyncGenerator[FileInputData, None]:
        async for item in data:

            if detect_file_type(item) == 'docx' or detect_file_type(item) == 'doc':
                pdf_content = convert_doc_to_pdf(item)
                yield pdf_content

            elif detect_file_type(item) == 'pdf':
                yield item


# python -m doc_chunking.core.processors.doc_to_pdf_processor
if __name__ == "__main__":
    async def main():
        from processor_pipeline import AsyncPipeline    
        from doc_chunking.core.processors.page_chunker import PdfPageImageSplitterProcessor
        pipeline = AsyncPipeline([
            WordToPdfProcessor(),
            PdfPageImageSplitterProcessor()
        ])
        async for item in pipeline.astream('/Users/tatoaoliang/Downloads/Work/doc_chunking/tests/test_data/1-1 买卖合同（通用版）.docx'):
            # saved to pdf result into test.pdf
            print(item)

    import asyncio

    result = asyncio.run(main())