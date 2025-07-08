from docx2pdf import convert
from pathlib import Path

def convert_docx_to_pdf(docx_path: Path) -> Path:
    """
    Convert a DOCX file to a PDF file.
    """
    pdf_path = docx_path.with_suffix('.pdf')
    convert(docx_path, pdf_path)
    return pdf_path