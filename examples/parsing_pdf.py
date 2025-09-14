from doc_chunking import PdfStyleCVMixLayoutExtractor

# python examples/parsing_pdf.py
if __name__ == "__main__":
    result = PdfStyleCVMixLayoutExtractor().detect_layout("examples/test.pdf")
    import json
    with open("result.json", "w") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)