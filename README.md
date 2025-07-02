# Doc Chunking Library

A powerful document analysis and chunking library for PDF and DOCX files using AI-powered processing. This library provides intelligent document structure detection, hierarchical section extraction, and content chunking capabilities.

## Features

- 📄 **PDF Processing**: Extract layout and structure from PDF documents
- 📝 **DOCX Processing**: Analyze DOCX document structure and content
- 🔍 **AI-Powered Structure Analysis**: Intelligent detection of headings, sections, and document hierarchy
- ✂️ **Smart Chunking**: Context-aware document chunking with hierarchical relationships
- 🌐 **API Server**: Optional FastAPI server for web integration
- 🧠 **LLM Integration**: Leverage language models for advanced document understanding

## Installation

### Basic Installation

```bash
pip install doc-chunking
```

### Installation with Optional Dependencies

```bash
# For Streamlit UI components
pip install doc-chunking[streamlit-ui]

# For development tools
pip install doc-chunking[dev]

# Install all optional dependencies
pip install doc-chunking[all]
```

### Development Installation

```bash
git clone https://github.com/yourusername/doc-chunking.git
cd doc-chunking
pip install -e .[dev]
```

## Quick Start

### Basic Usage

```python
import doc_chunking

# Process a PDF file
pdf_sections = doc_chunking.quick_pdf_chunking("document.pdf")

# Process a DOCX file  
docx_sections = doc_chunking.quick_docx_chunking("document.docx")

# Access the hierarchical structure
for section in pdf_sections.sub_sections:
    print(f"Section: {section.title}")
    print(f"Content: {section.content[:100]}...")
  
    # Process subsections
    for subsection in section.sub_sections:
        print(f"  Subsection: {subsection.title}")
```

### Advanced Usage

```python
from doc_chunking import (
    PDFLayoutExtractor, 
    DocxLayoutExtractor,
    section_reconstructor,
    title_structure_builder_llm
)

# Extract layout from PDF
pdf_extractor = PDFLayoutExtractor()
layout_result = pdf_extractor.extract_layout("document.pdf")

# Build title structure using LLM
title_structure = title_structure_builder_llm(layout_result)

# Reconstruct hierarchical sections
sections = section_reconstructor(title_structure, layout_result)

# Access detailed section information
def print_section_tree(section, level=0):
    indent = "  " * level
    print(f"{indent}{section.title}")
    print(f"{indent}Content length: {len(section.content)}")
    print(f"{indent}Element ID: {section.element_id}")
  
    for subsection in section.sub_sections:
        print_section_tree(subsection, level + 1)

print_section_tree(sections)
```

### Streaming Processing

```python
import asyncio
from doc_chunking import streaming_section_reconstructor, title_structure_builder_llm

async def process_document_streaming(layout_result):
    # Create a streaming title structure generator
    async def title_stream():
        title_structure = title_structure_builder_llm(layout_result)
        # Simulate streaming by yielding chunks
        for line in title_structure.split('\n'):
            yield line + '\n'
            await asyncio.sleep(0.01)  # Simulate processing delay
  
    # Process sections as they become available
    async for partial_sections in streaming_section_reconstructor(title_stream(), layout_result):
        print(f"Received {len(partial_sections.sub_sections)} sections so far...")
        # Process partial results as needed

# Run the streaming example
# asyncio.run(process_document_streaming(layout_result))
```

## API Server

The library includes an optional FastAPI server for web integration:

```bash
# Start the server
doc-chunking-server

# With custom options
doc-chunking-server --host 0.0.0.0 --port 8080 --reload
```

### API Endpoints

- `POST /api/upload-document` - Upload and process a document
- `POST /api/analyze-pdf-structure` - Analyze PDF structure
- `POST /api/analyze-structure` - Analyze document structure
- `GET /docs` - Interactive API documentation

## Configuration

### Environment Variables

```bash
# OpenAI API key for LLM processing
export OPENAI_API_KEY="your-api-key-here"

# Optional: Custom model configurations
export DOC_CHUNKING_MODEL="gpt-4"
export DOC_CHUNKING_MAX_TOKENS=4000
```

### Configuration File

Create a `.env` file in your project directory:

```env
DOC_CHUNKING_LLM_API_KEY=""
DOC_CHUNKING_LLM_BASE_URL=""
DOC_CHUNKING_LLM_MODEL="qwen3-4b"
```

The library uses Pydantic models for type safety and validation:

```python
from doc_chunking import Section, LayoutExtractionResult, LayoutElement

# Section represents a hierarchical document section
section = Section(
    title="Chapter 1",
    content="Chapter content...",
    level=0,
    element_id=1,
    sub_sections=[],
    parent_section=None
)

# LayoutExtractionResult contains the document layout
layout = LayoutExtractionResult(
    elements=[
        LayoutElement(
            id=1,
            text="Document text",
            type="paragraph",
            bbox=[100, 200, 300, 250]
        )
    ]
)
```

## Supported Document Types

### PDF Documents

- Text-based PDFs with extractable content
- Mixed text and image PDFs
- Multi-column layouts
- Complex document structures

### DOCX Documents

- Microsoft Word documents (.docx)
- Rich formatting and styles
- Embedded images and tables
- Hierarchical heading structures

## Performance Optimization

### Batch Processing

```python
from doc_chunking import PDFLayoutExtractor

extractor = PDFLayoutExtractor()

# Process multiple documents efficiently
documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
results = []

for doc_path in documents:
    layout = extractor.extract_layout(doc_path)
    sections = doc_chunking.quick_pdf_chunking(doc_path)
    results.append((doc_path, sections))
```

### Memory Management

For large documents, consider processing in chunks:

```python
# For very large documents, you might want to process sections independently
def process_large_document(doc_path, chunk_size=10):
    extractor = PDFLayoutExtractor()
    layout = extractor.extract_layout(doc_path)
  
    # Process elements in chunks
    elements = layout.elements
    for i in range(0, len(elements), chunk_size):
        chunk_elements = elements[i:i+chunk_size]
        # Process chunk_elements as needed
        yield chunk_elements
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/yourusername/doc-chunking.git
cd doc-chunking
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .[dev]
```

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black doc_chunking/
isort doc_chunking/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### v0.1.0

- Initial release
- PDF and DOCX processing capabilities
- AI-powered structure analysis
- Hierarchical section extraction
- FastAPI server integration
- Streaming processing support

## Support

- 📖 [Documentation](https://github.com/yourusername/doc-chunking#readme)
- 🐛 [Issue Tracker](https://github.com/yourusername/doc-chunking/issues)
- 💬 [Discussions](https://github.com/yourusername/doc-chunking/discussions)

## Citation

If you use this library in your research, please cite:

```bibtex
@software{doc_chunking,
  title={Doc Chunking: AI-Powered Document Analysis and Chunking Library},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/doc-chunking}
}
```
