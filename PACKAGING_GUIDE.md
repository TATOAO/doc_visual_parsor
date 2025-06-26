# Doc Chunking Library Packaging Guide

This guide explains how the `doc-chunking` library has been packaged and how to use it.

## 🎯 Packaging Summary

The project has been successfully packaged as a Python library using modern packaging standards:

- **Package name**: `doc-chunking`
- **Version**: `0.1.0`
- **Package format**: Modern `pyproject.toml` configuration
- **Build system**: setuptools with pip-installable wheel and source distributions
- **Command-line tools**: Includes `doc-chunking-server` script

## 📦 Installation

### From PyPI (when published)
```bash
pip install doc-chunking
```

### From Local Build
```bash
# Install from wheel
pip install dist/doc_chunking-0.1.0-py3-none-any.whl

# Or install from source
pip install dist/doc_chunking-0.1.0.tar.gz
```

### Development Installation
```bash
pip install -e .
```

## 🚀 Quick Usage Examples

### Basic Document Processing

```python
import doc_chunking

# Quick PDF processing
sections = doc_chunking.quick_pdf_chunking("document.pdf")
print(f"Found {len(sections.sub_sections)} main sections")

# Quick DOCX processing
sections = doc_chunking.quick_docx_chunking("document.docx")
for section in sections.sub_sections:
    print(f"Section: {section.title}")
    print(f"Content preview: {section.content[:100]}...")
```

### Advanced Usage with Individual Components

```python
import doc_chunking

# Get extractors
pdf_extractor = doc_chunking.get_pdf_extractor()
docx_extractor = doc_chunking.get_docx_extractor()

# Process a document
layout_result = pdf_extractor.extract_layout("document.pdf")
print(f"Extracted {len(layout_result.elements)} layout elements")

# Get specific functions
section_reconstructor = doc_chunking.get_section_reconstructor()
title_builder = doc_chunking.get_title_structure_builder()
```

### Working with Data Models

```python
from doc_chunking import Section, LayoutExtractionResult, LayoutElement

# Create a section programmatically
section = Section(
    title="Introduction",
    content="This is the introduction content...",
    level=0,
    element_id=1,
    sub_sections=[],
    parent_section=None
)

print(f"Section: {section.title} at level {section.level}")
```

## 🖥️ Command Line Usage

The package includes a command-line server:

```bash
# Start the server with defaults
doc-chunking-server

# Custom configuration
doc-chunking-server --host localhost --port 8080 --reload

# See all options
doc-chunking-server --help
```

## 📁 Package Structure

```
doc_chunking/
├── __init__.py                 # Main package interface
├── schemas/                    # Pydantic data models
│   ├── layout_schemas.py      # Layout extraction models
│   └── schemas.py             # Core data models
├── layout_detection/          # Document layout analysis
│   ├── layout_extraction/     # PDF/DOCX extractors
│   ├── visual_detection/      # Computer vision components
│   └── utils/                 # Layout processing utilities
├── layout_structuring/        # Structure building
│   ├── title_structure_builder_llm/  # LLM-based structuring
│   └── content_merger/        # Content merging utilities
├── naive_llm/                 # LLM processing agents
│   ├── agents/                # Processing agents
│   └── helpers/               # Helper utilities
├── documents_chunking/        # Main chunking logic
├── utils/                     # General utilities
└── scripts/                   # Command-line scripts
```

## 🔧 Build Process

### Prerequisites

```bash
# Install build tools
pip install build twine
```

### Building the Package

```bash
# Clean previous builds
rm -rf build/ dist/ doc_chunking.egg-info/

# Build the package
python -m build

# This creates:
# - dist/doc_chunking-0.1.0-py3-none-any.whl (wheel)
# - dist/doc_chunking-0.1.0.tar.gz (source distribution)
```

### Testing the Build

```bash
# Install locally for testing
pip install dist/doc_chunking-0.1.0-py3-none-any.whl

# Test import
python -c "import doc_chunking; print('✓ Import successful')"

# Test command-line tool
doc-chunking-server --help
```

### Quality Checks

```bash
# Check package metadata
twine check dist/*

# Run tests
pytest tests/
```

## 📤 Publishing to PyPI

### Test PyPI (Recommended First)

```bash
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ doc-chunking
```

### Production PyPI

```bash
# Upload to production PyPI
twine upload dist/*

# Install from PyPI
pip install doc-chunking
```

## 🔄 Version Management

Update version in `pyproject.toml`:

```toml
[project]
version = "0.2.0"  # Update version here
```

Then rebuild:

```bash
python -m build
```

## 🐛 Troubleshooting

### Import Errors

If you encounter import errors:

1. Check that all dependencies are installed
2. Verify the package structure is correct
3. Ensure relative imports are used correctly within the package

### Missing Dependencies

The package requires several dependencies. Install with optional extras:

```bash
# Install with all optional dependencies
pip install doc-chunking[all]

# Install specific extras
pip install doc-chunking[streamlit-ui]  # For Streamlit UI components
pip install doc-chunking[dev]           # For development tools
```

### Build Issues

If build fails:

1. Ensure you have the latest build tools: `pip install --upgrade build setuptools wheel`
2. Check for circular imports in the package
3. Verify all required files are included in `MANIFEST.in`

## 🧪 Testing in Different Environments

### Virtual Environment Testing

```bash
# Create fresh environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install package
pip install dist/doc_chunking-0.1.0-py3-none-any.whl

# Test functionality
python -c "import doc_chunking; print('✓ Works in clean environment')"

# Deactivate
deactivate
rm -rf test_env
```

### Docker Testing

```dockerfile
FROM python:3.10-slim

COPY dist/doc_chunking-0.1.0-py3-none-any.whl /tmp/
RUN pip install /tmp/doc_chunking-0.1.0-py3-none-any.whl

CMD ["python", "-c", "import doc_chunking; print('✓ Works in Docker')"]
```

## 📚 Integration Examples

### In a FastAPI Application

```python
from fastapi import FastAPI
import doc_chunking

app = FastAPI()

@app.post("/process-pdf/")
async def process_pdf(file_path: str):
    sections = doc_chunking.quick_pdf_chunking(file_path)
    return {
        "sections_count": len(sections.sub_sections),
        "sections": [{"title": s.title, "level": s.level} for s in sections.sub_sections]
    }
```

### In a Django Application

```python
# views.py
from django.http import JsonResponse
import doc_chunking

def process_document(request):
    if request.method == 'POST':
        file_path = request.POST.get('file_path')
        sections = doc_chunking.quick_pdf_chunking(file_path)
        return JsonResponse({
            'sections': [s.title for s in sections.sub_sections]
        })
```

### In a Jupyter Notebook

```python
import doc_chunking

# Process document
sections = doc_chunking.quick_pdf_chunking("research_paper.pdf")

# Analyze structure
def analyze_structure(section, level=0):
    indent = "  " * level
    print(f"{indent}{section.title} ({len(section.content)} chars)")
    for subsection in section.sub_sections:
        analyze_structure(subsection, level + 1)

analyze_structure(sections)
```

## 🎉 Success!

Your document chunking project is now successfully packaged as a Python library! You can:

- ✅ Install it with `pip install doc-chunking`
- ✅ Import it with `import doc_chunking`
- ✅ Use convenience functions like `doc_chunking.quick_pdf_chunking()`
- ✅ Access individual components through lazy loading
- ✅ Run the server with `doc-chunking-server`
- ✅ Integrate it into other Python projects

The library follows modern Python packaging best practices and is ready for distribution! 