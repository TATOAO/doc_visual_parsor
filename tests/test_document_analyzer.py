import pytest
from unittest.mock import Mock, patch, MagicMock

# Add backend to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backend.document_analyzer import (
    extract_pdf_document_structure, 
    analyze_document_structure, 
    get_structure_summary
)


class TestDocumentAnalyzer:
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_pdf_file = Mock()
        self.mock_pdf_file.type = "application/pdf"
        self.mock_pdf_file.getvalue.return_value = b"mock pdf content"
        
        self.mock_docx_file = Mock()
        self.mock_docx_file.type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        
    @patch('backend.document_analyzer.get_pdf_document_object')
    @patch('backend.document_analyzer.st')
    def test_extract_pdf_document_structure_success(self, mock_st, mock_get_pdf_doc):
        """Test successful PDF document structure extraction."""
        # Setup mock PDF document
        mock_pdf_doc = Mock()
        mock_pdf_doc.page_count = 2
        mock_get_pdf_doc.return_value = mock_pdf_doc
        
        # Mock page structure
        mock_page = Mock()
        mock_pdf_doc.load_page.return_value = mock_page
        
        # Mock text blocks with heading-like content
        mock_blocks = {
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Chapter 1: Introduction",
                                    "size": 18,
                                    "flags": 16,  # Bold flag
                                    "bbox": [0, 100, 200, 120]
                                }
                            ]
                        }
                    ]
                },
                {
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "1.1 Overview",
                                    "size": 14,
                                    "flags": 16,  # Bold flag
                                    "bbox": [0, 150, 150, 170]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        mock_page.get_text.return_value = mock_blocks
        
        # Test the function
        result = extract_pdf_document_structure(self.mock_pdf_file)
        
        # Assertions
        assert len(result) == 4  # 2 headings × 2 pages
        assert all('text' in item for item in result)
        assert all('page' in item for item in result)
        assert all('level' in item for item in result)
        
        # Check if session state is set
        mock_st.session_state.pdf_document = mock_pdf_doc

    @patch('backend.document_analyzer.get_pdf_document_object')
    @patch('backend.document_analyzer.st')
    def test_extract_pdf_document_structure_no_doc(self, mock_st, mock_get_pdf_doc):
        """Test PDF structure extraction when document can't be opened."""
        mock_get_pdf_doc.return_value = None
        
        # Test the function
        result = extract_pdf_document_structure(self.mock_pdf_file)
        
        # Assertions
        assert result == []

    @patch('backend.document_analyzer.get_pdf_document_object')
    @patch('backend.document_analyzer.st')
    def test_extract_pdf_document_structure_with_patterns(self, mock_st, mock_get_pdf_doc):
        """Test PDF structure extraction with common heading patterns."""
        # Setup mock PDF document
        mock_pdf_doc = Mock()
        mock_pdf_doc.page_count = 1
        mock_get_pdf_doc.return_value = mock_pdf_doc
        
        # Mock page structure
        mock_page = Mock()
        mock_pdf_doc.load_page.return_value = mock_page
        
        # Mock text blocks with different heading patterns
        mock_blocks = {
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Chapter 1",
                                    "size": 12,
                                    "flags": 0,
                                    "bbox": [0, 100, 100, 120]
                                }
                            ]
                        }
                    ]
                },
                {
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "1. Introduction",
                                    "size": 12,
                                    "flags": 0,
                                    "bbox": [0, 150, 150, 170]
                                }
                            ]
                        }
                    ]
                },
                {
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "EXECUTIVE SUMMARY",
                                    "size": 12,
                                    "flags": 0,
                                    "bbox": [0, 200, 200, 220]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        mock_page.get_text.return_value = mock_blocks
        
        # Test the function
        result = extract_pdf_document_structure(self.mock_pdf_file)
        
        # Assertions
        assert len(result) == 3
        assert any("Chapter 1" in item['text'] for item in result)
        assert any("1. Introduction" in item['text'] for item in result)
        assert any("EXECUTIVE SUMMARY" in item['text'] for item in result)

    @patch('backend.document_analyzer.extract_pdf_document_structure')
    def test_analyze_document_structure_pdf(self, mock_extract_pdf):
        """Test document structure analysis for PDF files."""
        expected_structure = [
            {'text': 'Chapter 1', 'page': 0, 'level': 1},
            {'text': 'Section 1.1', 'page': 0, 'level': 2}
        ]
        mock_extract_pdf.return_value = expected_structure
        
        # Test the function
        result = analyze_document_structure(self.mock_pdf_file)
        
        # Assertions
        assert result == expected_structure
        mock_extract_pdf.assert_called_once_with(self.mock_pdf_file)

    @patch('backend.docx_processor.extract_docx_structure')
    def test_analyze_document_structure_docx(self, mock_extract_docx):
        """Test document structure analysis for DOCX files."""
        mock_docx_structure = [
            {'text': 'Heading 1', 'level': 1, 'paragraph_index': 0},
            {'text': 'Heading 2', 'level': 2, 'paragraph_index': 2}
        ]
        mock_extract_docx.return_value = mock_docx_structure
        
        # Test the function
        result = analyze_document_structure(self.mock_docx_file)
        
        # Assertions
        assert len(result) == 2
        assert result[0]['text'] == 'Heading 1'
        assert result[0]['page'] == 0  # DOCX doesn't have pages
        assert result[0]['level'] == 1
        assert 'paragraph_index' in result[0]

    @patch('backend.document_analyzer.st')
    def test_analyze_document_structure_unsupported(self, mock_st):
        """Test document structure analysis for unsupported file types."""
        mock_file = Mock()
        mock_file.type = "text/plain"
        
        # Test the function
        result = analyze_document_structure(mock_file)
        
        # Assertions
        assert result == []
        mock_st.warning.assert_called()

    def test_get_structure_summary_empty(self):
        """Test structure summary with empty structure."""
        result = get_structure_summary([])
        assert result == "No structure detected"

    def test_get_structure_summary_with_structure(self):
        """Test structure summary with various heading levels."""
        structure = [
            {'level': 1, 'text': 'Chapter 1'},
            {'level': 1, 'text': 'Chapter 2'},
            {'level': 2, 'text': 'Section 2.1'},
            {'level': 2, 'text': 'Section 2.2'},
            {'level': 3, 'text': 'Subsection 2.2.1'},
            {'level': 4, 'text': 'Sub-subsection'},
            {'level': 5, 'text': 'Level 5 heading'}
        ]
        
        result = get_structure_summary(structure)
        
        # Should contain counts for each level
        assert "2 Main Headings" in result
        assert "2 Sections" in result
        assert "1 Subsections" in result
        assert "1 Sub-subsections" in result
        assert "1 Level 5 headings" in result

    def test_get_structure_summary_single_level(self):
        """Test structure summary with only one heading level."""
        structure = [
            {'level': 2, 'text': 'Section 1'},
            {'level': 2, 'text': 'Section 2'},
            {'level': 2, 'text': 'Section 3'}
        ]
        
        result = get_structure_summary(structure)
        assert result == "3 Sections"

    @patch('backend.document_analyzer.get_pdf_document_object')
    @patch('backend.document_analyzer.st')
    def test_extract_pdf_document_structure_error(self, mock_st, mock_get_pdf_doc):
        """Test PDF structure extraction with error during processing."""
        # Setup mock PDF document that raises error during page processing
        mock_pdf_doc = Mock()
        mock_pdf_doc.page_count = 1
        mock_get_pdf_doc.return_value = mock_pdf_doc
        
        mock_pdf_doc.load_page.side_effect = Exception("Page load error")
        
        # Test the function
        result = extract_pdf_document_structure(self.mock_pdf_file)
        
        # Should handle error gracefully and return empty structure
        assert result == []

    @patch('backend.document_analyzer.get_pdf_document_object')
    @patch('backend.document_analyzer.st')
    def test_extract_pdf_document_structure_duplicate_removal(self, mock_st, mock_get_pdf_doc):
        """Test that duplicate headings are removed from structure."""
        # Setup mock PDF document
        mock_pdf_doc = Mock()
        mock_pdf_doc.page_count = 1
        mock_get_pdf_doc.return_value = mock_pdf_doc
        
        # Mock page structure with duplicate content
        mock_page = Mock()
        mock_pdf_doc.load_page.return_value = mock_page
        
        mock_blocks = {
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Chapter 1: Introduction",
                                    "size": 18,
                                    "flags": 16,
                                    "bbox": [0, 100, 200, 120]
                                }
                            ]
                        }
                    ]
                },
                {
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Chapter 1: Introduction",  # Duplicate
                                    "size": 18,
                                    "flags": 16,
                                    "bbox": [0, 200, 200, 220]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        mock_page.get_text.return_value = mock_blocks
        
        # Test the function
        result = extract_pdf_document_structure(self.mock_pdf_file)
        
        # Should only have one instance of the heading
        assert len(result) == 1
        assert result[0]['text'] == "Chapter 1: Introduction" 