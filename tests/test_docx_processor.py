import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Add backend to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backend.docx_processor import extract_docx_content, extract_docx_structure


class TestDOCXProcessor:
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_docx_file = Mock()
        self.mock_docx_file.getvalue.return_value = b"mock docx content"
        
    @patch('backend.docx_processor.tempfile.NamedTemporaryFile')
    @patch('backend.docx_processor.docx.Document')
    @patch('backend.docx_processor.os.unlink')
    @patch('backend.docx_processor.st')
    def test_extract_docx_content_success(self, mock_st, mock_unlink, mock_docx_document, mock_temp_file):
        """Test successful DOCX content extraction."""
        # Setup mocks
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.docx"
        mock_temp.__enter__ = Mock(return_value=mock_temp)
        mock_temp.__exit__ = Mock(return_value=None)
        mock_temp_file.return_value = mock_temp
        
        # Mock document paragraphs
        mock_paragraph1 = Mock()
        mock_paragraph1.text = "First paragraph content"
        mock_paragraph2 = Mock()
        mock_paragraph2.text = "  Second paragraph content  "
        mock_paragraph3 = Mock()
        mock_paragraph3.text = "   "  # Empty paragraph (should be skipped)
        
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_paragraph1, mock_paragraph2, mock_paragraph3]
        mock_docx_document.return_value = mock_doc
        
        # Test the function
        result = extract_docx_content(self.mock_docx_file)
        
        # Assertions
        assert len(result) == 2
        assert result[0] == "First paragraph content"
        assert result[1] == "  Second paragraph content  "
        mock_temp.write.assert_called_once_with(b"mock docx content")
        mock_docx_document.assert_called_once_with("/tmp/test.docx")
        mock_unlink.assert_called_once_with("/tmp/test.docx")

    @patch('backend.docx_processor.tempfile.NamedTemporaryFile')
    @patch('backend.docx_processor.docx.Document')
    @patch('backend.docx_processor.st')
    def test_extract_docx_content_error(self, mock_st, mock_docx_document, mock_temp_file):
        """Test DOCX content extraction with error."""
        # Setup mocks to raise exception
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.docx"
        mock_temp.__enter__ = Mock(return_value=mock_temp)
        mock_temp.__exit__ = Mock(return_value=None)
        mock_temp_file.return_value = mock_temp
        
        mock_docx_document.side_effect = Exception("DOCX open error")
        
        # Test the function
        result = extract_docx_content(self.mock_docx_file)
        
        # Assertions
        assert result == []
        mock_st.error.assert_called()

    @patch('backend.docx_processor.tempfile.NamedTemporaryFile')
    @patch('backend.docx_processor.docx.Document')
    @patch('backend.docx_processor.os.unlink')
    @patch('backend.docx_processor.st')
    def test_extract_docx_structure_success(self, mock_st, mock_unlink, mock_docx_document, mock_temp_file):
        """Test successful DOCX structure extraction."""
        # Setup mocks
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.docx"
        mock_temp.__enter__ = Mock(return_value=mock_temp)
        mock_temp.__exit__ = Mock(return_value=None)
        mock_temp_file.return_value = mock_temp
        
        # Mock document paragraphs with different styles
        mock_paragraph1 = Mock()
        mock_paragraph1.text = "Chapter 1: Introduction"
        mock_paragraph1.style.name = "Heading 1"
        
        mock_paragraph2 = Mock()
        mock_paragraph2.text = "Section A"
        mock_paragraph2.style.name = "Heading 2"
        
        mock_paragraph3 = Mock()
        mock_paragraph3.text = "Regular paragraph"
        mock_paragraph3.style.name = "Normal"
        
        mock_paragraph4 = Mock()
        mock_paragraph4.text = "Subsection"
        mock_paragraph4.style.name = "Heading 3"
        
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_paragraph1, mock_paragraph2, mock_paragraph3, mock_paragraph4]
        mock_docx_document.return_value = mock_doc
        
        # Test the function
        result = extract_docx_structure(self.mock_docx_file)
        
        # Assertions
        assert len(result) == 3  # Only headings should be included
        
        assert result[0]['text'] == "Chapter 1: Introduction"
        assert result[0]['level'] == 1
        assert result[0]['paragraph_index'] == 0
        
        assert result[1]['text'] == "Section A"
        assert result[1]['level'] == 2
        assert result[1]['paragraph_index'] == 1
        
        assert result[2]['text'] == "Subsection"
        assert result[2]['level'] == 3
        assert result[2]['paragraph_index'] == 3

    @patch('backend.docx_processor.tempfile.NamedTemporaryFile')
    @patch('backend.docx_processor.docx.Document')
    @patch('backend.docx_processor.os.unlink')
    @patch('backend.docx_processor.st')
    def test_extract_docx_structure_with_numbered_headings(self, mock_st, mock_unlink, mock_docx_document, mock_temp_file):
        """Test DOCX structure extraction with numbered headings."""
        # Setup mocks
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.docx"
        mock_temp.__enter__ = Mock(return_value=mock_temp)
        mock_temp.__exit__ = Mock(return_value=None)
        mock_temp_file.return_value = mock_temp
        
        # Mock paragraph with numbered heading style
        mock_paragraph = Mock()
        mock_paragraph.text = "Advanced Topic"
        mock_paragraph.style.name = "Heading 5"
        
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_paragraph]
        mock_docx_document.return_value = mock_doc
        
        # Test the function
        result = extract_docx_structure(self.mock_docx_file)
        
        # Assertions
        assert len(result) == 1
        assert result[0]['text'] == "Advanced Topic"
        assert result[0]['level'] == 5
        assert result[0]['paragraph_index'] == 0

    @patch('backend.docx_processor.tempfile.NamedTemporaryFile')
    @patch('backend.docx_processor.docx.Document')
    @patch('backend.docx_processor.st')
    def test_extract_docx_structure_error(self, mock_st, mock_docx_document, mock_temp_file):
        """Test DOCX structure extraction with error."""
        # Setup mocks to raise exception
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.docx"
        mock_temp.__enter__ = Mock(return_value=mock_temp)
        mock_temp.__exit__ = Mock(return_value=None)
        mock_temp_file.return_value = mock_temp
        
        mock_docx_document.side_effect = Exception("DOCX structure error")
        
        # Test the function
        result = extract_docx_structure(self.mock_docx_file)
        
        # Assertions
        assert result == []
        mock_st.error.assert_called()

    @patch('backend.docx_processor.tempfile.NamedTemporaryFile')
    @patch('backend.docx_processor.docx.Document')
    @patch('backend.docx_processor.os.unlink')
    @patch('backend.docx_processor.st')
    def test_extract_docx_structure_no_headings(self, mock_st, mock_unlink, mock_docx_document, mock_temp_file):
        """Test DOCX structure extraction with no headings."""
        # Setup mocks
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.docx"
        mock_temp.__enter__ = Mock(return_value=mock_temp)
        mock_temp.__exit__ = Mock(return_value=None)
        mock_temp_file.return_value = mock_temp
        
        # Mock document with only normal paragraphs
        mock_paragraph1 = Mock()
        mock_paragraph1.text = "Normal paragraph 1"
        mock_paragraph1.style.name = "Normal"
        
        mock_paragraph2 = Mock()
        mock_paragraph2.text = "Normal paragraph 2"
        mock_paragraph2.style.name = "Body Text"
        
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_paragraph1, mock_paragraph2]
        mock_docx_document.return_value = mock_doc
        
        # Test the function
        result = extract_docx_structure(self.mock_docx_file)
        
        # Assertions
        assert len(result) == 0  # No headings should be found

    @patch('backend.docx_processor.tempfile.NamedTemporaryFile')
    @patch('backend.docx_processor.docx.Document')
    @patch('backend.docx_processor.os.unlink')
    @patch('backend.docx_processor.st')
    def test_extract_docx_structure_empty_paragraphs(self, mock_st, mock_unlink, mock_docx_document, mock_temp_file):
        """Test DOCX structure extraction with empty paragraphs."""
        # Setup mocks
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.docx"
        mock_temp.__enter__ = Mock(return_value=mock_temp)
        mock_temp.__exit__ = Mock(return_value=None)
        mock_temp_file.return_value = mock_temp
        
        # Mock document with empty heading paragraph
        mock_paragraph1 = Mock()
        mock_paragraph1.text = "   "  # Empty/whitespace only
        mock_paragraph1.style.name = "Heading 1"
        
        mock_paragraph2 = Mock()
        mock_paragraph2.text = "Valid Heading"
        mock_paragraph2.style.name = "Heading 1"
        
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_paragraph1, mock_paragraph2]
        mock_docx_document.return_value = mock_doc
        
        # Test the function
        result = extract_docx_structure(self.mock_docx_file)
        
        # Assertions
        assert len(result) == 1  # Only non-empty heading should be included
        assert result[0]['text'] == "Valid Heading" 