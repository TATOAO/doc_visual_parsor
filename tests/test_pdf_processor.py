import pytest
import io
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile
import os
from PIL import Image

# Add backend to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backend.pdf_processor import extract_pdf_pages, get_pdf_document_object, close_pdf_document


class TestPDFProcessor:
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_pdf_file = Mock()
        self.mock_pdf_file.getvalue.return_value = b"mock pdf content"
        
    @patch('backend.pdf_processor.tempfile.NamedTemporaryFile')
    @patch('backend.pdf_processor.fitz.open')
    @patch('backend.pdf_processor.os.unlink')
    @patch('backend.pdf_processor.os.path.exists')
    @patch('backend.pdf_processor.st')
    def test_extract_pdf_pages_success(self, mock_st, mock_exists, mock_unlink, mock_fitz_open, mock_temp_file):
        """Test successful PDF page extraction."""
        # Setup mocks
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.pdf"
        mock_temp.__enter__ = Mock(return_value=mock_temp)
        mock_temp.__exit__ = Mock(return_value=None)
        mock_temp_file.return_value = mock_temp
        
        # Mock os.path.exists to return True so cleanup happens
        mock_exists.return_value = True
        
        mock_pdf_doc = Mock()
        mock_pdf_doc.page_count = 2
        mock_fitz_open.return_value = mock_pdf_doc
        
        # Mock page processing
        mock_page = Mock()
        mock_matrix = Mock()
        mock_pixmap = Mock()
        mock_pixmap.tobytes.return_value = b"mock image data"
        
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_pdf_doc.load_page.return_value = mock_page
        
        # Mock PIL Image
        with patch('backend.pdf_processor.Image.open') as mock_image_open:
            mock_image = Mock(spec=Image.Image)
            mock_image_open.return_value = mock_image
            
            with patch('backend.pdf_processor.fitz.Matrix') as mock_matrix_class:
                mock_matrix_class.return_value = mock_matrix
                
                # Test the function
                result = extract_pdf_pages(self.mock_pdf_file)
                
                # Assertions
                assert len(result) == 2
                assert all(isinstance(page, Mock) for page in result)
                mock_temp.write.assert_called_once_with(b"mock pdf content")
                mock_fitz_open.assert_called_once_with("/tmp/test.pdf")
                mock_pdf_doc.close.assert_called_once()
                mock_exists.assert_called_once_with("/tmp/test.pdf")
                mock_unlink.assert_called_once_with("/tmp/test.pdf")

    @patch('backend.pdf_processor.tempfile.NamedTemporaryFile')
    @patch('backend.pdf_processor.fitz.open')
    @patch('backend.pdf_processor.st')
    def test_extract_pdf_pages_error(self, mock_st, mock_fitz_open, mock_temp_file):
        """Test PDF page extraction with error."""
        # Setup mocks to raise exception
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.pdf"
        mock_temp.__enter__ = Mock(return_value=mock_temp)
        mock_temp.__exit__ = Mock(return_value=None)
        mock_temp_file.return_value = mock_temp
        
        mock_fitz_open.side_effect = Exception("PDF open error")
        
        # Test the function
        result = extract_pdf_pages(self.mock_pdf_file)
        
        # Assertions
        assert result == []
        mock_st.error.assert_called()

    @patch('backend.pdf_processor.tempfile.NamedTemporaryFile')
    @patch('backend.pdf_processor.fitz.open')
    @patch('backend.pdf_processor.os.unlink')
    @patch('backend.pdf_processor.st')
    def test_get_pdf_document_object_success(self, mock_st, mock_unlink, mock_fitz_open, mock_temp_file):
        """Test successful PDF document object retrieval."""
        # Setup mocks
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.pdf"
        mock_temp.__enter__ = Mock(return_value=mock_temp)
        mock_temp.__exit__ = Mock(return_value=None)
        mock_temp_file.return_value = mock_temp
        
        mock_pdf_doc = Mock()
        mock_fitz_open.return_value = mock_pdf_doc
        
        # Test the function
        result = get_pdf_document_object(self.mock_pdf_file)
        
        # Assertions
        assert result == mock_pdf_doc
        mock_temp.write.assert_called_once_with(b"mock pdf content")
        mock_fitz_open.assert_called_once_with("/tmp/test.pdf")
        mock_unlink.assert_called_once_with("/tmp/test.pdf")

    @patch('backend.pdf_processor.tempfile.NamedTemporaryFile')
    @patch('backend.pdf_processor.fitz.open')
    @patch('backend.pdf_processor.st')
    def test_get_pdf_document_object_error(self, mock_st, mock_fitz_open, mock_temp_file):
        """Test PDF document object retrieval with error."""
        # Setup mocks to raise exception
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.pdf"
        mock_temp.__enter__ = Mock(return_value=mock_temp)
        mock_temp.__exit__ = Mock(return_value=None)
        mock_temp_file.return_value = mock_temp
        
        mock_fitz_open.side_effect = Exception("PDF open error")
        
        # Test the function
        result = get_pdf_document_object(self.mock_pdf_file)
        
        # Assertions
        assert result is None
        mock_st.error.assert_called()

    @patch('backend.pdf_processor.st')
    def test_close_pdf_document_success(self, mock_st):
        """Test successful PDF document closing."""
        mock_pdf_doc = Mock()
        
        # Test the function
        close_pdf_document(mock_pdf_doc)
        
        # Assertions
        mock_pdf_doc.close.assert_called_once()

    @patch('backend.pdf_processor.st')
    def test_close_pdf_document_error(self, mock_st):
        """Test PDF document closing with error."""
        mock_pdf_doc = Mock()
        mock_pdf_doc.close.side_effect = Exception("Close error")
        
        # Test the function
        close_pdf_document(mock_pdf_doc)
        
        # Assertions
        mock_st.warning.assert_called()

    @patch('backend.pdf_processor.st')
    def test_close_pdf_document_none(self, mock_st):
        """Test PDF document closing with None document."""
        # Test the function
        close_pdf_document(None)
        
        # Assertions - should not raise any errors
        mock_st.warning.assert_not_called() 