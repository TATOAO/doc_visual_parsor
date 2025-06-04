import pytest
from unittest.mock import Mock, patch, MagicMock

# Add backend to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backend.session_manager import (
    initialize_session_state, 
    reset_document_state, 
    is_new_file, 
    navigate_to_section, 
    get_document_info
)


class TestSessionManager:
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_file = Mock()
        self.mock_file.name = "test_document.pdf"
        self.mock_file.size = 1024
        self.mock_file.type = "application/pdf"

    @patch('backend.session_manager.st')
    def test_initialize_session_state(self, mock_st):
        """Test session state initialization."""
        # Mock empty session state
        mock_st.session_state = {}
        
        # Test the function
        initialize_session_state()
        
        # Check that all required keys are initialized
        expected_keys = [
            'uploaded_file', 'document_pages', 'current_page', 
            'chapters', 'document_structure', 'pdf_document', 'target_page'
        ]
        
        for key in expected_keys:
            assert key in mock_st.session_state

    @patch('backend.session_manager.st')
    def test_initialize_session_state_partial(self, mock_st):
        """Test session state initialization with some existing keys."""
        # Mock partially initialized session state
        mock_st.session_state = {
            'uploaded_file': self.mock_file,
            'current_page': 5
        }
        
        initial_file = mock_st.session_state['uploaded_file']
        initial_page = mock_st.session_state['current_page']
        
        # Test the function
        initialize_session_state()
        
        # Check that existing values are preserved
        assert mock_st.session_state['uploaded_file'] == initial_file
        assert mock_st.session_state['current_page'] == initial_page
        
        # Check that missing keys are initialized
        assert 'document_pages' in mock_st.session_state
        assert 'document_structure' in mock_st.session_state

    @patch('backend.session_manager.st')
    def test_reset_document_state(self, mock_st):
        """Test document state reset."""
        # Mock session state with document data
        mock_pdf_doc = Mock()
        mock_st.session_state = {
            'document_structure': [{'text': 'Chapter 1'}],
            'document_pages': ['page1', 'page2'],
            'current_page': 5,
            'target_page': 3,
            'pdf_document': mock_pdf_doc
        }
        
        # Test the function
        reset_document_state()
        
        # Check that document-related state is reset
        assert mock_st.session_state['document_structure'] == []
        assert mock_st.session_state['document_pages'] == []
        assert mock_st.session_state['current_page'] == 0
        assert mock_st.session_state['target_page'] == 0
        assert mock_st.session_state['pdf_document'] is None
        
        # Check that PDF document was closed
        mock_pdf_doc.close.assert_called_once()

    @patch('backend.session_manager.st')
    def test_reset_document_state_no_pdf_doc(self, mock_st):
        """Test document state reset with no PDF document."""
        # Mock session state without PDF document
        mock_st.session_state = {
            'document_structure': [{'text': 'Chapter 1'}],
            'pdf_document': None
        }
        
        # Test the function (should not raise error)
        reset_document_state()
        
        # Check that state is reset
        assert mock_st.session_state['document_structure'] == []
        assert mock_st.session_state['pdf_document'] is None

    @patch('backend.session_manager.st')
    def test_reset_document_state_pdf_close_error(self, mock_st):
        """Test document state reset when PDF close raises error."""
        # Mock session state with PDF document that raises error on close
        mock_pdf_doc = Mock()
        mock_pdf_doc.close.side_effect = Exception("Close error")
        mock_st.session_state = {
            'document_structure': [],
            'pdf_document': mock_pdf_doc
        }
        
        # Test the function (should handle error gracefully)
        reset_document_state()
        
        # Check that state is still reset despite error
        assert mock_st.session_state['pdf_document'] is None

    @patch('backend.session_manager.st')
    def test_is_new_file_no_existing_file(self, mock_st):
        """Test new file detection when no file is uploaded."""
        mock_st.session_state = {'uploaded_file': None}
        
        result = is_new_file(self.mock_file)
        
        assert result is True

    @patch('backend.session_manager.st')
    def test_is_new_file_same_file(self, mock_st):
        """Test new file detection with same file."""
        mock_st.session_state = {'uploaded_file': self.mock_file}
        
        result = is_new_file(self.mock_file)
        
        assert result is False

    @patch('backend.session_manager.st')
    def test_is_new_file_different_file(self, mock_st):
        """Test new file detection with different file."""
        old_file = Mock()
        old_file.name = "old_document.pdf"
        mock_st.session_state = {'uploaded_file': old_file}
        
        result = is_new_file(self.mock_file)
        
        assert result is True

    @patch('backend.session_manager.st')
    def test_navigate_to_section_with_pages(self, mock_st):
        """Test navigation to section with document pages."""
        mock_st.session_state = {
            'document_pages': ['page1', 'page2', 'page3'],
            'current_page': 0,
            'target_page': 0
        }
        
        # Test navigation to page 2
        navigate_to_section(1)
        
        # Check that current page is updated
        assert mock_st.session_state['current_page'] == 1
        assert mock_st.session_state['target_page'] == 1
        
        # Check that success message is shown and rerun is called
        mock_st.success.assert_called()
        mock_st.rerun.assert_called()

    @patch('backend.session_manager.st')
    def test_navigate_to_section_page_beyond_range(self, mock_st):
        """Test navigation to section beyond page range."""
        mock_st.session_state = {
            'document_pages': ['page1', 'page2'],
            'current_page': 0,
            'target_page': 0
        }
        
        # Test navigation to page beyond range
        navigate_to_section(5)
        
        # Check that current page is clamped to max available
        assert mock_st.session_state['current_page'] == 1  # len(pages) - 1
        assert mock_st.session_state['target_page'] == 5

    @patch('backend.session_manager.st')
    def test_navigate_to_section_without_pages(self, mock_st):
        """Test navigation to section without document pages."""
        mock_st.session_state = {
            'document_pages': [],
            'target_page': 0
        }
        
        # Test navigation
        navigate_to_section(2)
        
        # Check that only target page is updated
        assert mock_st.session_state['target_page'] == 2
        mock_st.success.assert_called()
        mock_st.rerun.assert_called()

    @patch('backend.session_manager.st')
    def test_get_document_info_no_file(self, mock_st):
        """Test getting document info when no file is uploaded."""
        mock_st.session_state = {'uploaded_file': None}
        
        result = get_document_info()
        
        assert result is None

    @patch('backend.session_manager.st')
    def test_get_document_info_with_file(self, mock_st):
        """Test getting document info with uploaded file."""
        mock_st.session_state = {
            'uploaded_file': self.mock_file,
            'document_structure': [
                {'text': 'Chapter 1'}, 
                {'text': 'Chapter 2'}
            ],
            'document_pages': ['page1', 'page2', 'page3']
        }
        
        result = get_document_info()
        
        # Check returned info
        assert result['name'] == "test_document.pdf"
        assert result['size'] == "1.0 KB"
        assert result['type'] == "application/pdf"
        assert result['sections'] == 2
        assert result['pages'] == 3

    @patch('backend.session_manager.st')
    def test_get_document_info_no_structure(self, mock_st):
        """Test getting document info with no document structure."""
        mock_st.session_state = {
            'uploaded_file': self.mock_file,
            'document_structure': [],
            'document_pages': []
        }
        
        result = get_document_info()
        
        # Check returned info
        assert result['sections'] == 0
        assert result['pages'] == "N/A"

    @patch('backend.session_manager.st')
    def test_get_document_info_none_structure(self, mock_st):
        """Test getting document info with None document structure."""
        mock_st.session_state = {
            'uploaded_file': self.mock_file,
            'document_structure': None,
            'document_pages': None
        }
        
        result = get_document_info()
        
        # Check returned info handles None values
        assert result['sections'] == 0
        assert result['pages'] == "N/A" 