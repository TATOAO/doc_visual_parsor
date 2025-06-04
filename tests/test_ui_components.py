import pytest
from unittest.mock import Mock, patch, MagicMock

# Add backend to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backend.ui_components import (
    check_pdf_viewer_availability,
    display_docx_content,
    render_upload_area
)


class SessionStateMock:
    """Mock class that supports both attribute and dictionary-style access"""
    def __init__(self, initial_data=None):
        self._data = initial_data or {}
        
    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattribute__(name)
        return self._data.get(name)
        
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            if not hasattr(self, '_data'):
                super().__setattr__('_data', {})
            self._data[name] = value
            
    def __contains__(self, key):
        return key in self._data
        
    def __getitem__(self, key):
        return self._data[key]
        
    def __setitem__(self, key, value):
        self._data[key] = value


class TestUIComponents:
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_file = Mock()
        self.mock_file.name = "test_document.pdf"
        self.mock_file.type = "application/pdf"

    @patch('backend.ui_components.HAS_PDF_VIEWER', True)
    @patch('backend.ui_components.st')
    def test_check_pdf_viewer_availability_available(self, mock_st):
        """Test PDF viewer availability check when viewer is available."""
        result = check_pdf_viewer_availability()
        
        assert result is True
        mock_st.warning.assert_not_called()

    @patch('backend.ui_components.HAS_PDF_VIEWER', False)
    @patch('backend.ui_components.st')
    def test_check_pdf_viewer_availability_not_available(self, mock_st):
        """Test PDF viewer availability check when viewer is not available."""
        result = check_pdf_viewer_availability()
        
        assert result is False
        mock_st.warning.assert_called_once()

    @patch('backend.ui_components.st')
    def test_display_docx_content_with_content(self, mock_st):
        """Test DOCX content display with content."""
        content = [
            "First paragraph of the document",
            "Second paragraph with more content",
            "Third paragraph concluding the document"
        ]
        
        display_docx_content(content)
        
        # Check that markdown header is displayed
        mock_st.markdown.assert_called_with("### Document Content")
        
        # Check that each paragraph is written
        assert mock_st.write.call_count == len(content)
        for paragraph in content:
            mock_st.write.assert_any_call(paragraph)

    @patch('backend.ui_components.st')
    def test_display_docx_content_empty(self, mock_st):
        """Test DOCX content display with empty content."""
        display_docx_content([])
        
        # Should return early without calling any streamlit functions
        mock_st.markdown.assert_not_called()
        mock_st.write.assert_not_called()

    @patch('backend.ui_components.st')
    def test_display_docx_content_none(self, mock_st):
        """Test DOCX content display with None content."""
        display_docx_content(None)
        
        # Should return early without calling any streamlit functions
        mock_st.markdown.assert_not_called()
        mock_st.write.assert_not_called()

    @patch('backend.ui_components.st')
    def test_render_upload_area(self, mock_st):
        """Test upload area rendering."""
        render_upload_area()
        
        # Check that markdown is called with upload area HTML
        mock_st.markdown.assert_called_once()
        call_args = mock_st.markdown.call_args[0][0]
        
        # Verify key elements are in the HTML
        assert "upload-area" in call_args
        assert "📤 Upload Your Document" in call_args
        assert "Drag and drop" in call_args
        assert "PDF, DOCX" in call_args
        assert "unsafe_allow_html=True" in str(mock_st.markdown.call_args)

    @patch('backend.ui_components.st')
    @patch('backend.ui_components.navigate_to_section')
    def test_render_sidebar_structure_with_structure(self, mock_navigate, mock_st):
        """Test sidebar structure rendering with document structure."""
        # Import here to avoid issues with the mock
        from backend.ui_components import render_sidebar_structure
        
        document_structure = [
            {'text': 'Chapter 1: Introduction', 'level': 1, 'page': 0},
            {'text': 'Section 1.1', 'level': 2, 'page': 0},
            {'text': 'Chapter 2: Methods', 'level': 1, 'page': 5},
            {'text': 'Subsection 2.1.1', 'level': 3, 'page': 7}
        ]
        
        # Mock button interactions
        mock_st.button.return_value = False  # No button clicked
        
        render_sidebar_structure(document_structure)
        
        # Check that table of contents header is displayed
        mock_st.markdown.assert_any_call("#### Table of Contents")
        
        # Check that buttons are created for each structure item
        assert mock_st.button.call_count == len(document_structure)
        
        # Check that captions are created for page numbers
        assert mock_st.caption.call_count == len(document_structure)

    @patch('backend.ui_components.st')
    def test_render_sidebar_structure_empty(self, mock_st):
        """Test sidebar structure rendering with empty structure."""
        from backend.ui_components import render_sidebar_structure
        
        render_sidebar_structure([])
        
        # Should display the empty state message
        mock_st.markdown.assert_called()
        call_args = mock_st.markdown.call_args[0][0]
        assert "Document structure will appear here" in call_args

    @patch('backend.ui_components.st')
    def test_render_document_info_with_info(self, mock_st):
        """Test document info rendering with info."""
        from backend.ui_components import render_document_info
        
        doc_info = {
            'name': 'test_document.pdf',
            'size': '1.5 KB',
            'type': 'application/pdf',
            'sections': 3
        }
        
        render_document_info(doc_info)
        
        # Check that document info header is displayed
        mock_st.markdown.assert_any_call("#### Document Info")
        
        # Check that document details are written
        mock_st.write.assert_any_call("**Name:** test_document.pdf")
        mock_st.write.assert_any_call("**Size:** 1.5 KB")
        mock_st.write.assert_any_call("**Type:** application/pdf")
        mock_st.write.assert_any_call("**Sections:** 3")

    @patch('backend.ui_components.st')
    def test_render_document_info_no_sections(self, mock_st):
        """Test document info rendering with no sections."""
        from backend.ui_components import render_document_info
        
        doc_info = {
            'name': 'test_document.pdf',
            'size': '1.5 KB',
            'type': 'application/pdf',
            'sections': 0
        }
        
        render_document_info(doc_info)
        
        # Should not display sections info when count is 0
        mock_st.write.assert_any_call("**Name:** test_document.pdf")
        # Check that sections line is not called
        write_calls = [call[0][0] for call in mock_st.write.call_args_list]
        assert "**Sections:** 0" not in write_calls

    @patch('backend.ui_components.st')
    def test_render_document_info_none(self, mock_st):
        """Test document info rendering with None info."""
        from backend.ui_components import render_document_info
        
        render_document_info(None)
        
        # Should not call any streamlit functions
        mock_st.markdown.assert_not_called()
        mock_st.write.assert_not_called()

    @patch('backend.ui_components.st')
    @patch('backend.ui_components.HAS_PDF_VIEWER', True)
    def test_render_control_panel_with_pdf_file(self, mock_st):
        """Test control panel rendering with PDF file."""
        from backend.ui_components import render_control_panel
        
        mock_st.button.return_value = False  # No buttons clicked
        mock_st.session_state = SessionStateMock({
            'document_structure': [{'text': 'Chapter 1'}],
            'document_pages': ['page1', 'page2']
        })
        
        render_control_panel(self.mock_file)
        
        # Check that controls header is displayed
        mock_st.markdown.assert_any_call("### ⚙️ Controls")
        
        # Check that action buttons are created
        button_calls = [call[0][0] for call in mock_st.button.call_args_list]
        assert "🔄 Reprocess Document" in button_calls
        assert "📤 Export Structure" in button_calls

    @patch('backend.ui_components.st')
    def test_render_control_panel_no_file(self, mock_st):
        """Test control panel rendering with no file."""
        from backend.ui_components import render_control_panel
        
        render_control_panel(None)
        
        # Check that info message is displayed
        mock_st.info.assert_called_with("Upload a document to access controls")

    @patch('backend.ui_components.st')
    def test_display_pdf_viewer_no_pages(self, mock_st):
        """Test PDF viewer display with no pages."""
        from backend.ui_components import display_pdf_viewer
        
        display_pdf_viewer([])
        
        # Should return early without displaying anything
        mock_st.columns.assert_not_called()

    @patch('backend.ui_components.st')
    def test_display_pdf_viewer_with_pages(self, mock_st):
        """Test PDF viewer display with pages."""
        from backend.ui_components import display_pdf_viewer
        
        # Mock session state
        mock_session_state = Mock()
        mock_session_state.current_page = 1
        mock_st.session_state = mock_session_state
        
        # Mock pages
        mock_pages = [Mock(), Mock(), Mock()]
        
        # Mock columns with context manager support
        mock_col1, mock_col2, mock_col3 = Mock(), Mock(), Mock()
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        mock_col3.__enter__ = Mock(return_value=mock_col3)
        mock_col3.__exit__ = Mock(return_value=None)
        
        mock_st.columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        # Mock button returns
        mock_st.button.return_value = False
        
        display_pdf_viewer(mock_pages)
        
        # Check that columns are created
        mock_st.columns.assert_called_once_with([1, 3, 1])
        
        # Check that image is displayed
        mock_st.image.assert_called_once()
        
        # Check that page info is written
        mock_st.write.assert_called_with("Page 2 of 3") 