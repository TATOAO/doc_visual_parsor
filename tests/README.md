# Backend Tests Documentation

## Overview

This directory contains comprehensive unit tests for all backend modules of the Document Visual Parser application. The tests are designed to ensure code quality, reliability, and maintainability.

## Test Structure

### Test Files

- `test_pdf_processor.py` - Tests for PDF processing functionality
- `test_docx_processor.py` - Tests for DOCX processing functionality  
- `test_document_analyzer.py` - Tests for document structure analysis
- `test_session_manager.py` - Tests for session state management
- `test_ui_components.py` - Tests for UI component functions

### Testing Framework

- **pytest** - Main testing framework
- **unittest.mock** - Mocking framework for isolating units under test
- **pytest-cov** - Coverage reporting
- **pytest-mock** - Additional mocking utilities

## Running Tests

### Quick Start

1. **Install test dependencies:**
   ```bash
   python run_tests.py --install-deps
   ```

2. **Run all tests:**
   ```bash
   python run_tests.py
   ```

3. **Run tests with coverage:**
   ```bash
   python run_tests.py --coverage
   ```

### Advanced Usage

#### Run specific test files:
```bash
# Run only PDF processor tests
python run_tests.py --test tests/test_pdf_processor.py

# Run only session manager tests
python run_tests.py --test tests/test_session_manager.py
```

#### Run specific test methods:
```bash
# Run specific test method
python run_tests.py --test tests/test_pdf_processor.py::TestPDFProcessor::test_extract_pdf_pages_success
```

#### Run tests in parallel:
```bash
python run_tests.py --parallel
```

#### Run with verbose output:
```bash
python run_tests.py --verbose
```

#### Combine options:
```bash
python run_tests.py --coverage --verbose --parallel
```

### Direct pytest Usage

You can also run pytest directly:

```bash
# Basic test run
pytest tests/

# With coverage
pytest tests/ --cov=backend --cov-report=html

# Specific test file
pytest tests/test_pdf_processor.py -v

# Run tests matching a pattern
pytest tests/ -k "test_extract"
```

## Test Coverage

The tests aim for comprehensive coverage of all backend functionality:

### PDF Processor (`test_pdf_processor.py`)
- ✅ PDF page extraction (success and error cases)
- ✅ PDF document object creation
- ✅ PDF document closing
- ✅ Error handling and cleanup

### DOCX Processor (`test_docx_processor.py`)
- ✅ DOCX content extraction
- ✅ DOCX structure analysis based on heading styles
- ✅ Different heading level detection
- ✅ Empty document handling
- ✅ Error cases

### Document Analyzer (`test_document_analyzer.py`)
- ✅ PDF structure extraction with font analysis
- ✅ Heading pattern recognition
- ✅ Universal document structure analysis
- ✅ Structure summary generation
- ✅ Duplicate removal
- ✅ Error handling

### Session Manager (`test_session_manager.py`)
- ✅ Session state initialization
- ✅ Document state reset
- ✅ New file detection
- ✅ Navigation functionality
- ✅ Document info retrieval

### UI Components (`test_ui_components.py`)
- ✅ PDF viewer availability check
- ✅ DOCX content display
- ✅ Upload area rendering
- ✅ Sidebar structure rendering
- ✅ Document info display
- ✅ Control panel rendering

## Test Design Patterns

### Mocking Strategy

The tests extensively use mocking to isolate units under test:

1. **Streamlit Functions**: All `st.*` functions are mocked to avoid UI dependencies
2. **File I/O**: File operations are mocked using `tempfile` and `Mock` objects
3. **External Libraries**: PyMuPDF and python-docx are mocked for controlled testing
4. **Session State**: Streamlit session state is mocked for state management tests

### Test Organization

Each test class follows a consistent pattern:

```python
class TestModuleName:
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Initialize common test data
        
    def test_function_name_success(self):
        """Test successful operation."""
        # Test happy path
        
    def test_function_name_error(self):
        """Test error handling."""
        # Test error conditions
        
    def test_function_name_edge_case(self):
        """Test edge cases."""
        # Test boundary conditions
```

### Assertions

Tests use descriptive assertions and check multiple aspects:

- Return values and types
- Function call counts and arguments
- State changes
- Error handling
- Resource cleanup

## Coverage Goals

The test suite aims for:

- **80%+ overall coverage** (enforced by default)
- **90%+ function coverage** for critical functions
- **100% coverage** for error handling paths

### Viewing Coverage Reports

After running tests with coverage:

```bash
python run_tests.py --coverage
```

Coverage reports are generated in multiple formats:

1. **Terminal output**: Shows coverage summary
2. **HTML report**: Detailed report in `htmlcov/` directory
   - Open `htmlcov/index.html` in browser for interactive view

## Adding New Tests

When adding new backend functions, follow these guidelines:

### 1. Test File Structure

Create tests in the appropriate test file:
- PDF functionality → `test_pdf_processor.py`
- DOCX functionality → `test_docx_processor.py`
- Analysis functions → `test_document_analyzer.py`
- Session management → `test_session_manager.py`
- UI components → `test_ui_components.py`

### 2. Test Method Naming

Use descriptive test method names:
```python
def test_function_name_condition_expected_result(self):
    """Test function_name when condition should result in expected_result."""
```

### 3. Required Test Cases

For each function, include tests for:
- **Success case**: Normal operation with valid inputs
- **Error cases**: Invalid inputs, exceptions, failures
- **Edge cases**: Boundary values, empty inputs, None values
- **State changes**: If function modifies state

### 4. Mock Strategy

Mock external dependencies consistently:
```python
@patch('module.external_dependency')
@patch('module.st')  # Always mock streamlit
def test_function(self, mock_st, mock_dependency):
    # Test implementation
```

## Continuous Integration

For CI/CD integration, use:

```bash
# Run tests with strict coverage requirement
python run_tests.py --coverage

# Exit codes:
# 0 = All tests pass
# 1 = Tests failed or coverage below threshold
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running tests from project root
2. **Mock Issues**: Check that all external dependencies are properly mocked
3. **Coverage Issues**: Run with `--verbose` to see which lines aren't covered

### Debug Test Failures

```bash
# Run with verbose output and no coverage
python run_tests.py --verbose --test tests/failing_test.py

# Run single test method
python run_tests.py --test tests/test_file.py::TestClass::test_method
```

## Best Practices

1. **Keep tests focused**: One test should verify one specific behavior
2. **Use descriptive names**: Test names should explain what is being tested
3. **Mock external dependencies**: Don't rely on external services or files
4. **Test both success and failure paths**: Include error handling tests
5. **Keep setup minimal**: Only create what's needed for each test
6. **Use fixtures for complex setup**: Share common test data across tests
7. **Verify all side effects**: Check function calls, state changes, etc.

## Performance

The test suite is designed to run quickly:

- Most tests complete in milliseconds
- Parallel execution available with `--parallel` flag
- Mocking eliminates I/O bottlenecks
- Focused unit tests avoid complex integration scenarios 