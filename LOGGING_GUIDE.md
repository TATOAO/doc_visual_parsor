# Logging Configuration Guide

This guide explains how to properly configure and use logging levels in the Document Visual Parser project.

## Overview

The project now uses a centralized logging configuration that supports:
- Environment-based log levels
- Consistent formatting across modules
- Proper separation of development and production logging
- Easy configuration through environment variables

## Logging Levels

### When to Use Each Level

| Level | When to Use | Examples |
|-------|-------------|----------|
| **DEBUG** | Detailed diagnostic info for development | Variable values, function entry/exit, detailed flow |
| **INFO** | General application flow and important events | Server startup, processing milestones, user actions |
| **WARNING** | Something unexpected but handled gracefully | Deprecated features, recoverable errors, fallbacks |
| **ERROR** | Serious problems that need attention | File processing failures, database errors |
| **CRITICAL** | Very serious errors that may stop the application | System failures, security issues |

## Configuration

### Setting Log Levels

#### Via Environment Variables
```bash
# Development (verbose logging)
export LOG_LEVEL=DEBUG
export ENVIRONMENT=development

# Production (minimal logging)
export LOG_LEVEL=INFO
export ENVIRONMENT=production

# Debugging specific issues
export LOG_LEVEL=WARNING
```

#### Via Command Line
```bash
# Development mode with debug logging
LOG_LEVEL=DEBUG python run_backend.py

# Production mode with info logging
LOG_LEVEL=INFO ENVIRONMENT=production python run_backend.py
```

### Recommended Settings by Environment

#### Development
```bash
LOG_LEVEL=DEBUG
ENVIRONMENT=development
```
- Shows all log messages
- Enables auto-reload
- Detailed error traces

#### Staging/Testing
```bash
LOG_LEVEL=INFO
ENVIRONMENT=staging
```
- Important events and errors
- Performance monitoring
- User action tracking

#### Production
```bash
LOG_LEVEL=WARNING
ENVIRONMENT=production
```
- Only warnings, errors, and critical issues
- Minimal performance impact
- Focus on actionable problems

## Usage in Your Code

### Setting Up Logging in a Module

```python
# At the top of your module
from backend.logging_config import get_backend_logger

# Initialize logger (use module name)
logger = get_backend_logger("your_module_name")
```

### Logging Examples

```python
# DEBUG: Detailed diagnostic information
logger.debug(f"Processing page {page_num} with size {img.size}")
logger.debug(f"Function parameters: file={filename}, options={options}")

# INFO: Important application events
logger.info(f"Successfully processed PDF with {page_count} pages")
logger.info(f"Server started on port {port}")
logger.info(f"User uploaded file: {filename}")

# WARNING: Handled problems that should be noted
logger.warning(f"Could not process page {page_num}, skipping")
logger.warning(f"Using fallback method for file type detection")
logger.warning(f"Temporary file cleanup failed: {cleanup_error}")

# ERROR: Serious problems that prevent normal operation
logger.error(f"Failed to open PDF file: {str(e)}")
logger.error(f"Database connection failed: {str(e)}")
logger.error(f"Invalid file format: {filename}")

# ERROR with exception details (for debugging)
logger.error(f"Processing failed: {str(e)}", exc_info=True)

# CRITICAL: Very serious problems
logger.critical(f"Server cannot start: {str(e)}")
logger.critical(f"Security violation detected: {details}")
```

### Structured Logging

For better log analysis, include relevant context:

```python
# Good: Include context
logger.info(f"Document processed successfully", extra={
    'filename': filename,
    'pages': page_count,
    'processing_time': elapsed_time
})

# Better: Use consistent format
logger.info(f"Document processed | file={filename} | pages={page_count} | time={elapsed_time}s")
```

## Migration from Print Statements

### Before (using print)
```python
print(f"Error processing page {page_num}: {error}")
print(f"Warning: Could not clean up temp file")
print(f"Processing complete for {filename}")
```

### After (using proper logging)
```python
logger.error(f"Error processing page {page_num}: {error}")
logger.warning(f"Could not clean up temp file")
logger.info(f"Processing complete for {filename}")
```

## File Logging (Production)

For production environments, you can enable file logging:

```python
from backend.logging_config import setup_file_logging

# Enable file logging
setup_file_logging("logs/app.log", level="INFO")
```

This will:
- Create a `logs/` directory
- Write logs to `logs/app.log`
- Include function names and line numbers
- Rotate logs automatically (configure as needed)

## Best Practices

### 1. Use Appropriate Levels
```python
# ❌ Wrong: Everything as INFO
logger.info(f"Variable x = {x}")  # Should be DEBUG
logger.info(f"File not found")    # Should be ERROR

# ✅ Correct: Appropriate levels
logger.debug(f"Variable x = {x}")
logger.error(f"File not found: {filename}")
```

### 2. Include Context
```python
# ❌ Vague
logger.error("Processing failed")

# ✅ Specific
logger.error(f"PDF processing failed for {filename}: {str(e)}")
```

### 3. Use String Formatting Efficiently
```python
# ❌ Inefficient (always evaluates)
logger.debug("Complex calculation: " + str(expensive_function()))

# ✅ Efficient (only evaluates if DEBUG level is active)
logger.debug("Complex calculation: %s", expensive_function())
```

### 4. Don't Log Sensitive Information
```python
# ❌ Security risk
logger.info(f"User login: {username}:{password}")

# ✅ Safe
logger.info(f"User login: {username}")
```

## Testing Logging

### Verify Your Logging Setup
```bash
# Test different log levels
LOG_LEVEL=DEBUG python -c "
from backend.logging_config import get_backend_logger
logger = get_backend_logger('test')
logger.debug('Debug message')
logger.info('Info message')
logger.warning('Warning message')
logger.error('Error message')
"
```

### Expected Output at Different Levels

#### DEBUG Level (shows everything)
```
2024-01-15 10:30:45 - backend.test - DEBUG - Debug message
2024-01-15 10:30:45 - backend.test - INFO - Info message
2024-01-15 10:30:45 - backend.test - WARNING - Warning message
2024-01-15 10:30:45 - backend.test - ERROR - Error message
```

#### INFO Level (no debug)
```
2024-01-15 10:30:45 - backend.test - INFO - Info message
2024-01-15 10:30:45 - backend.test - WARNING - Warning message
2024-01-15 10:30:45 - backend.test - ERROR - Error message
```

#### WARNING Level (only warnings and errors)
```
2024-01-15 10:30:45 - backend.test - WARNING - Warning message
2024-01-15 10:30:45 - backend.test - ERROR - Error message
```

## Quick Reference

| Command | Purpose |
|---------|---------|
| `LOG_LEVEL=DEBUG python run_backend.py` | Development with full logging |
| `LOG_LEVEL=INFO python run_backend.py` | Standard production logging |
| `LOG_LEVEL=WARNING python run_backend.py` | Minimal production logging |
| `LOG_LEVEL=ERROR python run_backend.py` | Only errors and critical issues |

## Next Steps

1. **Update other backend modules** to use the new logging system (replace print statements)
2. **Set appropriate log levels** for your deployment environment
3. **Monitor logs** to ensure they provide useful information without being too verbose
4. **Consider log aggregation** tools like ELK stack or Grafana for production monitoring 