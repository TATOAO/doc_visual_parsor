"""
Logging configuration for the document processing system.

This module provides centralized logging configuration for the entire application,
including structured logging with proper formatters and handlers.
"""

import logging
import logging.config
import sys
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Format the message with color
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_colors: bool = True,
    include_traceback: bool = True
) -> logging.Logger:
    """
    Setup comprehensive logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        use_colors: Whether to use colored output for console
        include_traceback: Whether to include traceback in error logs
    
    Returns:
        Configured logger instance
    """
    # Validate log level
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    level = level.upper()
    if level not in valid_levels:
        level = 'INFO'
    
    # Base configuration
    config: Dict[str, Any] = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s - %(name)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'detailed' if level == 'DEBUG' else 'simple',
                'stream': sys.stdout
            }
        },
        'loggers': {
            'doc_chunking': {
                'level': level,
                'handlers': ['console'],
                'propagate': False
            },
            'uvicorn': {
                'level': 'WARNING' if level != 'DEBUG' else 'INFO',
                'handlers': ['console'],
                'propagate': False
            },
            'uvicorn.access': {
                'level': 'WARNING' if level != 'DEBUG' else 'INFO',
                'handlers': ['console'],
                'propagate': False
            }
        },
        'root': {
            'level': level,
            'handlers': ['console']
        }
    }
    
    # Add colored formatter if requested
    if use_colors and sys.stdout.isatty():
        config['formatters']['colored'] = {
            '()': ColoredFormatter,
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
        config['handlers']['console']['formatter'] = 'colored'
    
    # Add file handler if log file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': level,
            'formatter': 'detailed',
            'filename': str(log_path),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'encoding': 'utf8'
        }
        
        # Add file handler to all loggers
        for logger_config in config['loggers'].values():
            logger_config['handlers'].append('file')
        config['root']['handlers'].append('file')
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Get main logger
    logger = logging.getLogger('doc_chunking')
    logger.info(f"Logging configured at {level} level")
    
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(f"doc_chunking.{name}")


def configure_for_production():
    """Configure logging for production environment."""
    setup_logging(
        level="INFO",
        log_file="logs/doc_chunking.log",
        use_colors=False,
        include_traceback=True
    )


def configure_for_development():
    """Configure logging for development environment."""
    setup_logging(
        level="DEBUG",
        use_colors=True,
        include_traceback=True
    ) 