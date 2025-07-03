#!/usr/bin/env python3
"""
Demonstration script for the logging configuration system.
Run this with different LOG_LEVEL environment variables to see the differences.

Examples:
    LOG_LEVEL=DEBUG python demo_logging.py
    LOG_LEVEL=INFO python demo_logging.py
    LOG_LEVEL=WARNING python demo_logging.py
    LOG_LEVEL=ERROR python demo_logging.py
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.logging_config import get_backend_logger, LogLevel

def demonstrate_logging_levels():
    """Demonstrate different logging levels"""
    logger = get_backend_logger("demo")
    
    print(f"🔧 Current LOG_LEVEL: {os.getenv('LOG_LEVEL', 'INFO')}")
    print(f"📊 The following messages will be shown based on your log level:")
    print("-" * 60)
    
    # DEBUG level messages
    logger.debug("🔍 DEBUG: Detailed diagnostic information")
    logger.debug("🔍 DEBUG: Variable values, function entry/exit")
    logger.debug("🔍 DEBUG: Step-by-step processing details")
    
    # INFO level messages  
    logger.info("ℹ️  INFO: Application started successfully")
    logger.info("ℹ️  INFO: Processing document with 5 pages")
    logger.info("ℹ️  INFO: User uploaded file: example.pdf")
    
    # WARNING level messages
    logger.warning("⚠️  WARNING: Could not process page 3, skipping")
    logger.warning("⚠️  WARNING: Using fallback method for file detection")
    logger.warning("⚠️  WARNING: Temporary file cleanup failed")
    
    # ERROR level messages
    logger.error("❌ ERROR: Failed to open PDF file")
    logger.error("❌ ERROR: Database connection failed")
    logger.error("❌ ERROR: Invalid file format detected")
    
    # CRITICAL level messages
    logger.critical("🚨 CRITICAL: Server cannot start")
    logger.critical("🚨 CRITICAL: Security violation detected")
    
    print("-" * 60)
    print("✅ Logging demonstration complete!")

def demonstrate_different_scenarios():
    """Demonstrate logging in different scenarios"""
    logger = get_backend_logger("scenarios")
    
    print("\n📝 Scenario Examples:")
    print("-" * 60)
    
    # Scenario 1: Processing a document
    filename = "sample_document.pdf"
    try:
        logger.info(f"Starting to process document: {filename}")
        logger.debug(f"File size: 2.5MB, Pages: 10")
        
        # Simulate processing pages
        for page in range(1, 4):
            logger.debug(f"Processing page {page}")
            if page == 2:
                logger.warning(f"Page {page} has low quality, using enhanced processing")
        
        logger.info(f"Successfully processed {filename} with 3 pages")
        
    except Exception as e:
        logger.error(f"Failed to process {filename}: {str(e)}")
    
    # Scenario 2: API endpoint handling
    logger.info("Handling API request: POST /api/upload-document")
    logger.debug("Request payload size: 1.2MB")
    logger.debug("Content-Type: application/pdf")
    
    # Scenario 3: System monitoring
    logger.info("System health check completed")
    logger.debug("Memory usage: 512MB, CPU: 45%")
    
    print("-" * 60)

def show_log_level_comparison():
    """Show what messages appear at different log levels"""
    print("\n📊 Log Level Comparison:")
    print("-" * 60)
    
    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    messages = {
        "DEBUG": "🔍 Detailed diagnostic info",
        "INFO": "ℹ️  General application events", 
        "WARNING": "⚠️  Unexpected but handled issues",
        "ERROR": "❌ Serious problems",
        "CRITICAL": "🚨 System-threatening issues"
    }
    
    current_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    print(f"Current level: {current_level}")
    print(f"Messages shown at {current_level} level:")
    
    # Find the index of current level
    try:
        current_index = levels.index(current_level)
    except ValueError:
        current_index = 1  # Default to INFO
    
    for i, level in enumerate(levels):
        if i >= current_index:
            print(f"  ✅ {level}: {messages[level]}")
        else:
            print(f"  ❌ {level}: {messages[level]} (hidden)")
    
    print("-" * 60)

if __name__ == "__main__":
    print("🎯 Logging System Demonstration")
    print("=" * 60)
    
    show_log_level_comparison()
    demonstrate_logging_levels()
    demonstrate_different_scenarios()
    
    print(f"\n💡 Tips:")
    print(f"  • Run with different LOG_LEVEL values: DEBUG, INFO, WARNING, ERROR")
    print(f"  • Example: LOG_LEVEL=DEBUG python {__file__}")
    print(f"  • Example: LOG_LEVEL=WARNING python {__file__}")
    print(f"  • Current level filters out lower-priority messages") 