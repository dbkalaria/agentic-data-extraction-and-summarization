"""
logging_config.py
-------------------

This module provides a centralized logging configuration for the entire project.

It sets up a single logger instance that can be imported and used by any other
module. This ensures that all log messages have a consistent format and destination.

The logger is configured to output to both the console (INFO level) and a
dedicated log file (`project.log`, DEBUG level).

Usage (in other modules):
-------------------------
from logging_config import logger

logger.info("This is an informational message.")
logger.error("This is an error message.")
"""

import logging
import sys

# 1. Create a logger instance
logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)  # Set the lowest level to capture all messages

# 2. Create handlers
# Console handler (prints to stdout)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)  # Only show INFO and above on the console

# File handler (writes to a file)
file_handler = logging.FileHandler("project.log")
file_handler.setLevel(logging.DEBUG)  # Log everything to the file

# 3. Create a formatter and set it for both handlers
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 4. Add the handlers to the logger
# Avoid adding handlers if they already exist (e.g., in interactive environments)
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
