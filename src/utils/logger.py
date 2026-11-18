# src/utils/logger.py

import logging
import os
from logging.handlers import RotatingFileHandler
from ..config import LOG_CONFIG, LOG_DIR

def setup_logger(name='SPIN-Net'):
    """Set up and return a logger instance."""
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers from the logger.
    if logger.handlers:
        logger.handlers.clear()

    # File handler.
    fh = RotatingFileHandler(
        LOG_CONFIG['handlers']['file']['filename'],
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    fh.setFormatter(logging.Formatter(LOG_CONFIG['formatters']['standard']['format']))
    logger.addHandler(fh)

    # Console handler.
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_CONFIG['formatters']['standard']['format']))
    logger.addHandler(ch)

    return logger

class Logger:
    def __init__(self, name='SPIN-Net'):
        self.logger = setup_logger(name)

    def info(self, msg):
        self.logger.info(msg)

    def error(self, msg):
        self.logger.error(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def debug(self, msg):
        self.logger.debug(msg)
