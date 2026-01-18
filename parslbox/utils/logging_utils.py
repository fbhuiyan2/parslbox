import logging
import sys

# Get the root logger for the entire parslbox application
# This allows any module to get this logger by name
logger = logging.getLogger("parslbox")


def validate_log_level(log_level: str) -> int:
    """
    Convert string log level to logging constant.
    
    Args:
        log_level: String log level (case-insensitive)
        
    Returns:
        Logging level constant
        
    Raises:
        ValueError: If log level is invalid
    """
    valid_levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO, 
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    
    level_lower = log_level.lower()
    if level_lower not in valid_levels:
        raise ValueError(f"Invalid log level: {log_level}. Valid options: {', '.join(valid_levels.keys())}")
    
    return valid_levels[level_lower]

def setup_logging(log_file=None, log_level=logging.INFO):
    """
    Configures a standardized logger for the application.

    This setup ensures that every log message is immediately flushed, which is
    critical for non-interactive batch jobs.

    - It always logs to the console (stdout).
    - If a log_file path is provided, it also logs to that file.
    
    Args:
        log_file: Optional path to log file
        log_level: Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    # Prevent adding duplicate handlers if this is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(log_level)
    
    # Create a standard formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --- Console Handler (always on) ---
    # Using a StreamHandler that writes to sys.stdout.
    # Python's streams connected to terminals are typically line-buffered,
    # and batch systems handle stdout redirection efficiently.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File Handler ---
    if log_file:
        # The FileHandler will open the file and keep it open.
        # Ensure it flushes by accessing the stream.
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        
        # Immediate flushing to a file.
        # Monkey-patch the handler's stream to always flush.
        # This ensures no buffering.
        class UnbufferedStream:
            def __init__(self, stream):
                self.stream = stream
            def write(self, data):
                self.stream.write(data)
                self.stream.flush()
            def flush(self):
                self.stream.flush()

        file_handler.stream = UnbufferedStream(file_handler.stream)
        logger.addHandler(file_handler)
