import logging
import sys

# Get the root logger for the entire parslbox application
# This allows any module to get this logger by name
logger = logging.getLogger("parslbox")

def setup_logging(log_file=None):
    """
    Configures a standardized logger for the application.

    This setup ensures that every log message is immediately flushed, which is
    critical for non-interactive batch jobs.

    - It always logs to the console (stdout).
    - If a log_file path is provided, it also logs to that file.
    """
    # Prevent adding duplicate handlers if this is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    
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
