import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional
import functools
import traceback

# Default log format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
JSON_FORMAT = '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'

class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record):
        """Format log record as JSON"""
        log_entry = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'function_name'):
            log_entry['function_name'] = record.function_name

        return json.dumps(log_entry)

def setup_logging(
    level: str = None,
    format_type: str = 'standard',
    log_file: str = None,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging configuration

    Args:
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type (str): Format type ('standard' or 'json')
        log_file (str): Log file path
        max_bytes (int): Maximum log file size in bytes
        backup_count (int): Number of backup log files

    Returns:
        logging.Logger: Configured root logger
    """
    # Get log level from environment or parameter
    log_level = (level or os.getenv('LOG_LEVEL', 'INFO')).upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set root logger level
    root_logger.setLevel(numeric_level)

    # Choose formatter
    if format_type.lower() == 'json':
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(DEFAULT_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Configure AWS boto3 logging to reduce noise
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {log_level}, format: {format_type}")

    return root_logger

def get_logger(name: str, level: str = None) -> logging.Logger:
    """
    Get a configured logger instance

    Args:
        name (str): Logger name (usually __name__)
        level (str): Optional logging level override

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)

    if level:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)

    return logger

def log_function_call(func):
    """
    Decorator to log function calls with parameters and execution time

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        function_name = f"{func.__module__}.{func.__name__}"

        # Log function start
        start_time = datetime.now()
        logger.info(
            f"Function call started: {function_name}",
            extra={
                'function_name': function_name,
                'args_count': len(args),
                'kwargs_count': len(kwargs)
            }
        )

        try:
            # Execute function
            result = func(*args, **kwargs)

            # Log successful completion
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            logger.info(
                f"Function call completed: {function_name} ({execution_time:.3f}s)",
                extra={
                    'function_name': function_name,
                    'execution_time': execution_time,
                    'success': True
                }
            )

            return result

        except Exception as e:
            # Log error
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            logger.error(
                f"Function call failed: {function_name} ({execution_time:.3f}s) - {str(e)}",
                extra={
                    'function_name': function_name,
                    'execution_time': execution_time,
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
            )

            raise

    return wrapper

def log_error(logger: logging.Logger, error: Exception, context: Dict[str, Any] = None):
    """
    Log error with context information

    Args:
        logger (logging.Logger): Logger instance
        error (Exception): Exception to log
        context (Dict): Additional context information
    """
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': traceback.format_exc()
    }

    if context:
        error_info.update(context)

    logger.error(
        f"Error occurred: {error_info['error_type']} - {error_info['error_message']}",
        extra=error_info
    )

class ContextLogger:
    """Logger with context information"""

    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        """
        Initialize context logger

        Args:
            logger (logging.Logger): Base logger
            context (Dict): Context information
        """
        self.logger = logger
        self.context = context

    def _log_with_context(self, level: int, message: str, *args, **kwargs):
        """Log message with context"""
        extra = kwargs.get('extra', {})
        extra.update(self.context)
        kwargs['extra'] = extra
        self.logger.log(level, message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs):
        """Log debug message with context"""
        self._log_with_context(logging.DEBUG, message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Log info message with context"""
        self._log_with_context(logging.INFO, message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Log warning message with context"""
        self._log_with_context(logging.WARNING, message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Log error message with context"""
        self._log_with_context(logging.ERROR, message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """Log critical message with context"""
        self._log_with_context(logging.CRITICAL, message, *args, **kwargs)

def create_context_logger(name: str, context: Dict[str, Any]) -> ContextLogger:
    """
    Create a logger with context information

    Args:
        name (str): Logger name
        context (Dict): Context information

    Returns:
        ContextLogger: Context-aware logger
    """
    base_logger = get_logger(name)
    return ContextLogger(base_logger, context)

# Lambda-specific logging helpers
def setup_lambda_logging():
    """Set up logging for AWS Lambda environment"""
    # Lambda already configures basic logging
    # Just set the level and format
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Reduce AWS SDK noise
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    return root_logger

def get_lambda_logger(name: str, request_id: str = None) -> ContextLogger:
    """
    Get a Lambda-specific logger with request context

    Args:
        name (str): Logger name
        request_id (str): Lambda request ID

    Returns:
        ContextLogger: Lambda context logger
    """
    context = {}

    if request_id:
        context['request_id'] = request_id

    # Add Lambda-specific context from environment
    if os.getenv('AWS_LAMBDA_FUNCTION_NAME'):
        context['function_name'] = os.getenv('AWS_LAMBDA_FUNCTION_NAME')
        context['function_version'] = os.getenv('AWS_LAMBDA_FUNCTION_VERSION')

    return create_context_logger(name, context)

# Performance logging
def log_performance(operation: str, start_time: datetime, end_time: datetime = None, **kwargs):
    """
    Log performance metrics

    Args:
        operation (str): Operation name
        start_time (datetime): Operation start time
        end_time (datetime): Operation end time (defaults to now)
        **kwargs: Additional metrics
    """
    if end_time is None:
        end_time = datetime.now()

    duration = (end_time - start_time).total_seconds()

    logger = get_logger('performance')

    metrics = {
        'operation': operation,
        'duration_seconds': duration,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat()
    }
    metrics.update(kwargs)

    logger.info(f"Performance: {operation} completed in {duration:.3f}s", extra=metrics)

# Initialize default logging if this module is imported
if not logging.getLogger().handlers:
    setup_logging()