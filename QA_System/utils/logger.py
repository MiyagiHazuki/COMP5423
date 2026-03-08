import os
import datetime
import sys

# Log file directory
LOG_DIR = "log"

# Log levels
LOG_LEVEL_DEBUG = 10
LOG_LEVEL_INFO = 20
LOG_LEVEL_WARNING = 30
LOG_LEVEL_ERROR = 40
LOG_LEVEL_CRITICAL = 50

# Current log level
CURRENT_LOG_LEVEL = LOG_LEVEL_INFO

# Log color codes
COLORS = {
    "RESET": "\033[0m",
    "DEBUG": "\033[36m",    # Cyan
    "INFO": "\033[32m",     # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",    # Red
    "CRITICAL": "\033[35m", # Purple
    "BOLD": "\033[1m",      # Bold
}

# Level mapping
LEVEL_NAMES = {
    LOG_LEVEL_DEBUG: "DEBUG",
    LOG_LEVEL_INFO: "INFO",
    LOG_LEVEL_WARNING: "WARNING",
    LOG_LEVEL_ERROR: "ERROR",
    LOG_LEVEL_CRITICAL: "CRITICAL",
}

def ensure_log_dir_exists():
    """Ensure log directory exists"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

def get_log_filename():
    """Get log filename based on current time"""
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(LOG_DIR, f"{timestamp}.txt")

# Global variable to store current log file path
current_log_file = None

def initialize_logger():
    """Initialize logger, create a new log file"""
    global current_log_file
    ensure_log_dir_exists()
    current_log_file = get_log_filename()
    
    # Create log file and write initial information
    with open(current_log_file, "w", encoding="utf-8") as f:
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"=== Log started at {start_time} ===\n")
    
    return current_log_file

def set_log_level(level):
    """Set log level
    
    Parameters:
        level: Log level, can be integer or string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    global CURRENT_LOG_LEVEL
    
    if isinstance(level, str):
        level = level.upper()
        if level == "DEBUG":
            CURRENT_LOG_LEVEL = LOG_LEVEL_DEBUG
        elif level == "INFO":
            CURRENT_LOG_LEVEL = LOG_LEVEL_INFO
        elif level == "WARNING":
            CURRENT_LOG_LEVEL = LOG_LEVEL_WARNING
        elif level == "ERROR":
            CURRENT_LOG_LEVEL = LOG_LEVEL_ERROR
        elif level == "CRITICAL":
            CURRENT_LOG_LEVEL = LOG_LEVEL_CRITICAL
        else:
            raise ValueError(f"Unknown log level: {level}")
    else:
        CURRENT_LOG_LEVEL = level
    
    log_message(f"Log level set to: {LEVEL_NAMES.get(CURRENT_LOG_LEVEL, str(CURRENT_LOG_LEVEL))}", LOG_LEVEL_INFO)

def should_log(level):
    """Determine whether to record logs at this level"""
    return level >= CURRENT_LOG_LEVEL

def log_message(message, level=LOG_LEVEL_INFO):
    """Write message to log file
    
    Parameters:
        message: Log message
        level: Log level (default is INFO)
    """
    global current_log_file
    
    # Check if this level should be logged
    if not should_log(level):
        return
    
    # Initialize log file if not initialized
    if current_log_file is None:
        initialize_logger()
    
    # Get current time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get level name
    level_name = LEVEL_NAMES.get(level, str(level))
    
    # Write message to log file (log file does not include color codes)
    with open(current_log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{level_name}] {message}\n")
    
    # Add color if output to terminal
    if sys.stdout.isatty():
        color = COLORS.get(level_name, COLORS["RESET"])
        print(f"{color}[{timestamp}] [{level_name}] {message}{COLORS['RESET']}")

def debug(message):
    """Record DEBUG level log"""
    log_message(message, LOG_LEVEL_DEBUG)

def info(message):
    """Record INFO level log"""
    log_message(message, LOG_LEVEL_INFO)

def warning(message):
    """Record WARNING level log"""
    log_message(message, LOG_LEVEL_WARNING)

def error(message):
    """Record ERROR level log"""
    log_message(message, LOG_LEVEL_ERROR)

def critical(message):
    """Record CRITICAL level log"""
    log_message(message, LOG_LEVEL_CRITICAL)

# For backward compatibility, keep original log_message function behavior
def log_message_compat(message, level=None):
    """Compatible implementation of the original log_message function, supports optional level parameter"""
    # If no level provided, use INFO level
    if level is None:
        level = LOG_LEVEL_INFO
    # Call the new log_message function
    log_message_original(message, level)

# Save reference to original function
log_message_original = log_message

# Override original log_message function to maintain complete backward compatibility
log_message = log_message_compat 