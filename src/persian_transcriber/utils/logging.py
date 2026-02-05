"""
Logging configuration for Persian Transcriber.

This module provides a centralized logging setup with support for
both console and file output, colored formatting, and configurable
log levels. Configuration is read from config.yaml if available.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


# Default log format
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Shorter format for console
CONSOLE_FORMAT = "%(levelname)-8s | %(message)s"

# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    "level": "INFO",
    "log_file_path": None,
    "format": DEFAULT_FORMAT,
    "date_format": DEFAULT_DATE_FORMAT,
    "use_colors": True,
    "colors": {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red",
    },
}

# ANSI color codes
ANSI_COLORS = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "reset": "\033[0m",
    "bold": "\033[1m",
}

# Module-level logger cache
_loggers: Dict[str, logging.Logger] = {}
_config_cache: Optional[Dict[str, Any]] = None
_initialized: bool = False


def _load_config() -> Dict[str, Any]:
    """Load logging configuration from config.yaml."""
    global _config_cache

    if _config_cache is not None:
        return _config_cache

    config = DEFAULT_CONFIG.copy()

    # Search for config.yaml in common locations
    search_paths = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "config.yml",
        Path(__file__).parent.parent.parent.parent / "config.yaml",
        Path(__file__).parent.parent.parent.parent / "config.yml",
        Path.home() / ".persian_transcriber" / "config.yaml",
    ]

    # Also check environment variable
    if os.environ.get("PERSIAN_TRANSCRIBER_CONFIG"):
        search_paths.insert(0, Path(os.environ["PERSIAN_TRANSCRIBER_CONFIG"]))

    config_path = None
    for path in search_paths:
        if path.exists():
            config_path = path
            break

    if config_path is not None:
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)

            if yaml_config and "logging" in yaml_config:
                logging_config = yaml_config["logging"]
                for key in DEFAULT_CONFIG:
                    if key in logging_config:
                        config[key] = logging_config[key]
        except ImportError:
            # PyYAML not installed, use defaults
            pass
        except Exception:
            # Config file error, use defaults
            pass

    _config_cache = config
    return config


class ColoredFormatter(logging.Formatter):
    """
    Logging formatter that adds colors to log levels for console output.

    Colors are configurable via config.yaml:
    - DEBUG: Cyan (default)
    - INFO: Green (default)
    - WARNING: Yellow (default)
    - ERROR: Red (default)
    - CRITICAL: Bold Red (default)
    """

    # Legacy ANSI color codes (used if config doesn't specify)
    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: bool = True,
        color_config: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize the colored formatter.

        Args:
            fmt: Log message format string.
            datefmt: Date format string.
            use_colors: Whether to use colors (disable for non-TTY output).
            color_config: Custom color configuration from config.yaml.
        """
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_colors = use_colors and self._supports_color()
        self.color_config = color_config or {}

    @staticmethod
    def _supports_color() -> bool:
        """Check if the terminal supports ANSI colors."""
        if not hasattr(sys.stderr, "isatty") or not sys.stderr.isatty():
            return False

        if sys.platform != "win32":
            return True

        # Windows-specific color support check
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False

    def _get_color_code(self, level_name: str, level_no: int) -> str:
        """Get ANSI color code for a log level."""
        # Try config-based colors first
        if level_name in self.color_config:
            color_spec = self.color_config[level_name]
            codes = []
            for part in color_spec.split(","):
                part = part.strip().lower()
                if part in ANSI_COLORS:
                    codes.append(ANSI_COLORS[part])
            if codes:
                return "".join(codes)

        # Fall back to legacy colors
        return self.COLORS.get(level_no, "")

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with colors.

        Args:
            record: The log record to format.

        Returns:
            str: Formatted log message with optional colors.
        """
        if self.use_colors:
            color = self._get_color_code(record.levelname, record.levelno)
            reset = ANSI_COLORS["reset"]
            record.levelname = f"{color}{record.levelname}{reset}"

        return super().format(record)


def setup_logging(
    level: Optional[Union[int, str]] = None,
    log_file: Optional[Path] = None,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    use_colors: Optional[bool] = None,
    quiet: bool = False,
    verbose: bool = False,
) -> logging.Logger:
    """
    Configure logging for the Persian Transcriber package.

    Settings are loaded from config.yaml and can be overridden via parameters.

    Sets up logging with:
    - Console output with optional colors
    - Optional file output
    - Configurable log level and format

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG, or string like "INFO").
        log_file: Optional path to log file. If provided, logs will also
                  be written to this file.
        log_format: Custom log format string. Uses config or DEFAULT_FORMAT.
        date_format: Custom date format string. Uses config or DEFAULT_DATE_FORMAT.
        use_colors: Whether to use colored output in console.
        quiet: If True, only show warnings and errors in console.
        verbose: If True, set level to DEBUG.

    Returns:
        logging.Logger: The root logger for the package.

    Example:
        >>> from persian_transcriber.utils.logging import setup_logging
        >>> import logging
        >>> logger = setup_logging(level=logging.DEBUG)
        >>> logger.info("Transcription started")
    """
    global _initialized

    # Load configuration from config.yaml
    config = _load_config()

    # Determine log level
    if verbose:
        level = logging.DEBUG
    elif level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
    else:
        level = getattr(logging, config["level"].upper(), logging.INFO)

    # Get other settings from config or parameters
    if log_file is None and config.get("log_file_path"):
        log_file_str = config["log_file_path"]
        # Expand placeholders
        now = datetime.now()
        log_file_str = log_file_str.replace("{date}", now.strftime("%Y-%m-%d"))
        log_file_str = log_file_str.replace("{time}", now.strftime("%H-%M-%S"))
        log_file = Path(log_file_str)

    if use_colors is None:
        use_colors = config.get("use_colors", True)

    if log_format is None:
        log_format = config.get("format", DEFAULT_FORMAT)

    if date_format is None:
        date_format = config.get("date_format", DEFAULT_DATE_FORMAT)

    color_config = config.get("colors", {})

    # Get the package root logger
    root_logger = logging.getLogger("persian_transcriber")

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Set the log level
    root_logger.setLevel(level)

    # Create console handler
    if not quiet:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)

        console_formatter = ColoredFormatter(
            fmt=CONSOLE_FORMAT,
            datefmt=date_format,
            use_colors=use_colors,
            color_config=color_config,
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_file,
            mode="a",
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file

        file_formatter = logging.Formatter(
            fmt=log_format,
            datefmt=date_format,
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Prevent propagation to root logger
    root_logger.propagate = False

    _initialized = True

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    This function returns a logger that is a child of the package logger,
    ensuring consistent formatting and configuration.

    Args:
        name: Name of the logger (typically __name__ of the module).

    Returns:
        logging.Logger: A logger instance for the specified module.

    Example:
        >>> from persian_transcriber.utils.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing audio file")
    """
    global _initialized

    # Auto-initialize with config if not already done
    if not _initialized:
        setup_logging()

    # Handle both full module paths and short names
    if not name.startswith("persian_transcriber"):
        name = f"persian_transcriber.{name}"

    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)

    return _loggers[name]


def set_log_level(level: Union[int, str]) -> None:
    """
    Set the log level for all persian_transcriber loggers.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO, or "DEBUG").

    Example:
        >>> import logging
        >>> from persian_transcriber.utils.logging import set_log_level
        >>> set_log_level(logging.DEBUG)  # Enable debug output
        >>> set_log_level("WARNING")  # Or use string
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger("persian_transcriber")
    root_logger.setLevel(level)

    for handler in root_logger.handlers:
        handler.setLevel(level)


def disable_logging() -> None:
    """
    Disable all logging output from persian_transcriber.

    Useful for library usage where you want to suppress all output.

    Example:
        >>> from persian_transcriber.utils.logging import disable_logging
        >>> disable_logging()
    """
    logging.getLogger("persian_transcriber").disabled = True


def enable_logging() -> None:
    """
    Re-enable logging output from persian_transcriber.

    Example:
        >>> from persian_transcriber.utils.logging import enable_logging
        >>> enable_logging()
    """
    logging.getLogger("persian_transcriber").disabled = False
