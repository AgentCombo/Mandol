"""Utilities for logging config."""

import logging
import logging.handlers
import os
import sys
import json
import copy
import queue
import atexit
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

# Avoid mutating LogRecord fields before other handlers process the record.
try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


class SafeColoredFormatter(logging.Formatter):
    
    # Avoid mutating LogRecord fields before other handlers process the record.
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        original_levelname = record.levelname
        
        try:
            # Avoid mutating LogRecord fields before other handlers process the record.
            if record.levelname in self.COLORS:
                record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            
            result = super().format(record)
            return result
            
        finally:
            # Avoid mutating LogRecord fields before other handlers process the record.
            # Avoid mutating LogRecord fields before other handlers process the record.
            record.levelname = original_levelname


class JsonFormatter(logging.Formatter):
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)


class MemorySystemLogger:
    
    def __init__(self, name: str = "AgentMemorySystem"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.handlers = {}
        self._is_configured = False
        # Avoid mutating LogRecord fields before other handlers process the record.
        self.log_queue: Optional[queue.Queue] = None
        self.queue_listener: Optional[logging.handlers.QueueListener] = None
    
    def setup(self, 
              level: int = logging.INFO,
              console_output: bool = True,
              file_output: bool = True,
              log_dir: Optional[str] = None,
              max_file_size: int = 1024 * 1024 * 1024,  # 1GB
              backup_count: int = 5,
              format_style: str = "detailed",
              use_color: Optional[bool] = None,
              async_logging: bool = True) -> logging.Logger:
        """Run setup."""
        if self._is_configured:
            pass
        
        # Avoid mutating LogRecord fields before other handlers process the record.
        self.logger.propagate = False
        
        # Avoid mutating LogRecord fields before other handlers process the record.
        self.logger.setLevel(level)
        
        self._shutdown_queue_listener()
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            self.handlers.clear()
        
        formatter = self._get_formatter(format_style)
        
        # Avoid mutating LogRecord fields before other handlers process the record.
        actual_handlers: List[logging.Handler] = []
        
        # Avoid mutating LogRecord fields before other handlers process the record.
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            
            # Avoid mutating LogRecord fields before other handlers process the record.
            if use_color is None:
                use_color = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
            
            colored_formatter = self._get_colored_formatter(format_style, use_color=use_color)
            console_handler.setFormatter(colored_formatter)
            
            actual_handlers.append(console_handler)
            self.handlers['console'] = console_handler
        
        # Avoid mutating LogRecord fields before other handlers process the record.
        if file_output:
            log_dir = log_dir or self._get_default_log_dir()
            try:
                os.makedirs(log_dir, exist_ok=True)
                
                # Avoid mutating LogRecord fields before other handlers process the record.
                log_file = os.path.join(log_dir, f"{self.name.lower()}.log")
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file, 
                    maxBytes=max_file_size, 
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                actual_handlers.append(file_handler)
                self.handlers['file'] = file_handler
                
                # Avoid mutating LogRecord fields before other handlers process the record.
                error_log_file = os.path.join(log_dir, f"{self.name.lower()}_error.log")
                error_handler = logging.handlers.RotatingFileHandler(
                    error_log_file,
                    maxBytes=max_file_size,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                error_handler.setLevel(logging.ERROR)
                error_handler.setFormatter(formatter)
                actual_handlers.append(error_handler)
                self.handlers['error'] = error_handler
                
            except Exception as e:
                sys.stderr.write(f"Failed to create log file: {e}\n")
        
        # Avoid mutating LogRecord fields before other handlers process the record.
        if async_logging and actual_handlers:
            # Avoid mutating LogRecord fields before other handlers process the record.
            self.log_queue = queue.Queue(-1)
            
            queue_handler = logging.handlers.QueueHandler(self.log_queue)
            queue_handler.setLevel(level)
            self.logger.addHandler(queue_handler)
            
            # Avoid mutating LogRecord fields before other handlers process the record.
            # Avoid mutating LogRecord fields before other handlers process the record.
            self.queue_listener = logging.handlers.QueueListener(
                self.log_queue,
                *actual_handlers,
                respect_handler_level=True
            )
            self.queue_listener.start()
            
            # Avoid mutating LogRecord fields before other handlers process the record.
            atexit.register(self._shutdown_queue_listener)
            
            self._async_enabled = True
        else:
            # Avoid mutating LogRecord fields before other handlers process the record.
            for handler in actual_handlers:
                self.logger.addHandler(handler)
            self._async_enabled = False
        
        self._is_configured = True
        mode_str = "async" if self._async_enabled else "sync"
        self.logger.debug(f"Logging system configured: level={logging.getLevelName(level)}, mode={mode_str}")
        
        return self.logger
    
    def _shutdown_queue_listener(self):
        """Run shutdown queue listener."""
        if self.queue_listener is not None:
            try:
                self.queue_listener.stop()
            except Exception:
                pass
            self.queue_listener = None
        self.log_queue = None
    
    def _get_formatter(self, style: str) -> logging.Formatter:
        """Get formatter."""
        if style == "simple":
            return logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
        elif style == "detailed":
            return logging.Formatter(
                '%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        elif style == "json":
            return JsonFormatter()
        else:
            return logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def _get_colored_formatter(self, style: str, use_color: bool = True) -> logging.Formatter:
        """Get colored formatter."""
        if not use_color:
            # Avoid mutating LogRecord fields before other handlers process the record.
            return self._get_formatter(style)
        
        if HAS_COLORLOG:
            # Avoid mutating LogRecord fields before other handlers process the record.
            log_colors = {
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
            
            # Avoid mutating LogRecord fields before other handlers process the record.
            if style == "simple":
                fmt = '%(asctime)s - %(name)s - %(log_color)s%(levelname)s%(reset)s - %(message)s'
                datefmt = '%H:%M:%S'
            elif style == "detailed":
                fmt = '%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(log_color)s%(levelname)s%(reset)s - %(message)s'
                datefmt = '%Y-%m-%d %H:%M:%S'
            else:
                fmt = '%(asctime)s - %(name)s - %(log_color)s%(levelname)s%(reset)s - %(message)s'
                datefmt = None
                
            return colorlog.ColoredFormatter(
                fmt,
                datefmt=datefmt,
                log_colors=log_colors,
                reset=True,
                style='%'
            )
        else:
            if style == "simple":
                return SafeColoredFormatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S'
                )
            elif style == "detailed":
                return SafeColoredFormatter(
                    '%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            else:
                return SafeColoredFormatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
    
    def _get_default_log_dir(self) -> str:
        """Get default log dir."""
        try:
            log_dir = Path.cwd() / "logs"
            return str(log_dir)
        except Exception:
            return "logs"
    
    def add_handler(self, name: str, handler: logging.Handler):
        """Add handler."""
        self.logger.addHandler(handler)
        self.handlers[name] = handler
    
    def remove_handler(self, name: str):
        """Remove handler."""
        if name in self.handlers:
            self.logger.removeHandler(self.handlers[name])
            del self.handlers[name]
    
    def set_level(self, level: int):
        """Set level."""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
    
    def get_stats(self) -> Dict[str, Any]:
        """Return stats."""
        stats = {
            'logger_name': self.name,
            'level': logging.getLevelName(self.logger.level),
            'handlers_count': len(self.logger.handlers),
            'handlers': list(self.handlers.keys()),
            'is_configured': self._is_configured,
            'using_colorlog': HAS_COLORLOG,
            'async_enabled': getattr(self, '_async_enabled', False),
        }
        if self.log_queue is not None:
            stats['queue_size'] = self.log_queue.qsize()
        return stats


# Avoid mutating LogRecord fields before other handlers process the record.
_global_logger_manager = None


def setup_logging(level: int = logging.INFO, 
                 console_output: bool = True,
                 file_output: bool = True,
                 log_dir: Optional[str] = None,
                 use_color: Optional[bool] = None,
                 **kwargs) -> logging.Logger:
    """Run setup logging."""
    global _global_logger_manager
    
    if _global_logger_manager is None:
        _global_logger_manager = MemorySystemLogger("AgentMemorySystem")
    
    return _global_logger_manager.setup(
        level=level,
        console_output=console_output, 
        file_output=file_output,
        log_dir=log_dir,
        use_color=use_color,
        **kwargs
    )


def get_logger(name: str) -> logging.Logger:
    """Return logger."""
    # Avoid mutating LogRecord fields before other handlers process the record.
    if _global_logger_manager is None:
        setup_logging()
    
    return logging.getLogger(f"AgentMemorySystem.{name}")


def set_log_level(level: int):
    """Set log level."""
    global _global_logger_manager
    
    if _global_logger_manager:
        _global_logger_manager.set_level(level)
    else:
        logging.getLogger().setLevel(level)


def create_module_logger(module_name: str, 
                        level: Optional[int] = None) -> logging.Logger:
    """Build module logger."""
    logger = get_logger(module_name)
    if level is not None:
        logger.setLevel(level)
    return logger


# Avoid mutating LogRecord fields before other handlers process the record.
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}


def get_log_level_from_string(level_str: str) -> int:
    """Return log level from string."""
    return LOG_LEVELS.get(level_str.upper(), logging.INFO)


def configure_development_logging():
    """Configure development logging."""
    return setup_logging(
        level=logging.DEBUG,
        console_output=True,
        file_output=True,
        format_style="detailed"
    )


def configure_production_logging():
    """Configure production logging."""
    return setup_logging(
        level=logging.INFO,
        console_output=False,
        file_output=True,
        format_style="json",
        async_logging=True
    )


def configure_testing_logging():
    """Configure testing logging."""
    return setup_logging(
        level=logging.WARNING,
        console_output=True,
        file_output=False,
        format_style="simple"
    )
    
def configure_benchmark_logging():
    """Configure benchmark logging."""
    return setup_logging(
        level=logging.INFO,
        console_output=True,
        file_output=True,
        format_style="detailed",
        async_logging=True
    )
    
def configure_speed_logging():
    """Configure speed logging."""
    return setup_logging(
        level=logging.WARNING,
        console_output=True,
        file_output=False,
        format_style="simple",
        async_logging=False
    )


def auto_configure_logging():
    env = os.getenv('ENVIRONMENT')
    
    if env is None:
        return None
    
    env = env.lower()
    if env == 'production':
        return configure_production_logging()
    elif env == 'benchmark':
        return configure_benchmark_logging()
    elif env == 'testing':
        return configure_testing_logging()
    elif env == 'development':
        return configure_development_logging()
    elif env == 'speed': 
        return configure_speed_logging()
    else:
        return None


# Avoid mutating LogRecord fields before other handlers process the record.
if not logging.getLogger().handlers:
    pass
