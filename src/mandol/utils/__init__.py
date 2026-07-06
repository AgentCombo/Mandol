"""Common utilities with heavy model management exposed lazily."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
import logging
import os
from typing import Optional

from .logging_config import (
    MemorySystemLogger,
    auto_configure_logging,
    configure_development_logging,
    configure_production_logging,
    configure_testing_logging,
    create_module_logger,
    get_logger,
    set_log_level,
    setup_logging,
)

CONNECTION_TEST_AVAILABLE = (
    find_spec("neo4j") is not None and find_spec("pymilvus") is not None
)
CLEAR_UTILS_AVAILABLE = CONNECTION_TEST_AVAILABLE

_LAZY_EXPORTS = {
    "GlobalModelManager": (".model_manager", "GlobalModelManager", True),
    "global_model_manager": (".model_manager", "global_model_manager", True),
    "test_milvus_connection": (
        ".connection_checks",
        "test_milvus_connection",
        CONNECTION_TEST_AVAILABLE,
    ),
    "test_neo4j_connection": (
        ".connection_checks",
        "test_neo4j_connection",
        CONNECTION_TEST_AVAILABLE,
    ),
    "clear_milvus": (".clear", "clear_milvus", CLEAR_UTILS_AVAILABLE),
    "clear_neo4j": (".clear", "clear_neo4j", CLEAR_UTILS_AVAILABLE),
    "clear_local_files": (".clear", "clear_local_files", CLEAR_UTILS_AVAILABLE),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attr_name, available = _LAZY_EXPORTS[name]
        if not available:
            globals()[name] = None
            return None
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


def configure_logging(
    level: str = "INFO",
    console_output: bool = True,
    file_output: bool = False,
    log_dir: Optional[str] = None,
    format_style: str = "detailed",
):
    """Configure Mandol logging with the existing logging backend."""
    level_int = getattr(logging, level.upper(), logging.INFO)
    return setup_logging(
        level=level_int,
        console_output=console_output,
        file_output=file_output,
        log_dir=log_dir,
        format_style=format_style,
    )


def get_utils_status() -> dict:
    """Return availability information without importing optional clients."""
    return {
        "logging_configured": True,
        "connection_test_available": CONNECTION_TEST_AVAILABLE,
        "clear_utils_available": CLEAR_UTILS_AVAILABLE,
        "available_functions": [
            name for name in __all__ if not name.endswith("_AVAILABLE")
        ],
    }


def print_utils_status() -> None:
    """Print a compact utility-component status report."""
    status = get_utils_status()
    print("=" * 50)
    print("Utils Components Status")
    print("=" * 50)
    print("Logging: ")
    print(
        f"Connection Test: {'' if status['connection_test_available'] else ''}"
    )
    print(f"Clear Utils: {'' if status['clear_utils_available'] else ''}")
    print("=" * 50)


if os.getenv("SKIP_AUTO_LOGGING", "").lower() != "true":
    try:
        setup_logging(
            level=logging.INFO,
            console_output=True,
            file_output=False,
            format_style="detailed",
        )
    except Exception as exc:  # Logging setup must never block package import.
        logging.warning("Mandol automatic logging configuration failed: %s", exc)


__all__ = [
    "setup_logging",
    "get_logger",
    "set_log_level",
    "create_module_logger",
    "configure_development_logging",
    "configure_production_logging",
    "configure_testing_logging",
    "auto_configure_logging",
    "MemorySystemLogger",
    "GlobalModelManager",
    "global_model_manager",
    "test_milvus_connection",
    "test_neo4j_connection",
    "CONNECTION_TEST_AVAILABLE",
    "clear_milvus",
    "clear_neo4j",
    "clear_local_files",
    "CLEAR_UTILS_AVAILABLE",
    "configure_logging",
    "get_utils_status",
    "print_utils_status",
]
