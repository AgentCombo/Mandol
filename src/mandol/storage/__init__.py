"""Package exports for storage."""

import logging
from ..utils.logging_config import create_module_logger

try:
    from .duckdb_operator import DuckDBOperator
    DUCKDB_AVAILABLE = True
except ImportError:
    DuckDBOperator = None
    DUCKDB_AVAILABLE = False

from .tiered_storage_manager import TieredEvictionResult, TieredStorageManager

__all__ = [
    'DuckDBOperator',
    'TieredStorageManager',
    'TieredEvictionResult',
    'create_duckdb_operator',
    'get_storage_status',
    'DUCKDB_AVAILABLE',
]

logger = create_module_logger("storage")


def create_duckdb_operator(
    db_path: str = ":memory:",
    **kwargs
):
    """Build duckdb operator."""
    if not DUCKDB_AVAILABLE:
        raise ImportError("DuckDB dependencies are not installed. Run: pip install duckdb pyarrow")
    return DuckDBOperator(db_path=db_path, **kwargs)


def get_storage_status() -> dict:
    """Return storage status."""
    status = {
        "duckdb_available": DUCKDB_AVAILABLE,
    }
    
    if DUCKDB_AVAILABLE:
        logger.info("Unified L2 storage is available (DuckDB)")
    else:
        logger.warning("DuckDB is unavailable; L2 storage support is limited")
    
    return status

# def get_default_configs():
#     return {
#         'milvus': DEFAULT_MILVUS_CONFIG.copy(),
#         'neo4j': DEFAULT_NEO4J_CONFIG.copy()
#     }

# def create_default_milvus_operator():
#     return create_milvus_operator(**DEFAULT_MILVUS_CONFIG)

# def create_default_neo4j_operator():
#     return create_neo4j_operator(**DEFAULT_NEO4J_CONFIG)

# import warnings

# if not MILVUS_AVAILABLE:
#     warnings.warn(

#         ImportWarning
#     )

# if not NEO4J_AVAILABLE:
#     warnings.warn(
#         ImportWarning
#     )
