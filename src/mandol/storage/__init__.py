"""RocksDB payload persistence and tiered-cache orchestration."""

from ..utils.logging_config import create_module_logger

try:
    from .rocksdb_payload_store import RocksDBPayloadStore
    ROCKSDB_AVAILABLE = True
except ImportError:
    RocksDBPayloadStore = None
    ROCKSDB_AVAILABLE = False

from .tiered_storage_manager import TieredEvictionResult, TieredStorageManager

__all__ = [
    "RocksDBPayloadStore",
    "TieredStorageManager",
    "TieredEvictionResult",
    "create_rocksdb_payload_store",
    "get_storage_status",
    "ROCKSDB_AVAILABLE",
]

logger = create_module_logger("storage")


def create_rocksdb_payload_store(
    db_path: str = "./l2_database/payloads.rocksdb",
):
    """Open the RocksDB store used by automatic payload paging."""
    if not ROCKSDB_AVAILABLE:
        raise ImportError("RocksDB support is unavailable. Install the rocksdict package.")
    return RocksDBPayloadStore(db_path=db_path)


def get_storage_status() -> dict:
    """Return storage status."""
    status = {
        "rocksdb_available": ROCKSDB_AVAILABLE,
    }

    if ROCKSDB_AVAILABLE:
        logger.info("RocksDB payload persistence is available.")
    else:
        logger.warning("RocksDB is unavailable; persistent payload paging is disabled.")

    return status
