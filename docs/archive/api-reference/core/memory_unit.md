# MemoryUnit API Reference

## Overview

The `MemoryUnit` is the smallest storage unit in the Mandol memory system. Each unit contains raw data, metadata, and optional vector representations. It is created via `MemoryUnit(uid=Uid(...), raw_data={...}, metadata={...})` and passed to `system.add(unit)`.

> **Insertion Mode**: The system automatically generates embeddings for the following fields in `raw_data`:
> - `text_content`: Text content → dense vector (Dense Embedding)
> - `image_path`: Image file path → image vector
>
> Other arbitrary fields (e.g., `speaker`, `source`) are stored as metadata but NOT automatically embedded.

## Core Attributes

- **uid** (`Uid`): Unique identifier (NewType from `mandol.domain.types`)
- **raw_data** (`Dict`): Raw data containing text, image paths, or other content
- **metadata** (`Dict`): User and system metadata (timestamps, session info, etc.)
- **embedding** (`Optional[Embedding]`): Dense vector representation (numpy ndarray)
- **sparse_embedding** (`Optional[Embedding]`): Sparse vector representation (numpy ndarray)

## Key Methods

- `to_dict()`: Serialize to dictionary for persistence
- `from_dict(data)` (classmethod): Deserialize from dictionary to MemoryUnit
- `get_user_metadata()`: Get user-defined metadata (excludes `_system_` prefix fields)
- `touch()`: Update `_system_updated_at` timestamp
- `__hash__()`: Hash based on uid
- `__eq__(other)`: Equality based on uid

## Usage

```python
from mandol import MemoryUnit, Uid
from mandol import MemorySystem

system = MemorySystem()

# Create a basic dialogue unit
unit = MemoryUnit(
    uid=Uid("dialogue_001"),
    raw_data={
        "text_content": "Zhang San went to Beijing on a business trip",
        "speaker": "user",
    },
    metadata={"timestamp": "2024-01-15T10:00:00"},
)
system.add(unit)

# Serialize and deserialize
data = unit.to_dict()
restored_unit = MemoryUnit.from_dict(data)
```
