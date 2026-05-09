# MemorySpace API Reference

## Overview

The `MemorySpace` is a logical container that organizes `MemoryUnit`s in a hierarchical tree structure. Each space can contain multiple memory units (`unit_uids`) and multiple child subspaces (`child_spaces`). The full space tree is managed by `SemanticMapService`.

## Core Attributes

- **name** (`SpaceName`): Space name identifier (NewType from `mandol.domain.types`)
- **unit_uids** (`Set[Uid]`): Set of memory unit UIDs contained in this space
- **child_spaces** (`Set[SpaceName]`): Set of child space names
- **summary_text** (`Optional[str]`): Optional summary text for the space
- **summary_embedding** (`Optional[Embedding]`): Optional summary vector representation (numpy ndarray)
- **metadata** (`Dict`): Metadata dictionary

## Key Methods

- `add_unit(uid: Uid)`: Add a memory unit to the space
- `remove_unit(uid: Uid)`: Remove a memory unit from the space
- `add_child_space(name: SpaceName)`: Add a child space
- `remove_child_space(name: SpaceName)`: Remove a child space
- `get_all_unit_uids(recursive: bool = True, resolver)`: Recursively get all unit UIDs (requires resolver for recursive access)
- `get_all_child_space_names(recursive: bool = True, resolver)`: Recursively get all child space names
- `set_summary(text: str, embedding: Embedding)`: Set space summary and its vector
- `touch()`: Update timestamp
- `to_dict()` / `from_dict(data)` (classmethod): Serialize / deserialize

## Hierarchy Example

```
root
├── root_base_memory
│   └── [unit_001, unit_002, ...]
└── root_high_level_memory
    ├── root_episodic
    │   ├── root_episodic_summary
    │   └── root_episodic_event
    ├── root_knowledge
    │   ├── root_knowledge_summary
    │   └── root_knowledge_entity
    ├── root_emotional
    ├── root_procedural
    └── root_insights
```

## Usage

```python
from mandol.domain.memory_space import MemorySpace
from mandol.domain.types import SpaceName, Uid

# Create root space
root = MemorySpace(name=SpaceName("root"))

# Add child spaces
root.add_child_space(SpaceName("base_memory"))
root.add_child_space(SpaceName("knowledge_entity"))

# Add units
root.add_unit(Uid("unit_001"))
```
