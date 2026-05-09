"""In-memory implementation of the UnitStore port.

Stores MemoryUnits and MemorySpaces in plain Python dictionaries.
Provides JSON serialization/deserialization for full state dumps.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..domain.memory_space import MemorySpace
from ..domain.memory_unit import MemoryUnit
from ..domain.types import SpaceName, Uid
from ..ports.unit_store import UnitStore


class InMemoryUnitStore(UnitStore):
    """In-memory unit and space store backed by Python dicts.

    Suitable for testing, small deployments, and as the hot cache layer
    in a multi-tier persistence setup. Supports JSON dump/load for
    checkpoint and restore workflows.

    Attributes:
        _units: Mapping from Uid to MemoryUnit.
        _spaces: Mapping from SpaceName to MemorySpace.
    """

    def __init__(self):
        self._units: Dict[Uid, MemoryUnit] = {}
        self._spaces: Dict[SpaceName, MemorySpace] = {}

    def dump_json(self) -> str:
        """Serialize the full store state to a JSON string.

        Returns:
            JSON-encoded string with 'units' and 'spaces' keys.
        """
        payload: Dict[str, Any] = {
            "units": [u.to_dict() for u in self.list_units()],
            "spaces": [s.to_dict() for s in self.list_spaces()],
        }
        return json.dumps(payload, ensure_ascii=False)

    def load_json(self, text: str) -> None:
        """Replace the current store state from a JSON string.

        Args:
            text: JSON-encoded string (produced by dump_json).
        """
        data = json.loads(text)
        units_raw = data.get("units") or []
        spaces_raw = data.get("spaces") or []

        units: List[MemoryUnit] = [MemoryUnit.from_dict(d) for d in units_raw]
        spaces: List[MemorySpace] = [MemorySpace.from_dict(d) for d in spaces_raw]

        self._units = {Uid(str(u.uid)): u for u in units}
        self._spaces = {SpaceName(str(s.name)): s for s in spaces}

    def upsert_units(self, units: Sequence[MemoryUnit]) -> None:
        for unit in units:
            self._units[Uid(str(unit.uid))] = unit

    def delete_units(self, uids: Iterable[Uid]) -> None:
        for uid in uids:
            self._units.pop(Uid(str(uid)), None)

    def get_unit(self, uid: Uid) -> Optional[MemoryUnit]:
        return self._units.get(Uid(str(uid)))

    def get_units(self, uids: Sequence[Uid]) -> List[MemoryUnit]:
        out: List[MemoryUnit] = []
        for uid in uids:
            unit = self.get_unit(uid)
            if unit is not None:
                out.append(unit)
        return out

    def list_units(self) -> List[MemoryUnit]:
        return list(self._units.values())

    def upsert_spaces(self, spaces: Sequence[MemorySpace]) -> None:
        for space in spaces:
            self._spaces[SpaceName(str(space.name))] = space

    def get_space(self, name: SpaceName) -> Optional[MemorySpace]:
        return self._spaces.get(SpaceName(str(name)))

    def list_spaces(self) -> List[MemorySpace]:
        return list(self._spaces.values())

    def clear(self) -> None:
        self._units.clear()
        self._spaces.clear()

    def flush(self) -> None:
        return
