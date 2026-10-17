"""MirrorMatrix and Memory Bank for ARC-AGI-2.

Indexes structural fingerprints and verified DSL transformation programs.
Seamlessly integrates FAISS when installed, with a high-performance pure NumPy L2/cosine
vector search fallback when FAISS is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.fingerprint.encoder import FINGERPRINT_DIM_V1
from src.dsl.program import DSLProgram

logger = logging.getLogger(__name__)

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


@dataclass
class MemoryEntry:
    """Represents a verified skill, reusable program, or solved task pattern in memory."""
    task_id: str
    fingerprint: List[float]
    program: Optional[DSLProgram] = None
    code: str = ""
    status: str = "verified_reusable"
    provenance: str = "verified_solver"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.code and self.program:
            self.code = self.program.to_code()

    @property
    def barcode(self) -> List[float]:
        return self.fingerprint

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "fingerprint": self.fingerprint,
            "code": self.code,
            "program": self.program.to_dict() if self.program else None,
            "status": self.status,
            "provenance": self.provenance,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        prog_data = data.get("program")
        prog = DSLProgram.from_dict(prog_data) if prog_data else None
        fp = data.get("fingerprint", data.get("barcode", []))
        return cls(
            task_id=data["task_id"],
            fingerprint=fp,
            program=prog,
            code=data.get("code", ""),
            status=data.get("status", "verified_reusable"),
            provenance=data.get("provenance", "verified_solver"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


# Backwards compatibility alias
MemoryRecord = MemoryEntry


class MirrorMatrix:
    """Vector memory retrieval bank for structural fingerprints and transformation programs."""

    def __init__(self, vector_dim: int = FINGERPRINT_DIM_V1, storage_path: Optional[Path] = None):
        self.vector_dim = vector_dim
        self.storage_path = storage_path
        self.entries: List[MemoryEntry] = []
        self._vectors: Optional[np.ndarray] = None  # shape (N, vector_dim)

        if HAS_FAISS:
            self.faiss_index = faiss.IndexFlatL2(vector_dim)
        else:
            self.faiss_index = None

        if storage_path and storage_path.exists():
            self.load(storage_path)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def records(self) -> List[MemoryEntry]:
        return self.entries

    @property
    def total_memories(self) -> int:
        return len(self.entries)

    def add_entry(self, entry: MemoryEntry) -> None:
        """Add a MemoryEntry to the memory bank and update index."""
        vec = np.array(entry.fingerprint, dtype=np.float32).flatten()
        if len(vec) != self.vector_dim:
            # Pad or truncate if dimension differs
            if len(vec) < self.vector_dim:
                vec = np.pad(vec, (0, self.vector_dim - len(vec)))
            else:
                vec = vec[:self.vector_dim]

        entry.fingerprint = vec.tolist()
        self.entries.append(entry)

        if HAS_FAISS and self.faiss_index is not None:
            self.faiss_index.add(np.expand_dims(vec, axis=0))

        if self._vectors is None:
            self._vectors = np.expand_dims(vec, axis=0)
        else:
            self._vectors = np.vstack([self._vectors, np.expand_dims(vec, axis=0)])

    def save_memory(
        self,
        barcode: Union[np.ndarray, List[float]],
        task_id: str,
        code_string: str,
        provenance: str = "verified_solver",
        tags: Optional[List[str]] = None,
        **metadata: Any
    ) -> None:
        """Legacy helper to add a memory entry."""
        entry = MemoryEntry(
            task_id=task_id,
            fingerprint=list(barcode),
            code=code_string,
            provenance=provenance,
            tags=tags or [],
            metadata=metadata,
        )
        self.add_entry(entry)

    def query_similar(
        self,
        query_fp: Union[np.ndarray, List[float]],
        top_k: int = 3,
    ) -> List[Tuple[MemoryEntry, float]]:
        """Retrieve top_k nearest matching MemoryEntries and their L2 distance.

        Returns:
            List of (MemoryEntry, distance) tuples.
        """
        if len(self.entries) == 0:
            return []

        k = min(top_k, len(self.entries))
        q_vec = np.array(query_fp, dtype=np.float32).flatten()
        if len(q_vec) != self.vector_dim:
            if len(q_vec) < self.vector_dim:
                q_vec = np.pad(q_vec, (0, self.vector_dim - len(q_vec)))
            else:
                q_vec = q_vec[:self.vector_dim]

        if HAS_FAISS and self.faiss_index is not None:
            distances, indices = self.faiss_index.search(np.expand_dims(q_vec, axis=0), k)
            results: List[Tuple[MemoryEntry, float]] = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx < len(self.entries):
                    results.append((self.entries[idx], float(dist)))
            return results

        # Pure NumPy L2 distance fallback
        diffs = self._vectors - q_vec
        dists = np.sum(diffs ** 2, axis=1)
        top_k_indices = np.argsort(dists)[:k]

        return [(self.entries[idx], float(dists[idx])) for idx in top_k_indices]

    def retrieve_memory(
        self,
        query_barcode: Union[np.ndarray, List[float]],
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Legacy dict-based retrieval helper."""
        matches = self.query_similar(query_barcode, top_k=k)
        ret = []
        for entry, dist in matches:
            d = entry.to_dict()
            d["distance"] = dist
            ret.append(d)
        return ret

    def save(self, path: Optional[Path] = None) -> None:
        """Persist memory bank to JSON disk."""
        target = path or self.storage_path
        if not target:
            raise ValueError("No storage path provided for MirrorMatrix")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "vector_dim": self.vector_dim,
            "entries": [e.to_dict() for e in self.entries],
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d memory entries to %s", len(self.entries), target)

    def load(self, path: Path) -> None:
        """Load memory bank from JSON disk."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.entries = []
        self._vectors = None
        if HAS_FAISS:
            self.faiss_index = faiss.IndexFlatL2(self.vector_dim)

        for item in data.get("entries", []):
            entry = MemoryEntry.from_dict(item)
            self.add_entry(entry)
        logger.info("Loaded %d memory entries from %s", len(self.entries), path)
