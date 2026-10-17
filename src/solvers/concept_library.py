"""Concept Invention and Library Learning for ARC-AGI-2.

Maintains a registry of verified, reusable abstract skills learned across tasks.
Extracts invariant subprograms, identifies compound operations, and exports/imports
skills with structured metadata and schema versioning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.dsl.program import DSLProgram

logger = logging.getLogger(__name__)


@dataclass
class LearnedSkill:
    """A verified, reusable abstract transformation learned from solved tasks."""
    skill_id: str
    program: DSLProgram
    preconditions: Dict[str, Any] = field(default_factory=dict)
    expected_effects: List[str] = field(default_factory=list)
    solved_task_ids: List[str] = field(default_factory=list)
    verification_status: str = "verified_reusable"
    complexity: int = 1
    known_limitations: str = ""
    schema_version: str = "v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "program": self.program.to_dict(),
            "preconditions": self.preconditions,
            "expected_effects": self.expected_effects,
            "solved_task_ids": self.solved_task_ids,
            "verification_status": self.verification_status,
            "complexity": self.complexity,
            "known_limitations": self.known_limitations,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnedSkill":
        prog = DSLProgram.from_dict(data["program"])
        return cls(
            skill_id=data["skill_id"],
            program=prog,
            preconditions=data.get("preconditions", {}),
            expected_effects=data.get("expected_effects", []),
            solved_task_ids=data.get("solved_task_ids", []),
            verification_status=data.get("verification_status", "verified_reusable"),
            complexity=data.get("complexity", prog.complexity),
            known_limitations=data.get("known_limitations", ""),
            schema_version=data.get("schema_version", "v1"),
        )


class ConceptLibrary:
    """Registry and synthesizer for learned reusable skills."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        self._skills: Dict[str, LearnedSkill] = {}
        if storage_path and storage_path.exists():
            self.load(storage_path)

    @property
    def skills(self) -> List[LearnedSkill]:
        return list(self._skills.values())

    def register_skill(
        self,
        program: DSLProgram,
        task_id: str,
        preconditions: Optional[Dict[str, Any]] = None,
        expected_effects: Optional[List[str]] = None,
    ) -> LearnedSkill:
        """Register a verified program as a reusable skill.

        If a skill with equivalent program representation exists, appends the solved task_id.
        """
        code = program.to_code()
        # Check if already present
        for skill in self._skills.values():
            if skill.program.to_code() == code:
                if task_id not in skill.solved_task_ids:
                    skill.solved_task_ids.append(task_id)
                return skill

        skill_id = f"skill_{len(self._skills) + 1:03d}_{program.name}"
        new_skill = LearnedSkill(
            skill_id=skill_id,
            program=program,
            preconditions=preconditions or {},
            expected_effects=expected_effects or [],
            solved_task_ids=[task_id],
            verification_status="verified_reusable",
            complexity=program.complexity,
        )
        self._skills[skill_id] = new_skill
        return new_skill

    def get_candidate_skills_for_task(self, task_preconditions: Dict[str, Any]) -> List[LearnedSkill]:
        """Retrieve skills matching a task's basic structural preconditions."""
        matched: List[LearnedSkill] = []
        for skill in self._skills.values():
            # Check shape compatibility
            if skill.preconditions.get("requires_same_shape") and not task_preconditions.get("same_shape", True):
                continue
            matched.append(skill)
        # Sort by popularity (how many tasks it has solved) and inverse complexity
        matched.sort(key=lambda s: (len(s.solved_task_ids), -s.complexity), reverse=True)
        return matched

    def save(self, path: Optional[Path] = None) -> None:
        """Persist concept library to JSON disk."""
        target = path or self.storage_path
        if not target:
            raise ValueError("No storage path specified for ConceptLibrary")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "skills": [s.to_dict() for s in self._skills.values()],
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d skills to %s", len(self._skills), target)

    def load(self, path: Path) -> None:
        """Load skills from JSON disk."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("skills", []):
            skill = LearnedSkill.from_dict(item)
            self._skills[skill.skill_id] = skill
        logger.info("Loaded %d skills from %s", len(self._skills), path)
