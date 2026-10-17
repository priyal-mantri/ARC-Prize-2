"""Bootstrap Memory Harvester.

Runs baseline and synthesis across training tasks to populate the MirrorMatrix
memory bank and ConceptLibrary with verified, reproducible programs and fingerprints.
Strictly isolates training data from held-out evaluation sets.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

from src.data.dataset_loader import ArcTask
from src.fingerprint.encoder import task_to_fingerprint
from src.memory.mirror_matrix import MemoryEntry, MirrorMatrix
from src.solvers.concept_library import ConceptLibrary
from src.solvers.hunter import HunterBaselineSolver
from src.search.beam_search import DSLBeamSearch

logger = logging.getLogger(__name__)


class BootstrapHarvester:
    """Harvests verified programs from training tasks to prime memory and skill libraries."""

    def __init__(
        self,
        memory_bank: Optional[MirrorMatrix] = None,
        concept_library: Optional[ConceptLibrary] = None,
    ):
        self.memory = memory_bank or MirrorMatrix()
        self.library = concept_library or ConceptLibrary()
        self.hunter = HunterBaselineSolver()
        self.beam_search = DSLBeamSearch(timeout_seconds=2.0)

    def harvest(
        self,
        tasks: List[ArcTask],
        max_tasks: Optional[int] = None,
        save_dir: Optional[Path] = None,
    ) -> int:
        """Run solvers across training tasks and store verified programs in memory.

        Args:
            tasks: List of ArcTask objects from the TRAINING split.
            max_tasks: Optional limit on number of tasks to process.
            save_dir: Optional directory to persist harvested memory and library.

        Returns:
            Number of tasks solved and harvested.
        """
        solved_count = 0
        target_tasks = tasks[:max_tasks] if max_tasks else tasks

        logger.info("Starting memory bootstrap harvesting across %d tasks...", len(target_tasks))

        for idx, task in enumerate(target_tasks):
            # Guard: ensure we only harvest training tasks
            if task.split != "training":
                logger.warning("Skipping task %s: bootstrap must only ingest training tasks!", task.task_id)
                continue

            # 1. Try Hunter baseline
            prog = self.hunter.solve(task)

            # 2. If Hunter didn't solve it, try quick DSL beam search
            if prog is None:
                search_res = self.beam_search.search(task, timeout=1.5)
                if search_res.solved and search_res.best_program:
                    prog = search_res.best_program

            if prog is not None:
                solved_count += 1
                # Compute 20D fingerprint
                fp = task_to_fingerprint(task)

                # Store in MirrorMatrix
                entry = MemoryEntry(
                    task_id=task.task_id,
                    fingerprint=fp,
                    program=prog,
                    status="verified_reusable",
                    tags=[prog.name, prog.origin],
                )
                self.memory.add_entry(entry)

                # Register in ConceptLibrary
                self.library.register_skill(
                    program=prog,
                    task_id=task.task_id,
                    preconditions={"requires_same_shape": all(
                        ex.input_grid.shape == ex.output_grid.shape for ex in task.train_examples
                    )},
                )

                logger.info(
                    "[%d/%d] Harvested solved task %s with program %s",
                    idx + 1, len(target_tasks), task.task_id, prog.name
                )

        logger.info("Bootstrap harvesting complete. Solved %d / %d tasks.", solved_count, len(target_tasks))

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            self.memory.save(save_dir / "mirror_matrix.json")
            self.library.save(save_dir / "concept_library.json")

        return solved_count
