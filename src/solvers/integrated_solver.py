"""Master Integrated Neuro-Symbolic ARC Solver.

Coordinates the complete end-to-end reasoning pipeline:
Perception -> Scene Graph -> Correspondence -> Fingerprinting -> Memory Retrieval ->
Hunter Baseline -> Relational Synthesis -> DSL Beam Search -> REPL Refinement ->
Multi-Test Prediction & Ranking.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.data.dataset_loader import ArcTask
from src.representation.grid import ArcGrid
from src.perception.detector import PerceptionEngine
from src.perception.hypotheses import PerceptionHypothesisGenerator
from src.scene_graph.builder import SceneGraphBuilder
from src.correspondence.matcher import ObjectCorrespondenceMatcher
from src.correspondence.relational_change import RelationalChangeAnalyzer, RelationalChangeReport
from src.fingerprint.encoder import task_to_fingerprint
from src.memory.mirror_matrix import MirrorMatrix
from src.solvers.hunter import HunterBaselineSolver
from src.solvers.llm_synthesizer import LLMSynthesizerInterface
from src.search.verifier import DemonstrationVerifier, TaskDemonstration, VerificationResult
from src.search.beam_search import DSLBeamSearch, SearchResult
from src.dsl.program import DSLProgram

logger = logging.getLogger(__name__)


@dataclass
class TestPrediction:
    """Pair of candidate attempts for a single test input."""
    test_index: int
    attempt_1: ArcGrid
    attempt_2: ArcGrid
    attempt_1_program: Optional[str] = None
    attempt_2_program: Optional[str] = None


@dataclass
class TaskSolution:
    """Complete solution record for an ARC task."""
    task_id: str
    solved: bool
    best_program: Optional[DSLProgram]
    test_predictions: List[TestPrediction] = field(default_factory=list)
    execution_time_sec: float = 0.0
    solver_pipeline_stage: str = "none"
    candidates_ranked: List[DSLProgram] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class IntegratedArcSolver:
    """End-to-end ARC-AGI-2 solver integrating symbolic, object-centric, and memory reasoning."""

    def __init__(
        self,
        memory_bank: Optional[MirrorMatrix] = None,
        search_timeout_seconds: float = 3.5,
        enable_repl_refinement: bool = True,
        enable_memory_retrieval: bool = True,
    ):
        self.memory = memory_bank or MirrorMatrix()
        self.search_timeout = search_timeout_seconds
        self.enable_repl = enable_repl_refinement
        self.enable_memory = enable_memory_retrieval

        # Subsystems
        self.perception_engine = PerceptionEngine()
        self.hypothesis_gen = PerceptionHypothesisGenerator(self.perception_engine)
        self.graph_builder = SceneGraphBuilder()
        self.matcher = ObjectCorrespondenceMatcher()
        self.relational_analyzer = RelationalChangeAnalyzer(self.perception_engine, self.matcher)
        self.verifier = DemonstrationVerifier(timeout_seconds=0.1)
        self.hunter = HunterBaselineSolver(self.verifier)
        self.synthesizer = LLMSynthesizerInterface()
        self.beam_search = DSLBeamSearch(
            verifier=self.verifier,
            beam_width=16,
            max_depth=2,
            timeout_seconds=search_timeout_seconds,
            enable_repl_refinement=enable_repl_refinement,
        )

    def solve(self, task: ArcTask) -> TaskSolution:
        """Execute full neuro-symbolic pipeline to solve an ARC task.

        Args:
            task: ArcTask containing train demonstrations and test inputs.

        Returns:
            TaskSolution with verified program and test predictions.
        """
        start_time = time.time()
        stage = "initialization"

        demonstrations = [
            TaskDemonstration(
                input_grid=ArcGrid(np_grid=ex.input_grid),
                output_grid=ArcGrid(np_grid=ex.output_grid),
            )
            for ex in task.train_examples
        ]

        if not demonstrations or not task.test_inputs:
            return self._empty_solution(task, start_time, "invalid_task")

        # -------------------------------------------------------------
        # STAGE 1: Fast Hunter Symbolic Baseline
        # -------------------------------------------------------------
        stage = "hunter_baseline"
        hunter_prog = self.hunter.solve(task)
        if hunter_prog is not None:
            return self._build_solution(
                task=task,
                solved=True,
                best_program=hunter_prog,
                candidate_programs=[hunter_prog],
                stage=stage,
                start_time=start_time,
                diagnostics={"hunter_solved": True},
            )

        # -------------------------------------------------------------
        # STAGE 2: Object-Centric Perception & Relational Change Analysis
        # -------------------------------------------------------------
        stage = "relational_change_analysis"
        change_report: Optional[RelationalChangeReport] = None
        try:
            change_report = self.relational_analyzer.analyze_task(demonstrations)
        except Exception as e:
            logger.debug("Relational change analysis error on %s: %s", task.task_id, e)

        # -------------------------------------------------------------
        # STAGE 3: Structural Fingerprint & Memory Retrieval
        # -------------------------------------------------------------
        memory_candidates: List[DSLProgram] = []
        if self.enable_memory and len(self.memory) > 0:
            stage = "memory_retrieval"
            try:
                fp = task_to_fingerprint(task)
                matches = self.memory.query_similar(fp, top_k=3)
                for entry, dist in matches:
                    if entry.program is not None:
                        # Verify retrieved program against current demonstrations
                        vres = self.verifier.verify(entry.program, demonstrations)
                        if vres.is_fully_solved:
                            return self._build_solution(
                                task=task,
                                solved=True,
                                best_program=entry.program,
                                candidate_programs=[entry.program],
                                stage="memory_exact_hit",
                                start_time=start_time,
                                diagnostics={"memory_source_task": entry.task_id, "distance": dist},
                            )
                        elif vres.mean_pixel_accuracy > 0.6:
                            memory_candidates.append(entry.program)
            except Exception as e:
                logger.debug("Memory retrieval error on %s: %s", task.task_id, e)

        # -------------------------------------------------------------
        # STAGE 4: Relational Symbolic Hypotheses
        # -------------------------------------------------------------
        stage = "relational_hypotheses"
        relational_candidates = self.synthesizer.generate_hypotheses(task, change_report, max_candidates=5)
        for prog in relational_candidates:
            vres = self.verifier.verify(prog, demonstrations)
            if vres.is_fully_solved:
                return self._build_solution(
                    task=task,
                    solved=True,
                    best_program=prog,
                    candidate_programs=[prog],
                    stage=stage,
                    start_time=start_time,
                    diagnostics={"relational_rule_solved": prog.name},
                )

        # -------------------------------------------------------------
        # STAGE 5: Constrained DSL Beam Search & REPL Refinement
        # -------------------------------------------------------------
        stage = "dsl_beam_search"
        remaining_timeout = max(0.5, self.search_timeout - (time.time() - start_time))
        search_res: SearchResult = self.beam_search.search(task, timeout=remaining_timeout)

        surviving_candidates: List[DSLProgram] = []
        if search_res.best_program:
            surviving_candidates.append(search_res.best_program)
        for p in search_res.candidate_programs:
            if p not in surviving_candidates:
                surviving_candidates.append(p)
        for p in memory_candidates:
            if p not in surviving_candidates:
                surviving_candidates.append(p)

        best_prog = search_res.best_program
        solved = search_res.solved and best_prog is not None

        return self._build_solution(
            task=task,
            solved=solved,
            best_program=best_prog,
            candidate_programs=surviving_candidates,
            stage=stage,
            start_time=start_time,
            diagnostics={
                "search_candidates_executed": search_res.candidates_executed,
                "search_candidates_generated": search_res.candidates_generated,
                "search_depth": search_res.depth_reached,
                **search_res.diagnostics,
            },
        )

    def _build_solution(
        self,
        task: ArcTask,
        solved: bool,
        best_program: Optional[DSLProgram],
        candidate_programs: List[DSLProgram],
        stage: str,
        start_time: float,
        diagnostics: Dict[str, Any],
    ) -> TaskSolution:
        """Execute top candidate programs on test inputs and package result."""
        test_predictions: List[TestPrediction] = []

        prog_1 = best_program
        prog_2 = candidate_programs[1] if len(candidate_programs) > 1 else None

        for idx, test_input_np in enumerate(task.test_inputs):
            test_grid = ArcGrid(np_grid=test_input_np)

            # Attempt 1
            att1_grid = test_grid  # Default fallback
            att1_name = "fallback_input"
            if prog_1 is not None:
                try:
                    res1 = prog_1.execute(test_grid)
                    if res1 is not None and res1.shape[0] > 0 and res1.shape[1] > 0:
                        att1_grid = res1
                        att1_name = prog_1.name
                except Exception as e:
                    logger.debug("Attempt 1 execution failed on %s: %s", task.task_id, e)

            # Attempt 2
            att2_grid = att1_grid  # Default to attempt 1
            att2_name = att1_name
            if prog_2 is not None and prog_2 != prog_1:
                try:
                    res2 = prog_2.execute(test_grid)
                    if res2 is not None and res2.shape[0] > 0 and res2.shape[1] > 0:
                        att2_grid = res2
                        att2_name = prog_2.name
                except Exception as e:
                    logger.debug("Attempt 2 execution failed on %s: %s", task.task_id, e)

            test_predictions.append(
                TestPrediction(
                    test_index=idx,
                    attempt_1=att1_grid,
                    attempt_2=att2_grid,
                    attempt_1_program=att1_name,
                    attempt_2_program=att2_name,
                )
            )

        duration = time.time() - start_time
        return TaskSolution(
            task_id=task.task_id,
            solved=solved,
            best_program=best_program,
            test_predictions=test_predictions,
            execution_time_sec=duration,
            solver_pipeline_stage=stage,
            candidates_ranked=candidate_programs,
            diagnostics=diagnostics,
        )

    def _empty_solution(self, task: ArcTask, start_time: float, reason: str) -> TaskSolution:
        """Generate safe fallback for invalid or unparseable tasks."""
        duration = time.time() - start_time
        preds = []
        for idx, t_np in enumerate(task.test_inputs):
            g = ArcGrid(np_grid=t_np)
            preds.append(TestPrediction(test_index=idx, attempt_1=g, attempt_2=g, attempt_1_program="fallback", attempt_2_program="fallback"))
        return TaskSolution(
            task_id=task.task_id,
            solved=False,
            best_program=None,
            test_predictions=preds,
            execution_time_sec=duration,
            solver_pipeline_stage=reason,
        )
