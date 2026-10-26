"""Master Integrated Neuro-Symbolic ARC Solver.

Coordinates the complete end-to-end reasoning pipeline:
Perception -> Scene Graph -> Relational Reasoning -> Fingerprinting -> Memory Retrieval ->
Hunter Baseline -> Neural (Qwen2.5-Coder) Hypotheses -> DSL Beam Search -> REPL Refinement ->
Multi-Test Prediction & Two-Attempt Submission Generation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np

from src.data.dataset_loader import ArcTask
from src.representation.grid import ArcGrid
from src.perception.detector import PerceptionEngine
from src.perception.hypotheses import PerceptionHypothesisGenerator
from src.scene_graph.builder import SceneGraphBuilder
from src.scene_graph.graph import ArcSceneGraph
from src.correspondence.matcher import ObjectCorrespondenceMatcher
from src.correspondence.relational_change import RelationalChangeAnalyzer, RelationalChangeReport
from src.fingerprint.encoder import task_to_fingerprint
from src.memory.mirror_matrix import MirrorMatrix
from src.solvers.concept_library import ConceptLibrary
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
    profiling: Dict[str, float] = field(default_factory=dict)
    component_statuses: Dict[str, str] = field(default_factory=dict)


class IntegratedArcSolver:
    """End-to-end ARC-AGI-2 solver integrating symbolic, object-centric, and neural reasoning."""

    def __init__(
        self,
        memory_bank: Optional[MirrorMatrix] = None,
        concept_library: Optional[ConceptLibrary] = None,
        num_guesses: int = 20,
        search_timeout_seconds: float = 4.0,
        enable_repl_refinement: bool = True,
        enable_memory_retrieval: bool = True,
        use_qwen: bool = True,
        use_scene_graph: bool = True,
        use_relational_reasoning: bool = True,
        use_memory: Optional[bool] = None,
        use_repl_refinement: Optional[bool] = None,
        model_path: Optional[str] = None,
    ):
        self.num_guesses = num_guesses
        self.search_timeout = search_timeout_seconds
        self.enable_repl = use_repl_refinement if use_repl_refinement is not None else enable_repl_refinement
        self.enable_memory = use_memory if use_memory is not None else enable_memory_retrieval
        self.use_qwen = use_qwen
        self.use_scene_graph = use_scene_graph
        self.use_relational_reasoning = use_relational_reasoning

        if memory_bank is not None:
            self.memory = memory_bank
        else:
            default_mem_path = Path("data/memory/mirror_matrix.json")
            if default_mem_path.exists():
                self.memory = MirrorMatrix(storage_path=default_mem_path)
            else:
                self.memory = MirrorMatrix()

        if concept_library is not None:
            self.concept_library = concept_library
        else:
            default_lib_path = Path("data/memory/concept_library.json")
            if default_lib_path.exists():
                self.concept_library = ConceptLibrary(storage_path=default_lib_path)
            else:
                self.concept_library = ConceptLibrary()

        # Core Subsystems
        self.perception_engine = PerceptionEngine()
        self.hypothesis_gen = PerceptionHypothesisGenerator(self.perception_engine)
        self.graph_builder = SceneGraphBuilder()
        self.matcher = ObjectCorrespondenceMatcher()
        self.relational_analyzer = RelationalChangeAnalyzer(self.perception_engine, self.matcher)
        self.verifier = DemonstrationVerifier(timeout_seconds=0.1)
        self.hunter = HunterBaselineSolver(self.verifier)
        self.synthesizer = LLMSynthesizerInterface(model_path=model_path)
        self.beam_search = DSLBeamSearch(
            verifier=self.verifier,
            beam_width=16,
            max_depth=2,
            timeout_seconds=search_timeout_seconds,
            enable_repl_refinement=enable_repl_refinement,
            max_candidates=num_guesses,
        )

    def solve(self, task: ArcTask) -> TaskSolution:
        """Execute full neuro-symbolic pipeline to solve an ARC task."""
        start_time = time.time()
        profiling: Dict[str, float] = {}
        component_statuses: Dict[str, str] = {
            "perception": "implemented_inactive",
            "scene_graph": "implemented_inactive",
            "relational_reasoning": "implemented_inactive",
            "memory_retrieval": "implemented_inactive",
            "qwen_neural": "implemented_inactive",
            "dsl_beam_search": "implemented_inactive",
        }

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
        t0 = time.time()
        stage = "hunter_baseline"
        hunter_prog = self.hunter.solve(task)
        profiling["hunter_time"] = time.time() - t0
        if hunter_prog is not None:
            component_statuses["dsl_beam_search"] = "integrated_and_influencing_predictions"
            return self._build_solution(
                task=task,
                solved=True,
                best_program=hunter_prog,
                candidate_programs=[hunter_prog],
                stage=stage,
                start_time=start_time,
                diagnostics={"hunter_solved": True},
                profiling=profiling,
                component_statuses=component_statuses,
            )

        # -------------------------------------------------------------
        # STAGE 2: Object-Centric Perception & Scene Graph Construction
        # -------------------------------------------------------------
        scene_graphs: List[ArcSceneGraph] = []
        scene_graph_facts: Dict[str, Any] = {
            "has_holes": False,
            "total_objects": 0,
            "alignments": [],
            "containment_detected": False,
        }

        if self.use_scene_graph:
            t0 = time.time()
            component_statuses["perception"] = "integrated_and_influencing_predictions"
            component_statuses["scene_graph"] = "integrated_and_influencing_predictions"
            try:
                for demo in demonstrations:
                    objs = self.perception_engine.detect_objects(demo.input_grid, background_color=0)
                    sg = self.graph_builder.build(demo.input_grid, objs, background_color=0)
                    scene_graphs.append(sg)
                    scene_graph_facts["total_objects"] += len(objs)
                    for n in sg.nodes.values():
                        if n.attributes.get("holes_count", 0) > 0:
                            scene_graph_facts["has_holes"] = True
                    for edge in sg.edges:
                        if edge.relation in ("contains", "inside"):
                            scene_graph_facts["containment_detected"] = True
            except Exception as e:
                logger.debug("Scene graph building error on %s: %s", task.task_id, e)
                component_statuses["scene_graph"] = "failed"
            profiling["scene_graph_time"] = time.time() - t0

        # -------------------------------------------------------------
        # STAGE 3: Relational Change Analysis
        # -------------------------------------------------------------
        change_report: Optional[RelationalChangeReport] = None
        if self.use_relational_reasoning:
            t0 = time.time()
            component_statuses["relational_reasoning"] = "integrated_and_influencing_predictions"
            try:
                change_report = self.relational_analyzer.analyze_task(demonstrations)
            except Exception as e:
                logger.debug("Relational change analysis error on %s: %s", task.task_id, e)
                component_statuses["relational_reasoning"] = "failed"
            profiling["relational_time"] = time.time() - t0

        # -------------------------------------------------------------
        # STAGE 4: Concept Library & Memory Retrieval
        # -------------------------------------------------------------
        memory_candidates: List[DSLProgram] = []
        if self.enable_memory:
            t0 = time.time()
            component_statuses["memory_retrieval"] = "executed_no_influence"
            # 4a. Concept Library Skills
            if len(self.concept_library.skills) > 0:
                same_shape = all(d.input_grid.shape == d.output_grid.shape for d in demonstrations)
                matching_skills = self.concept_library.get_candidate_skills_for_task({"same_shape": same_shape})
                for skill in matching_skills:
                    vres = self.verifier.verify(skill.program, demonstrations)
                    if vres.is_fully_solved:
                        component_statuses["memory_retrieval"] = "integrated_and_influencing_predictions"
                        return self._build_solution(
                            task=task,
                            solved=True,
                            best_program=skill.program,
                            candidate_programs=[skill.program],
                            stage="concept_library_hit",
                            start_time=start_time,
                            diagnostics={"skill_id": skill.skill_id, "solved_tasks": skill.solved_task_ids},
                            profiling=profiling,
                            component_statuses=component_statuses,
                        )
                    elif vres.mean_pixel_accuracy > 0.6:
                        memory_candidates.append(skill.program)

            # 4b. Vector Memory Retrieval
            if len(self.memory) > 0:
                try:
                    fp = task_to_fingerprint(task)
                    matches = self.memory.query_similar(fp, top_k=3)
                    for entry, dist in matches:
                        if entry.program is not None:
                            vres = self.verifier.verify(entry.program, demonstrations)
                            if vres.is_fully_solved:
                                component_statuses["memory_retrieval"] = "integrated_and_influencing_predictions"
                                return self._build_solution(
                                    task=task,
                                    solved=True,
                                    best_program=entry.program,
                                    candidate_programs=[entry.program],
                                    stage="memory_exact_hit",
                                    start_time=start_time,
                                    diagnostics={"memory_source_task": entry.task_id, "distance": dist},
                                    profiling=profiling,
                                    component_statuses=component_statuses,
                                )
                            elif vres.mean_pixel_accuracy > 0.6 and entry.program not in memory_candidates:
                                memory_candidates.append(entry.program)
                except Exception as e:
                    logger.debug("Memory retrieval error on %s: %s", task.task_id, e)
            profiling["memory_time"] = time.time() - t0

        # -------------------------------------------------------------
        # STAGE 5: Neural (Qwen2.5-Coder) / Relational Hypotheses
        # -------------------------------------------------------------
        neural_candidates: List[DSLProgram] = []
        if self.use_qwen:
            t0 = time.time()
            qwen_budget = min(10, max(2, self.num_guesses // 2))
            neural_candidates = self.synthesizer.generate_hypotheses(
                task=task,
                change_report=change_report,
                max_candidates=qwen_budget,
                scene_graph_facts=scene_graph_facts,
            )
            profiling["qwen_inference_time"] = time.time() - t0
            if self.synthesizer.telemetry.inference_success:
                component_statuses["qwen_neural"] = "integrated_and_influencing_predictions"
            else:
                component_statuses["qwen_neural"] = "unavailable"

            for prog in neural_candidates:
                vres = self.verifier.verify(prog, demonstrations)
                if vres.is_fully_solved:
                    return self._build_solution(
                        task=task,
                        solved=True,
                        best_program=prog,
                        candidate_programs=[prog],
                        stage="neural_relational_hit",
                        start_time=start_time,
                        diagnostics={
                            "solved_by": prog.name,
                            "origin": prog.origin,
                            "scene_graph_facts": scene_graph_facts,
                            "qwen_telemetry": self.synthesizer.telemetry.to_dict(),
                        },
                        profiling=profiling,
                        component_statuses=component_statuses,
                    )

        # -------------------------------------------------------------
        # STAGE 6: Constrained DSL Beam Search & REPL Refinement
        # -------------------------------------------------------------
        stage = "dsl_beam_search"
        component_statuses["dsl_beam_search"] = "integrated_and_influencing_predictions"
        remaining_timeout = max(0.5, self.search_timeout - (time.time() - start_time))

        t0 = time.time()
        search_res: SearchResult = self.beam_search.search(
            task,
            timeout=remaining_timeout,
            max_candidates=self.num_guesses,
            scene_graph_facts=scene_graph_facts,
        )
        profiling["beam_search_time"] = time.time() - t0

        # Pool surviving candidates with deduplication
        seen_code: Set[str] = set()
        surviving_candidates: List[DSLProgram] = []

        def add_survivor(p: DSLProgram):
            sig = p.to_code()
            if sig not in seen_code:
                seen_code.add(sig)
                surviving_candidates.append(p)

        if search_res.best_program:
            add_survivor(search_res.best_program)
        for p in search_res.candidate_programs:
            add_survivor(p)
        for p in neural_candidates:
            add_survivor(p)
        for p in memory_candidates:
            add_survivor(p)

        best_prog = surviving_candidates[0] if surviving_candidates else None
        solved = search_res.solved and best_prog is not None

        profiling["total_time"] = time.time() - start_time

        return self._build_solution(
            task=task,
            solved=solved,
            best_program=best_prog,
            candidate_programs=surviving_candidates,
            stage=stage,
            start_time=start_time,
            diagnostics={
                "unique_candidates_count": len(surviving_candidates),
                "candidates_executed": search_res.candidates_executed,
                "candidates_generated": search_res.candidates_generated,
                "search_depth": search_res.depth_reached,
                "scene_graph_facts": scene_graph_facts,
                "qwen_telemetry": self.synthesizer.telemetry.to_dict(),
                **search_res.diagnostics,
            },
            profiling=profiling,
            component_statuses=component_statuses,
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
        profiling: Dict[str, float],
        component_statuses: Dict[str, str],
    ) -> TaskSolution:
        """Execute top candidate programs on test inputs and package result with 2 attempts."""
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
        profiling["total_duration"] = duration

        return TaskSolution(
            task_id=task.task_id,
            solved=solved,
            best_program=best_program,
            test_predictions=test_predictions,
            execution_time_sec=duration,
            solver_pipeline_stage=stage,
            candidates_ranked=candidate_programs,
            diagnostics=diagnostics,
            profiling=profiling,
            component_statuses=component_statuses,
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
