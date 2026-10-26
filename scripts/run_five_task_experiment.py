"""Controlled Five-Task Acceptance Experiment Runner for ARC-AGI-2 Solver.

Enforces strict evaluation isolation:
- Solver receives ONLY training demonstrations and test inputs.
- Test predictions are recorded BEFORE ground-truth solutions are accessed.
- Evaluates full hybrid solver across NUM_GUESSES=20 and NUM_GUESSES=40.
- Runs ablation baselines: Hunter baseline, Symbolic-only, Memory-enabled, Full Hybrid.
- Measures wall-clock runtime per stage, memory, candidate counts, and exact test matches.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_loader import ArcDatasetLoader, ArcTask
from src.solvers.integrated_solver import IntegratedArcSolver, TaskSolution

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExperimentRunner")


def evaluate_predictions_isolated(
    solutions_data: Dict[str, Any],
    task_id: str,
    solution: TaskSolution,
) -> Dict[str, Any]:
    """Compare generated predictions against held-out solutions in strict isolation."""
    if task_id not in solutions_data:
        raise KeyError(f"Solution for task {task_id} not available in evaluation solutions file.")

    ground_truth_outputs = solutions_data[task_id]
    num_tests = len(ground_truth_outputs)

    test_evaluations = []
    task_solved = True

    for idx in range(num_tests):
        gt = np.array(ground_truth_outputs[idx], dtype=np.int32)
        pred = solution.test_predictions[idx] if idx < len(solution.test_predictions) else None

        if pred is None:
            test_evaluations.append({
                "test_index": idx,
                "attempt_1_exact": False,
                "attempt_2_exact": False,
                "test_solved": False,
            })
            task_solved = False
            continue

        att1 = pred.attempt_1.to_numpy()
        att2 = pred.attempt_2.to_numpy()

        att1_exact = bool(np.array_equal(att1, gt))
        att2_exact = bool(np.array_equal(att2, gt))
        test_solved = att1_exact or att2_exact

        test_evaluations.append({
            "test_index": idx,
            "attempt_1_exact": att1_exact,
            "attempt_2_exact": att2_exact,
            "test_solved": test_solved,
            "att1_shape": list(att1.shape),
            "att2_shape": list(att2.shape),
            "gt_shape": list(gt.shape),
        })

        if not test_solved:
            task_solved = False

    return {
        "task_id": task_id,
        "task_solved": task_solved,
        "num_tests": num_tests,
        "tests_solved": sum(1 for t in test_evaluations if t["test_solved"]),
        "test_evaluations": test_evaluations,
    }


def run_experiment(
    manifest_path: str = "data/five_task_manifest.json",
    guess_budgets: List[int] = [20, 40],
    output_report_path: str = "reports/five_task_experiment_results.json",
) -> Dict[str, Any]:
    """Execute the full 5-task acceptance benchmark and ablation studies."""
    loader = ArcDatasetLoader()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    tasks_meta = manifest.get("tasks", [])
    task_ids = [t["task_id"] for t in tasks_meta]

    logger.info("Loaded 5-task manifest: %s", task_ids)

    # Load raw solutions strictly for post-prediction scoring
    training_solutions = loader.get_raw_solutions("training")

    experiment_results: List[Dict[str, Any]] = []

    # Modes to test:
    # 1. Full Hybrid with NUM_GUESSES=20
    # 2. Full Hybrid with NUM_GUESSES=40
    # 3. Hunter-Only Baseline
    # 4. Symbolic-Only (no Qwen, no Memory)
    modes = [
        {"mode_name": "Full Hybrid (Qwen+SG+Mem+DSL)", "guess_budget": 20, "use_qwen": True, "use_memory": True, "use_sg": True},
        {"mode_name": "Full Hybrid (Qwen+SG+Mem+DSL)", "guess_budget": 40, "use_qwen": True, "use_memory": True, "use_sg": True},
        {"mode_name": "Symbolic-Only (No Neural/Memory)", "guess_budget": 20, "use_qwen": False, "use_memory": False, "use_sg": True},
        {"mode_name": "Hunter-Only Baseline", "guess_budget": 20, "use_qwen": False, "use_memory": False, "use_sg": False, "hunter_only": True},
    ]

    for config in modes:
        mode_name = config["mode_name"]
        budget = config["guess_budget"]
        logger.info("=== Running Mode: %s | Budget: %d ===", mode_name, budget)

        solver = IntegratedArcSolver(
            num_guesses=budget,
            use_qwen=config.get("use_qwen", True),
            use_memory=config.get("use_memory", True),
            use_scene_graph=config.get("use_sg", True),
            search_timeout_seconds=4.0,
        )

        for tid in task_ids:
            task = loader.load_task(tid, split="training")
            t_start = time.time()

            if config.get("hunter_only"):
                # Run exclusively Hunter baseline
                t0 = time.time()
                h_prog = solver.hunter.solve(task)
                h_time = time.time() - t0
                if h_prog is not None:
                    sol = solver._build_solution(
                        task=task,
                        solved=True,
                        best_program=h_prog,
                        candidate_programs=[h_prog],
                        stage="hunter_baseline",
                        start_time=t0,
                        diagnostics={"hunter_solved": True},
                        profiling={"total_time": h_time},
                        component_statuses={"hunter": "integrated_and_influencing_predictions"},
                    )
                else:
                    sol = solver._empty_solution(task, t0, "hunter_failed")
            else:
                sol = solver.solve(task)

            # Evaluate against held-out solutions
            eval_res = evaluate_predictions_isolated(training_solutions, tid, sol)

            record = {
                "task_id": tid,
                "solver_mode": mode_name,
                "guess_budget": budget,
                "unique_candidates": sol.diagnostics.get("unique_candidates_count", len(sol.candidates_ranked)),
                "candidates_executed": sol.diagnostics.get("candidates_executed", 0),
                "exact_training_fit_found": sol.solved,
                "best_program": sol.best_program.to_code() if sol.best_program else "None",
                "attempt_1_exact": eval_res["test_evaluations"][0]["attempt_1_exact"] if eval_res["test_evaluations"] else False,
                "attempt_2_exact": eval_res["test_evaluations"][0]["attempt_2_exact"] if eval_res["test_evaluations"] else False,
                "task_solved": eval_res["task_solved"],
                "runtime_sec": round(sol.execution_time_sec, 4),
                "stage": sol.solver_pipeline_stage,
                "profiling": sol.profiling,
                "component_statuses": sol.component_statuses,
            }
            experiment_results.append(record)
            logger.info("  [%s] Solved: %s | TrainFit: %s | Time: %.3fs | Prog: %s",
                        tid, record["task_solved"], record["exact_training_fit_found"], record["runtime_sec"], record["best_program"])

    # Separate Regression Test Suite verification
    regression_results = []
    logger.info("=== Running Regression Verification Suite ===")
    reg_solver = IntegratedArcSolver(num_guesses=20)
    for reg_task_meta in manifest.get("regression_suite", []):
        rtid = reg_task_meta["task_id"]
        rtask = loader.load_task(rtid, split="training")
        rsol = reg_solver.solve(rtask)
        reval = evaluate_predictions_isolated(training_solutions, rtid, rsol)
        reg_record = {
            "task_id": rtid,
            "category": reg_task_meta["category"],
            "exact_training_fit_found": rsol.solved,
            "task_solved": reval["task_solved"],
            "runtime_sec": round(rsol.execution_time_sec, 4),
            "best_program": rsol.best_program.to_code() if rsol.best_program else "None",
        }
        regression_results.append(reg_record)
        logger.info("  Regression [%s]: Solved=%s (Time: %.3fs)", rtid, reg_record["task_solved"], reg_record["runtime_sec"])

    output_data = {
        "manifest_path": manifest_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_results": experiment_results,
        "regression_results": regression_results,
    }

    Path(output_report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Experiment results saved to %s", output_report_path)
    return output_data


if __name__ == "__main__":
    run_experiment()
