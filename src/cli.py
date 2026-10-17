"""Unified Command-Line Interface for ARC-AGI-2 Research Solver.

Subcommands:
  solve        Solve a single task by ID and output predictions.
  perceive     Inspect object-centric perception and hypotheses for a task.
  scene-graph  Inspect scene graph nodes, edges, and relations for a task.
  relational   Inspect cross-demonstration object correspondence and relational changes.
  baseline     Run fast Hunter symbolic baseline on a dataset split.
  benchmark    Run full solver benchmark on a dataset split with ground truth scoring.
  harvest      Bootstrap memory and concept library from training tasks.
  submit       Generate a validated Kaggle competition submission file.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Optional

from src.data.dataset_loader import ArcDatasetLoader, ArcTask
from src.representation.grid import ArcGrid
from src.perception.detector import PerceptionEngine
from src.perception.hypotheses import PerceptionHypothesisGenerator
from src.scene_graph.builder import SceneGraphBuilder
from src.correspondence.matcher import ObjectCorrespondenceMatcher
from src.correspondence.relational_change import RelationalChangeAnalyzer
from src.solvers.hunter import HunterBaselineSolver
from src.solvers.integrated_solver import IntegratedArcSolver
from src.solvers.bootstrap import BootstrapHarvester
from src.benchmark.harness import BenchmarkHarness
from src.submission.adapter import SubmissionAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("arc_cli")


def get_task_by_id(loader: ArcDatasetLoader, task_id: str, split: str = "training") -> Optional[ArcTask]:
    """Retrieve an ArcTask by task_id from the specified split."""
    if split == "training":
        tasks = loader.load_training_tasks()
    elif split == "evaluation":
        tasks = loader.load_evaluation_tasks()
    elif split == "test":
        tasks = loader.load_test_challenges()
    else:
        raise ValueError(f"Unknown split: {split}")

    for t in tasks:
        if t.task_id == task_id:
            return t
    return None


def cmd_solve(args: argparse.Namespace) -> None:
    loader = ArcDatasetLoader(Path(args.data_dir))
    task = get_task_by_id(loader, args.task_id, args.split)
    if not task:
        print(f"Error: Task {args.task_id} not found in {args.split} split.")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"SOLVING TASK: {task.task_id} ({args.split} split)")
    print(f"Demonstrations: {len(task.train_examples)} | Test inputs: {len(task.test_inputs)}")
    print(f"=======================================================")

    solver = IntegratedArcSolver(search_timeout_seconds=args.timeout)
    solution = solver.solve(task)

    print(f"\n[Result] Solved: {solution.solved}")
    print(f"[Result] Pipeline Stage: {solution.solver_pipeline_stage}")
    print(f"[Result] Execution Time: {solution.execution_time_sec:.3f}s")
    if solution.best_program:
        print(f"[Result] Best Program: {solution.best_program.name}")
        print(f"[Result] Code:\n{solution.best_program.to_code()}")

    print("\n--- Test Predictions ---")
    for pred in solution.test_predictions:
        print(f"\nTest Input #{pred.test_index}:")
        print(f"Attempt 1 ({pred.attempt_1_program}) shape {pred.attempt_1.shape}:")
        print(pred.attempt_1.to_numpy())
        print(f"Attempt 2 ({pred.attempt_2_program}) shape {pred.attempt_2.shape}:")
        print(pred.attempt_2.to_numpy())


def cmd_perceive(args: argparse.Namespace) -> None:
    loader = ArcDatasetLoader(Path(args.data_dir))
    task = get_task_by_id(loader, args.task_id, args.split)
    if not task:
        print(f"Error: Task {args.task_id} not found.")
        sys.exit(1)

    engine = PerceptionEngine()
    hypo_gen = PerceptionHypothesisGenerator(engine)

    print(f"\n=== Perception Inspection: Task {task.task_id} ===")
    for idx, ex in enumerate(task.train_examples):
        in_grid = ArcGrid(np_grid=ex.input_grid)
        print(f"\n--- Train Demonstration #{idx} Input {in_grid.shape} ---")
        hypotheses = hypo_gen.generate_hypotheses(in_grid)
        for h in hypotheses:
            print(f"  Hypothesis: {h.name} (confidence={h.confidence:.2f}, bg={h.background_color})")
            print(f"    Objects detected: {len(h.objects)}")
            for obj in h.objects[:4]:
                print(f"      - Obj #{obj.object_id}: color={obj.color}, area={obj.area}, shape={obj.shape_type}, bbox={obj.bbox}")


def cmd_scene_graph(args: argparse.Namespace) -> None:
    loader = ArcDatasetLoader(Path(args.data_dir))
    task = get_task_by_id(loader, args.task_id, args.split)
    if not task:
        print(f"Error: Task {args.task_id} not found.")
        sys.exit(1)

    engine = PerceptionEngine()
    builder = SceneGraphBuilder()

    print(f"\n=== Scene Graph Inspection: Task {task.task_id} ===")
    first_ex = task.train_examples[0]
    in_grid = ArcGrid(np_grid=first_ex.input_grid)
    bg_color = engine.infer_background_color(in_grid)
    objects = engine.segment_connected_components(in_grid, background_color=bg_color)
    graph = builder.build(in_grid, objects, background_color=bg_color)

    print(f"Graph nodes: {graph.num_nodes} | edges: {graph.num_edges}")
    print("\nScene Graph Summary:")
    print(graph.summary())


def cmd_relational(args: argparse.Namespace) -> None:
    loader = ArcDatasetLoader(Path(args.data_dir))
    task = get_task_by_id(loader, args.task_id, args.split)
    if not task:
        print(f"Error: Task {args.task_id} not found.")
        sys.exit(1)

    analyzer = RelationalChangeAnalyzer()
    from src.search.verifier import TaskDemonstration
    demos = [
        TaskDemonstration(ArcGrid(np_grid=ex.input_grid), ArcGrid(np_grid=ex.output_grid))
        for ex in task.train_examples
    ]
    report = analyzer.analyze_task(demos)

    print(f"\n=== Relational Change Analysis: Task {task.task_id} ===")
    print(f"Always preserves shape: {report.always_preserves_shape}")
    print(f"Always preserves colors: {report.always_preserves_colors}")
    print(f"Consistent recolor rules: {report.consistent_recolor_rules}")
    print(f"Motion vectors: {report.motion_vectors}")
    print(f"Summary: {report.summary}")


def cmd_baseline(args: argparse.Namespace) -> None:
    loader = ArcDatasetLoader(Path(args.data_dir))
    tasks = loader.load_training_tasks() if args.split == "training" else loader.load_evaluation_tasks()
    targets = tasks[:args.max_tasks] if args.max_tasks else tasks

    hunter = HunterBaselineSolver()
    solved = 0
    print(f"\nRunning Hunter Baseline on {len(targets)} tasks from {args.split} split...")
    for idx, t in enumerate(targets):
        prog = hunter.solve(t)
        if prog:
            solved += 1
            print(f"  [Solved #{solved}] Task {t.task_id} via {prog.name}")

    print(f"\nHunter Baseline Solved: {solved} / {len(targets)} ({solved/len(targets):.2%})")


def cmd_benchmark(args: argparse.Namespace) -> None:
    loader = ArcDatasetLoader(Path(args.data_dir))
    if args.split == "training":
        tasks = loader.load_training_tasks()
        solutions = loader.load_training_solutions()
    elif args.split == "evaluation":
        tasks = loader.load_evaluation_tasks()
        solutions = loader.load_evaluation_solutions()
    else:
        raise ValueError("Invalid split for benchmarking")

    solver = IntegratedArcSolver(search_timeout_seconds=args.timeout)
    harness = BenchmarkHarness(solver)
    out_dir = Path(args.output_dir) if args.output_dir else Path("reports/benchmarks")

    report = harness.run_benchmark(
        tasks=tasks,
        solutions=solutions,
        split_name=args.split,
        max_tasks=args.max_tasks,
        save_dir=out_dir,
    )
    print("\n" + report.to_markdown_summary())


def cmd_harvest(args: argparse.Namespace) -> None:
    loader = ArcDatasetLoader(Path(args.data_dir))
    tasks = loader.load_training_tasks()
    harvester = BootstrapHarvester()
    out_dir = Path(args.output_dir) if args.output_dir else Path("data/memory")

    solved = harvester.harvest(tasks=tasks, max_tasks=args.max_tasks, save_dir=out_dir)
    print(f"\nHarvesting complete. Solved and indexed {solved} training tasks to {out_dir}.")


def cmd_submit(args: argparse.Namespace) -> None:
    loader = ArcDatasetLoader(Path(args.data_dir))
    test_tasks = loader.load_test_challenges()
    out_path = Path(args.output) if args.output else Path("submissions/submission.json")

    solver = IntegratedArcSolver(search_timeout_seconds=1.5)
    adapter = SubmissionAdapter(solver)

    adapter.generate_submission(
        test_tasks=test_tasks,
        output_path=out_path,
        max_tasks=args.max_tasks,
        dry_run=args.dry_run,
    )
    print(f"\nSubmission generated successfully at {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC-AGI-2 Neuro-Symbolic Research Solver CLI")
    parser.add_argument("--data-dir", default="data/raw", help="Path to raw dataset directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # solve
    p_solve = subparsers.add_parser("solve", help="Solve an ARC task by ID")
    p_solve.add_argument("--task-id", required=True, help="Task ID")
    p_solve.add_argument("--split", default="training", choices=["training", "evaluation", "test"])
    p_solve.add_argument("--timeout", type=float, default=3.5, help="Search timeout in seconds")
    p_solve.set_defaults(func=cmd_solve)

    # perceive
    p_perceive = subparsers.add_parser("perceive", help="Inspect perception and object hypotheses")
    p_perceive.add_argument("--task-id", required=True, help="Task ID")
    p_perceive.add_argument("--split", default="training", choices=["training", "evaluation", "test"])
    p_perceive.set_defaults(func=cmd_perceive)

    # scene-graph
    p_sg = subparsers.add_parser("scene-graph", help="Inspect scene graph representation")
    p_sg.add_argument("--task-id", required=True, help="Task ID")
    p_sg.add_argument("--split", default="training", choices=["training", "evaluation", "test"])
    p_sg.set_defaults(func=cmd_scene_graph)

    # relational
    p_rel = subparsers.add_parser("relational", help="Inspect cross-demonstration relational changes")
    p_rel.add_argument("--task-id", required=True, help="Task ID")
    p_rel.add_argument("--split", default="training", choices=["training", "evaluation", "test"])
    p_rel.set_defaults(func=cmd_relational)

    # baseline
    p_base = subparsers.add_parser("baseline", help="Run Hunter symbolic baseline")
    p_base.add_argument("--split", default="training", choices=["training", "evaluation"])
    p_base.add_argument("--max-tasks", type=int, default=50)
    p_base.set_defaults(func=cmd_baseline)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run full solver benchmark")
    p_bench.add_argument("--split", default="training", choices=["training", "evaluation"])
    p_bench.add_argument("--max-tasks", type=int, default=30)
    p_bench.add_argument("--timeout", type=float, default=2.0)
    p_bench.add_argument("--output-dir", default="reports/benchmarks")
    p_bench.set_defaults(func=cmd_benchmark)

    # harvest
    p_harv = subparsers.add_parser("harvest", help="Harvest memory from training split")
    p_harv.add_argument("--max-tasks", type=int, default=100)
    p_harv.add_argument("--output-dir", default="data/memory")
    p_harv.set_defaults(func=cmd_harvest)

    # submit
    p_sub = subparsers.add_parser("submit", help="Generate Kaggle competition submission JSON")
    p_sub.add_argument("--output", default="submissions/submission.json")
    p_sub.add_argument("--max-tasks", type=int, default=None)
    p_sub.add_argument("--dry-run", action="store_true", help="Fast dry-run mode")
    p_sub.set_defaults(func=cmd_submit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
