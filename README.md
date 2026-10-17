# ARC-AGI-2 Complete Neuro-Symbolic Research Solver

An end-to-end neuro-symbolic research solver for the **ARC-AGI-2** (Abstraction and Reasoning Corpus) competition. Combines object-centric perception, graph representations, cross-demonstration relational reasoning, domain-specific language (DSL) program synthesis, memory retrieval, and REPL failure refinement.

---

## 1. Architectural Pipeline

```
DATASET (Training / Evaluation / Test)
   │
   ▼
CANONICAL TASK NORMALIZATION (ArcGrid, BoundingBox)
   │
   ▼
OBJECT-CENTRIC PERCEPTION (PerceptionEngine, 4-conn & 8-conn components)
   │
   ▼
PERCEPTION HYPOTHESES (mono_4conn, mono_8conn, color_partition, multicolor)
   │
   ▼
SCENE GRAPH REPRESENTATION (ArcSceneGraph, NetworkX DiGraph)
   │
   ▼
CROSS-EXAMPLE CORRESPONDENCE (ObjectCorrespondenceMatcher)
   │
   ▼
RELATIONAL CHANGE INVARIANTS (RelationalChangeAnalyzer)
   │
   ▼
STRUCTURAL FINGERPRINTS (20-dimensional topological barcode)
   │
   ▼
MEMORY & SKILL RETRIEVAL (MirrorMatrix, ConceptLibrary)
   │
   ▼
HUNTER SYMBOLIC BASELINE & RELATIONAL HYPOTHESES
   │
   ▼
DSL PROGRAM SYNTHESIS (Beam Search over 45+ typed primitives)
   │
   ▼
CONSTRAINT FILTERING (Pruning shape/color inconsistent branches)
   │
   ▼
DEMONSTRATION VERIFIER & REPL REFINEMENT (Exact pixel verification)
   │
   ▼
CANDIDATE RANKING & MULTI-TEST INFERENCE (2 attempts per test query)
   │
   ▼
KAGGLE SUBMISSION ADAPTER (Schema-validated JSON output)
```

---

## 2. Directory Structure

```
ARC-Prize-2/
├── data/
│   ├── raw/                  # Official competition JSON challenge & solution files
│   └── memory/               # Harvested MirrorMatrix barcodes & ConceptLibrary skills
├── src/
│   ├── representation/       # Canonical immutable ArcGrid & BoundingBox domain models
│   ├── perception/           # Object segmentation, background inference & hypotheses
│   ├── scene_graph/          # Attributed directed scene graphs & spatial relations
│   ├── correspondence/       # Bipartite object matching & cross-demo relational changes
│   ├── fingerprint/          # 20D topological & geometric feature barcode encoder
│   ├── memory/               # MirrorMatrix vector memory bank & FAISS/NumPy search
│   ├── dsl/                  # 45+ deterministic primitives, DSLProgram, & execution
│   ├── search/               # ConstraintFilter, Verifier, REPLRefiner, & DSLBeamSearch
│   ├── solvers/              # HunterBaseline, BootstrapHarvester, & IntegratedArcSolver
│   ├── submission/           # Kaggle submission adapter and format validator
│   ├── benchmark/            # Reproducible benchmark harness and metrics
│   ├── visualization.py      # Canonical 10-color ARC colormap rendering
│   └── cli.py                # Unified CLI entrypoint
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_perception_and_relational_reasoning.ipynb
│   └── 03_solver_benchmark_and_diagnostics.ipynb
├── tests/                    # 33 unit and integration tests (100% passing)
├── reports/
│   ├── benchmarks/           # Empirical benchmark evaluation JSON and Markdown
│   ├── complete_architecture_report.md
│   ├── dataset_audit.md
│   └── dataset_provenance_review.md
├── submissions/              # Generated submission.json files
└── pytest.ini                # Pytest configuration
```

---

## 3. Installation & Environment

The architecture runs on Python 3.10+ using standard scientific computing libraries. All dependencies run 100% offline without requiring internet access or GPU weights.

```bash
# Clone or navigate to the repository
cd /home/naya/Documents/ARC-Prize-2

# Install dependencies (if not already installed)
pip install numpy scipy networkx matplotlib pytest nbformat nbclient
```

---

## 4. Command-Line Interface (CLI)

The unified CLI provides 8 core subcommands for solving, inspecting, benchmarking, and submitting.

### Solve a Single Task
```bash
# Solve real task 1cf80156 (solved via Hunter Baseline: crop_to_content)
python3 src/cli.py solve --task-id 1cf80156 --split training

# Solve real task 1e0a9b12 (solved via DSL Beam Search: gravity_down)
python3 src/cli.py solve --task-id 1e0a9b12 --split training --timeout 2.0
```

### Inspect Object-Centric Perception
```bash
python3 src/cli.py perceive --task-id 007bbfb7 --split training
```

### Inspect Scene Graph Construction
```bash
python3 src/cli.py scene-graph --task-id 007bbfb7 --split training
```

### Inspect Cross-Demonstration Relational Changes
```bash
python3 src/cli.py relational --task-id 007bbfb7 --split training
```

### Run Hunter Symbolic Baseline
```bash
python3 src/cli.py baseline --split training --max-tasks 50
```

### Run Benchmark with Ground-Truth Evaluation
```bash
# Training split subset benchmark
python3 src/cli.py benchmark --split training --max-tasks 50 --timeout 1.0

# Held-out evaluation split benchmark
python3 src/cli.py benchmark --split evaluation --max-tasks 50 --timeout 1.0
```

### Harvest Memory from Training Split
```bash
python3 src/cli.py harvest --max-tasks 100 --output-dir data/memory
```

### Generate Competition Submission
```bash
python3 src/cli.py submit --output submissions/submission.json --dry-run
```

---

## 5. Running Tests

Run the complete test suite across all 8 test modules:

```bash
python3 -m pytest tests/ -v
```

All 33 unit and integration tests execute in ~9 seconds and pass 100%.

---

## 6. Jupyter Notebooks

1. **`notebooks/01_dataset_audit.ipynb`**: Complete 16-section reproducible dataset audit, task statistics, and split analysis.
2. **`notebooks/02_perception_and_relational_reasoning.ipynb`**: Multi-hypothesis perception, scene graph nodes and edges, and invariant transformation discovery.
3. **`notebooks/03_solver_benchmark_and_diagnostics.ipynb`**: Baseline vs. synthesis solving demonstration on real tasks (`1cf80156`, `1e0a9b12`), and submission pipeline validation.

---

## 7. Submission Pipeline & Rules Compliance

- **No Data Leakage**: Ground truth solutions are strictly separated from solver task representations.
- **Multi-Test Handling**: Iterates over all test queries per task, generating `attempt_1` and `attempt_2`.
- **Offline Self-Contained**: Operates without live internet connections, remote API calls, or uninstalled models.
- **Preserved Archive**: The legacy reference project `/home/naya/Documents/ARC - 2` remains 100% read-only and untouched.
