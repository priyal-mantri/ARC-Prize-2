# ARC-AGI-2 Research Solver Workspace — Phase 1

This repository hosts a clean, modular, research-oriented ARC-AGI-2 solver architecture.

In Phase 1, the focus is strictly on **dataset discovery, structural audit, data pipeline integrity, and reproducible visualization**. No heuristic solver architectures, LLMs, or neural networks are implemented during this phase.

---

## Workspace Structure

```
ARC-Prize-2/
├── data/
│   └── raw/                       # Canonical competition JSON files (read-only)
│       ├── arc-agi_training_challenges.json
│       ├── arc-agi_training_solutions.json
│       ├── arc-agi_evaluation_challenges.json
│       ├── arc-agi_evaluation_solutions.json
│       ├── arc-agi_test_challenges.json
│       └── sample_submission.json
├── notebooks/
│   └── 01_dataset_audit.ipynb     # Fully executed 16-section reproducible audit notebook
├── reports/
│   └── dataset_audit.md           # Comprehensive technical report of audit findings
├── src/
│   ├── __init__.py
│   ├── visualization.py           # Canonical 10-color ARC plotting and rendering tools
│   └── data/
│       ├── __init__.py
│       ├── dataset_loader.py       # Safe discovery, caching, and ArcTask loading
│       ├── statistics.py          # Geometric, color, and distribution analytics
│       └── validation.py          # Structural integrity and rectangular grid checks
├── tests/
│   └── test_data_pipeline.py      # Automated unit and integration tests (9 tests)
├── pytest.ini                     # Pytest configuration with pythonpath = .
└── README.md                      # Project documentation and architectural overview
```

---

## Safety & Data Integrity Guarantees

1. **Zero Ground-Truth Leakage:** The `ArcTask` class exposes only training demonstration pairs and test input grids. Test output grids are never stored in or returned by `ArcTask`.
2. **Strict Evaluation Isolation:** Ground-truth solutions can only be retrieved via `loader.load_ground_truth(task_id)`, explicitly designated for offline evaluation benchmarks and never accessible to solver reasoning modules.
3. **Immutability:** Grid data is encapsulated in frozen dataclasses with numpy arrays.
4. **Reproducibility:** Paths are auto-discovered dynamically relative to the repository root. No absolute paths are hardcoded.
5. **No Data Modification:** The original dataset files and task IDs are completely preserved without alteration or fabrication.

---

## Quickstart

### 1. Environment Setup

The repository uses Python 3.12 with standard scientific packages:
```bash
# Verify environment dependencies
python3 -c "import numpy, matplotlib; print('Core dependencies OK')"
```

### 2. Running Automated Tests

Run the full test suite using `pytest`:
```bash
pytest tests/
```
All 9 unit and integration tests verify:
- Complete file discovery of all 6 canonical files.
- Exact task counts (1,000 train, 120 eval, 240 test).
- Zero task ID overlap between training and evaluation splits.
- Strict isolation of evaluation ground-truth.
- Robust detection of ragged, malformed, or out-of-range grids.
- Deterministic calculation of descriptive statistics.

### 3. Loading a Task in Python

```python
from src.data.dataset_loader import ArcDatasetLoader
from src.visualization import plot_task
import matplotlib.pyplot as plt

loader = ArcDatasetLoader()

# Load a task (safely exposes only train pairs and test inputs)
task = loader.load_task("00576224", split="training")
print(task)
# ArcTask(task_id='00576224', split='training', train_pairs=3, test_inputs=1)

# Render using the canonical 10-color ARC palette
fig = plot_task(task)
plt.show()
```

### 4. Exploring the Dataset Audit Notebook

Launch JupyterLab to inspect the interactive audit:
```bash
jupyter lab notebooks/01_dataset_audit.ipynb
```
The notebook covers 16 comprehensive sections including schema validation, grid dimension histograms, color distributions, multi-test query analysis, and representative puzzle visualizations.

---

## Summary of Dataset Audit Findings

| Property | Training Split | Evaluation Split | Public Offline Test Split |
| :--- | :---: | :---: | :---: |
| **Tasks** | **1,000** | **120** | **240** (subset of train) |
| **Demonstration Pairs** | 3,232 (mean: 3.23) | 358 (mean: 2.98) | 767 (mean: 3.20) |
| **Test Inputs** | 1,076 (mean: 1.08) | 172 (mean: 1.43) | 259 (mean: 1.08) |
| **Available Solutions** | 1,076 grids | 172 grids | 0 (hidden) |
| **Multi-Test Tasks (>1 test)** | **69 (6.9%)** | **49 (40.8%)** | **17 (7.1%)** |
| **Same Dimension Rate** | 65.4% | 69.0% | 67.7% |
| **Resized Dimension Rate** | 34.6% | 31.0% | 32.3% |
| **Square Grid Rate** | 62.0% | 60.4% | 61.8% |
| **Malformed Tasks** | **0 (100% clean)** | **0 (100% clean)** | **0 (100% clean)** |
| **Grid Dimensions (H × W)** | $[1, 30] \times [1, 30]$ | $[4, 30] \times [3, 30]$ | $[2, 30] \times [2, 30]$ |
| **Colors Present** | Canonical 10 (0..9) | Canonical 10 (0..9) | Canonical 10 (0..9) |

---

## Phase 2 Roadmap: Solver Foundations

Following the completion of the Phase 1 audit, Phase 2 will focus on:
1. **Grid Primitives & Morphological Segmentation:** Object detection, connected component extraction, color clustering, and bounding box computation.
2. **Geometry & Symmetry Invariants:** Reflection, rotation, crop, pad, tiling, and affine spatial transformations.
3. **Execution Engine & DSL Definition:** Safe, typed domain-specific primitives for grid transformation.
4. **Offline Benchmarking Harness:** Multi-attempt submission generator and validation metrics evaluator.
