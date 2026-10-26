# ARC-AGI-2 Solver Failure Diagnosis Report

**Document**: `reports/solver_failure_diagnosis.md`  
**Repository**: `/home/naya/Documents/ARC-Prize-2`  
**Author**: Antigravity Research Engineer  
**Date**: October 17, 2026  
**Status**: Comprehensive Empirical Investigation Complete  

---

## 1. Executive Reconciliation: Benchmark (0/50) vs. Claimed Task Solves (3 Tasks)

### 1.1 Reproducibility of Claimed Task Solves
The three claimed training task solutions reported in the previous development phase were subjected to strict, independent reproducibility audits using `IntegratedArcSolver` with explicit comparison against the ground-truth solutions in `data/raw/arc-agi_training_solutions.json`:

| Task ID | Training Set Index | Solver Stage | Verified Best Program | Test Attempt 1 Match | Test Attempt 2 Match | Reproducible? |
|---|---|---|---|---|---|---|
| `1cf80156` | Index 88 | `hunter_baseline` | `crop_to_content` | **True (100%)** | **True (100%)** | **Yes** |
| `1e0a9b12` | Index 93 | `hunter_baseline` | `gravity_down` | **True (100%)** | **True (100%)** | **Yes** |
| `1f85a75f` | Index 100 | `hunter_baseline` | `crop_to_largest_object` | **True (100%)** | **True (100%)** | **Yes** |

All three tasks achieve 100% exact test pixel match on both Attempt 1 and Attempt 2. The results are completely reproducible.

### 1.2 Resolving the Discrepancy
The apparent contradiction between the benchmark reporting **0 / 50 tasks solved** (`reports/benchmarks/benchmark_training.json`) and the 3 verified task solutions is explained by dataset slice indexing:

1. **Benchmark Slice**: In `reports/benchmarks/benchmark_training.json`, the benchmark harness was invoked with `max_tasks=50` on the training dataset. `ArcDatasetLoader.load_training_tasks()` iterates through `data/raw/arc-agi_training_challenges.json` in dictionary key order, evaluating **only the first 50 tasks (indices 0 through 49)**:
   - Tasks attempted in benchmark: `00576224` (idx 0), `007bbfb7` (idx 1), `009d5c81` (idx 2), `00d62c1b` (idx 3) ... `178fcbfb` (idx 49).
2. **Task Locations**: The three solvable tasks are located beyond index 49:
   - `1cf80156` is at **Index 88**
   - `1e0a9b12` is at **Index 93**
   - `1f85a75f` is at **Index 100**
3. **Conclusion**: Tasks `1cf80156`, `1e0a9b12`, and `1f85a75f` were **never executed** as part of the 50-task benchmark run. The 0 / 50 result on tasks 0–49 is mathematically consistent with the successes at indices 88, 93, and 100.

---

## 2. End-to-End Task Trace: Task `00d62c1b` (Training Index 3)

To isolate why the solver fails on tasks that should be within its capability, task `00d62c1b` was selected for a line-by-line execution trace through the live production solver.

### 2.1 Task Specifications
- **Input Grids**: Discrete green (color 3) closed loops and open lines on a black (color 0) background.
- **Ground Truth Transformation**: All enclosed 0-holes bounded by green (3) structures are filled with yellow (color 4); background and outer cells remain black (0).
- **Demonstrations**: 5 training pairs with varying dimensions: `(10, 10)`, `(10, 10)`, `(10, 10)`, `(20, 20)`, `(6, 6)`.
- **Test Input**: 1 query grid of size `(20, 20)`.

### 2.2 Trace Through Solver Stages

```
STAGE 1: Hunter Symbolic Baseline
- Tested: identity, rigid symmetries, crops, 1-to-1 color substitution, gravity, symmetry completion.
- Result: None satisfied 100% demonstration match. (Hunter does not test hole filling).

STAGE 2: Object-Centric Perception & Relational Change Analysis
- PerceptionEngine detected:
  - Input: 5 green objects (1 large irregular loop with 1 hole, 4 single-pixel dots).
  - Output: 8 objects (5 original green objects + 3 new yellow objects inside loop bounds).
- RelationalChangeAnalyzer extracted:
  - same_shape_preserved: True
  - consistent_recolor_rules: {} (green is preserved, yellow is newly introduced)
  - always_creates_or_destroys: True (new yellow objects appear)
  - motion_vectors: []

STAGE 3: Concept Library & Structural Memory
- Concept library: 3 skills evaluated (`crop_to_content`, `gravity_down`, `crop_to_largest_object`).
- Precondition filter: `gravity_down` requires same shape (tested, 0.0 accuracy). Others rejected or failed.

STAGE 4: Relational Symbolic Hypotheses
- LLMSynthesizerInterface evaluated rules based on `always_creates_or_destroys`:
  - Generated: `relational_keep_largest_object`, `relational_keep_smallest_object`.
  - Verification: 0.0% accuracy on demonstrations.

STAGE 5: DSL Beam Search & Constraint Filtering
- ConstraintFilter: Retained shape-preserving primitives (41 admissible).
- Candidate Generation:
  - Enumerable zero-parameter primitives: generated calls for 35 primitives.
  - Parameterized color swaps: generated `replace_color(3, 4)` and `replace_color(0, 4)`.
- Critical Failure in Candidate Generation:
  - `fill_enclosed` and `fill_enclosed_holes` were enumerated with ZERO arguments:
    `PrimitiveCall("fill_enclosed_holes", {})`.
  - Execution raised: `TypeError: fill_enclosed() missing 1 required positional argument: 'fill_color'`.
  - DemonstrationVerifier recorded `pixel_accuracy = 0.0` due to uncaught TypeError.
  - No candidate was ever generated with `fill_color=4`.
- Selection & Ranking:
  - `identity` matched 94.14% of training demonstration pixels by preserving background and boundaries.
  - Score formula awarded `identity`: `2.0 * 1.0 (shape match) + 0.9414 - 0.05 * 1 = 2.891`.
  - `identity` was ranked #1 and chosen as `best_program`.

STAGE 6: Test Output Prediction & Ground Truth Evaluation
- Attempt 1: Executed `identity` on test input `(20, 20)`.
- Attempt 2: Defaulted to `attempt_1`.
- Ground truth evaluation: 33 mismatched pixels (the exact 33 enclosed hole cells).
- Result: Task marked UNSOLVED (0 / 1 test inputs correct).
```

### 2.3 Grid Comparison: Ground Truth vs. Attempt 1 Prediction

#### Test Input Grid (`20 x 20`):
```
. . . . . . . . . . . . . . . . . . . .
. . 3 . . . . . . . . . . . . . . . . .
. 3 . 3 3 . . . . . . . . . . . . . . .
. . 3 . 3 3 3 3 3 . 3 3 . . . . . . . .
. . . . 3 . . . . 3 . . 3 . . . . . . .
. . . . 3 3 3 3 3 . 3 3 3 . . . . . . .
. . . . . . . . . . . . . 3 3 3 3 3 . .
. . . . . . . . . . . . . 3 . . . 3 . .
. . . . . . . . . . . . . 3 . . . 3 . .
. . . . . . . . . 3 3 3 3 3 . . . 3 . .
. . . . . . . . . 3 . . . 3 . . . 3 . .
. . . . . . . . 3 3 3 3 3 3 . . . 3 . .
. . . . . . 3 3 . 3 . . . 3 3 3 3 3 . .
. . 3 . . . . . 3 3 . . . . . . . . . .
. 3 . 3 . . . . . . . . . . . . . . . .
. . 3 . 3 . 3 3 3 3 3 3 . . . . . . . .
. . . . . . . 3 . . . 3 . . . . . . . .
. . . . . . . 3 . . . 3 . . . . . . . .
. . . . . . . 3 3 3 3 3 . . . . . . . .
. . . . . . . . . . . . . . . . . . . .
```

#### Expected Ground Truth (`20 x 20`):
```
. . . . . . . . . . . . . . . . . . . .
. . 3 . . . . . . . . . . . . . . . . .
. 3 4 3 3 . . . . . . . . . . . . . . .
. . 3 . 3 3 3 3 3 . 3 3 . . . . . . . .
. . . . 3 4 4 4 4 3 4 4 3 . . . . . . .
. . . . 3 3 3 3 3 . 3 3 3 . . . . . . .
. . . . . . . . . . . . . 3 3 3 3 3 . .
. . . . . . . . . . . . . 3 4 4 4 3 . .
. . . . . . . . . . . . . 3 4 4 4 3 . .
. . . . . . . . . 3 3 3 3 3 4 4 4 3 . .
. . . . . . . . . 3 4 4 4 3 4 4 4 3 . .
. . . . . . . . 3 3 3 3 3 3 4 4 4 3 . .
. . . . . . 3 3 4 3 . . . 3 3 3 3 3 . .
. . 3 . . . . . 3 3 . . . . . . . . . .
. 3 4 3 . . . . . . . . . . . . . . . .
. . 3 . 3 . 3 3 3 3 3 3 . . . . . . . .
. . . . . . . 3 4 4 4 3 . . . . . . . .
. . . . . . . 3 4 4 4 3 . . . . . . . .
. . . . . . . 3 3 3 3 3 . . . . . . . .
. . . . . . . . . . . . . . . . . . . .
```

#### Predicted Output by Solver Attempt 1 (`20 x 20`):
Identical to Test Input (`identity`). Mismatched on all 33 interior pixels (marked `4` in Ground Truth, but left as `0`).

---

## 3. Analysis and Definition of Evaluation Metrics

| Metric | Source Code Location | Evaluation Domain | Exact Mathematical Definition | Diagnostic Purpose & Failure Risk |
|---|---|---|---|---|
| **`top_accuracy`** | `src/search/beam_search.py:362` via `src/search/verifier.py:181` | **Training demonstrations only** (Never test input) | $\frac{1}{N} \sum_{i=1}^N \frac{\sum (P_i == T_i)}{\text{size}(T_i)}$ where $P_i, T_i$ are demonstration prediction and target arrays. | Measures average pixel match on train pairs. **Risk**: High values (e.g. 0.94) are trivially achieved by `identity` when background dominates (>90% cells), creating false confidence. |
| **`top_score`** | `src/search/beam_search.py:233-237` | **Training demonstrations only** | $2.0 \times \frac{\text{shape\_matches}}{N} + \text{top\_accuracy} - 0.05 \times \text{complexity}$ | Heuristic objective for beam ranking. **Risk**: Grants +2.0 to any shape-preserving candidate, causing degenerate programs (`identity`) to dominate search. |
| **`demonstration consistency` (`is_fully_solved`)** | `src/search/verifier.py:185` | **All training demonstrations** | $\bigwedge_{i=1}^N \left( \text{shape}(P_i) == \text{shape}(T_i) \land \sum (P_i \ne T_i) == 0 \right)$ | Strict boolean check requiring 100% exact pixel match across every train pair before declaring task solved at train time. |
| **`task-level solved status` (`is_task_solved`)** | `src/benchmark/harness.py:138-139` | **Held-out test queries against hidden ground truth** | $\left( \sum_{k=1}^M \mathbb{I}(\text{att1}_k == G_k \lor \text{att2}_k == G_k) == M \right) \land M > 0$ | Official ARC competition rule: Every test input $M$ must be 100% matched by at least one of the 2 attempts. |
| **`test-input accuracy`** | `src/benchmark/harness.py:170` | **Held-out test queries across all tasks** | $\frac{\text{Total Test Inputs Correct}}{\text{Total Test Inputs Attempted}}$ | Granular competition score tracking sub-task performance. |

---

## 4. Candidate Search and Primitive Audit

An audit was conducted on `PRIMITIVE_REGISTRY` (61 primitives) and `DSLBeamSearch` candidate generation.

### 4.1 Missing Parameterization and Uncaught TypeErrors
When `DSLBeamSearch` constructs depth-1 candidates, it iterates over all primitive names in `PRIMITIVE_REGISTRY` and excludes only a hardcoded set of 9 names:
```python
if name in {"replace_color", "crop", "pad", "scale_up", "tile_grid", 
            "keep_only_color", "remove_color", "keep_largest_color", "keep_smallest_color"}:
    continue
```
Because the exclusion set is incomplete, **27 of 61 registered primitives require parameters but are called with zero arguments**:
1. **Parameterized Primitives called with zero args**:
   - `fill_enclosed`, `fill_enclosed_holes` (requires `fill_color`)
   - `move_up`, `move_down`, `move_left`, `move_right` (requires `color_id`)
   - `slide_up`, `slide_down`, `slide_left`, `slide_right` (requires `color_id`)
   - `hollow_out` (requires `color_id`)
   - `color_top_half`, `fill_with_color` (requires `color`)
   - `extract_color_bounding_box` (requires `target_color`)
   - `flood_fill` (requires `r, c, new_color`)
   - `overlay` (requires `grid_fg`)
   - `draw_line_hv` (requires `r1, c1, r2, c2, color_id`)
2. **Defective Lambda Signatures in `DSL_REGISTRY`**:
   - `complete_symmetry_horizontal`: lambda passes `axis="horizontal"`, but `complete_symmetry` accepts no keyword arguments. Throws `TypeError`.
   - `complete_symmetry_vertical`: lambda passes `axis="vertical"`. Throws `TypeError`.
   - `scale_up`: lambda passes `factor_h=factor, factor_w=factor`, but `scale_grid` accepts `(grid, multiplier)`. Throws `TypeError`.
   - `pad`: lambda passes `pad_h=pad_width, pad_w=pad_width`, but `pad_grid` accepts `(grid, pad_size, pad_color=0)`. Throws `TypeError`.

### 4.2 Impact on Program Synthesis
On every search step, these 27 primitives are generated as invalid calls, execute, throw `TypeError`, receive `pixel_accuracy = 0.0`, and are discarded. Simultaneously, valid parameter assignments (such as `fill_enclosed(fill_color=4)` for task `00d62c1b`) are **never generated**.

---

## 5. Architectural Integration Findings

Every architectural subsystem was inspected for its actual runtime impact on final test predictions:

| Subsystem Component | Production Code Location | Call Path / Runtime Execution Status | Classification | Impact on Prediction |
|---|---|---|---|---|
| **Object-Centric Perception** (`PerceptionEngine`, `ArcObject`) | `src/perception/detector.py:166` | Called in `IntegratedArcSolver:135` via `relational_analyzer.analyze_task()`. Segments objects for train demonstrations. | **B** | Executed, but its outputs are only routed to relational change heuristics. |
| **Perception Hypotheses** (`PerceptionHypothesisGenerator`) | `src/perception/hypotheses.py:25` | Instantiated in `IntegratedArcSolver:77` (`self.hypothesis_gen`). **Never called** in `solve()`. | **C** | Implemented but completely inactive in solver pipeline. |
| **Scene Graph Builder** (`ArcSceneGraph`, `SceneGraphBuilder`) | `src/scene_graph/builder.py:13` | Instantiated in `IntegratedArcSolver:78` (`self.graph_builder`). **Never called** in `solve()`. Line 114 also contains an unimported `scipy` reference. | **C** | Implemented but completely inactive in solver pipeline. |
| **Correspondence Matcher** (`ObjectCorrespondenceMatcher`) | `src/correspondence/matcher.py:20` | Called inside `RelationalChangeAnalyzer` to pair demonstration objects across input/output. | **B** | Executed, but correspondences do not directly constrain or generate programs. |
| **Relational Change Analyzer** (`RelationalChangeReport`) | `src/correspondence/relational_change.py:25` | Called in `IntegratedArcSolver:135`. Generates `change_report` passed to `LLMSynthesizerInterface`. | **B** | Executed. Generates candidates in Stage 4, but if they fail exact train fit, they are discarded and omitted from beam search. |
| **Structural Fingerprint** (`task_to_fingerprint`) | `src/fingerprint/encoder.py:90` | Called in `IntegratedArcSolver:187`. Extracts 20D vector barcode to query vector memory. | **A** | Executed. Directly drives memory retrieval. |
| **MirrorMatrix Retrieval** | `src/memory/mirror_matrix.py:83` | Called in `IntegratedArcSolver:188`. Queries nearest fingerprint matches and verifies programs. | **A** | Executed. Solves task immediately on exact match; adds candidates to beam pool on $>0.6$ accuracy. |
| **Concept Library** | `src/solvers/concept_library.py:63` | Called in `IntegratedArcSolver:167`. Queries skills matching task shape preconditions. | **A** | Executed. Solves task immediately on exact match. |
| **Hunter Baseline** | `src/solvers/hunter.py:22` | Called in `IntegratedArcSolver:117` (Stage 1). Tests fast templates before any search. | **A** | Executed. **Sole source of all 3 confirmed task solves** (`1cf80156`, `1e0a9b12`, `1f85a75f`). |
| **Bootstrap Harvester** | `src/solvers/bootstrap.py:25` | Offline tool invoked via `src/cli.py harvest`. Indexes solved tasks into memory and library. | **A** | Executed offline. Successfully populated memory with 3 verified programs. |
| **DSL Beam Search** | `src/search/beam_search.py:41` | Called in `IntegratedArcSolver:231` (Stage 5). Core enumerative and beam search engine. | **A** | Executed. Generates final fallback attempts when earlier stages do not find a solution. |
| **REPL Refinement** | `src/search/repl_refiner.py:25` | Called in `DSLBeamSearch:209, 318`. Attempts color remapping and cropping on near-misses ($>40\%$). | **A** | Executed during search beam expansion. |

### Classification Key:
- **A**: Executed and affects candidate generation, ranking, or prediction.
- **B**: Executed but does not affect the final prediction.
- **C**: Implemented in codebase but never called by the solver pipeline.
- **D**: Not functionally implemented.
- **E**: Unknown / insufficient evidence.

---

## 6. Small Controlled Empirical Test: Five Representative Tasks

Five representative training tasks were evaluated under standard solver constraints (`search_timeout = 2.0s`). Ground truth solutions were isolated strictly to the evaluation harness:

| Task ID | Index | Input $\to$ Output Dimensions | Correct Transformation Program | Was Correct Program Generated? | Did It Fit All Train Demos? | Was It Selected by Solver? | Test Exact Match (Att 1 / Att 2) | First Point of Failure |
|---|---|---|---|---|---|---|---|---|
| `00576224` | 0 | `(2, 2) -> (6, 6)` | 3x3 tiled expansion with selective quadrant reflections | No | N/A | No (`rotate_90` chosen) | False / False | **DSL Expressivity**: Compound tiling/reflection cannot be represented in depth 2. `scale_up` lambda crashed with TypeError. |
| `007bbfb7` | 1 | `(3, 3) -> (9, 9)` | Kronecker fractal substitution (replace pixel with 3x3 subgrid) | No | N/A | No (`rotate_90` chosen) | False / False | **DSL Expressivity**: Fractal / recursive Kronecker product primitive does not exist in DSL. |
| `009d5c81` | 2 | `(14, 14) -> (14, 14)` | Detect multi-color components, extract dominant color, recolor by size | No | N/A | No (`keep_largest_object` chosen) | False / False | **Search & Objective Bias**: Correct program requires object-attribute conditioning. `keep_largest_object` matched 88.3% background cells coincidentally. |
| `00d62c1b` | 3 | `(10, 10)/(20, 20) -> same` | `fill_enclosed(grid, fill_color=4)` | **No** (called with 0 args) | **Yes** (when tested with `fill_color=4`, fits 100% demos and test) | No (`identity` chosen) | False / False | **Primitive Parameterization**: `fill_enclosed` crashed with `TypeError`. Never parameterized with `out_colors` (color 4). |
| `00dbd492` | 4 | `(15, 15)/(9, 9) -> same` | Connect corner rectangular boundary frames along diagonals | No | N/A | No (`identity` chosen) | False / False | **DSL Expressivity**: Diagonal boundary interpolation primitive not present in DSL. |

---

## 7. Uncertain Findings and Open Questions

1. **Upper Bound on Pure Search Depth**:
   While fixing primitive signatures and parameterization will immediately solve tasks like `00d62c1b`, expanding beam search from depth 2 to depth 3 without strong relational pruning causes combinatorial explosion ($60^3 = 216,000$ programs, exceeding per-task timeouts).
2. **Object-to-Grid Abstraction Gap**:
   Tasks `009d5c81` and `00dbd492` require reasoning directly over individual object properties (e.g. "for each object, count its neighbors and set color"). Because `ArcSceneGraph` is currently disconnected from `DSLBeamSearch`, the solver has no mechanism to synthesize object-level lambda functions.

---

## 8. Summary of Findings & Recommended Corrective Action

### Three Most Important Findings:
1. **Reconciliation Resolved**: The 3 claimed task solves (`1cf80156`, `1e0a9b12`, `1f85a75f`) are 100% reproducible. They coexisted with a 0/50 benchmark score because they reside at dataset indices 88, 93, and 100, which fell outside the 50-task benchmark evaluation window (indices 0–49).
2. **Missing Parameterization & Broken Primitives**: 27 of 61 primitives in `PRIMITIVE_REGISTRY` require arguments but were called with zero arguments by `DSLBeamSearch`, crashing with `TypeError` on every execution. In particular, `fill_enclosed` was never parameterized with `out_colors`, causing an immediate solve on task `00d62c1b` to be missed. Additionally, 4 registry lambdas (`complete_symmetry_horizontal`, `complete_symmetry_vertical`, `scale_up`, `pad`) contain defective keyword arguments.
3. **Disconnected Architectural Modules**: `SceneGraphBuilder` and `PerceptionHypothesisGenerator` are completely uncalled during solver execution. Relational candidates from `RelationalChangeAnalyzer` are discarded if they do not 100% solve the task at Stage 4, leaving whole-grid beam search to fall back to `identity` on high background overlap (e.g. 94% `top_accuracy`).

### Single Next Engineering Change:
**Harden Primitive Signatures and Parameterize Hole-Filling in `DSLBeamSearch`**:
1. Correct the defective lambda arguments in `src/dsl/primitives.py` (`complete_symmetry`, `scale_grid`, `pad_grid`).
2. Update `src/search/beam_search.py` candidate generation to automatically instantiate `fill_enclosed` (and `fill_enclosed_holes`) using candidate colors discovered in `out_colors - in_colors` (e.g., `fill_enclosed(color=c)`).
3. Update the exclusion list in `DSLBeamSearch` to prevent all 27 parameterized primitives from being invoked with zero arguments.
