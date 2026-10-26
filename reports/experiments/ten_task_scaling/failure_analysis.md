# Failure Analysis: 10-Task Candidate Budget Scaling Experiment

## 1. Overview of Bottlenecks Across Scaling Budgets (20 / 40 / 60)

The 10-task scaling experiment reveals that candidate budget scaling from 20 to 40 to 60 yields **identical task accuracy (6/10 solved, 60.0%)** within the current depth-2 beam search configuration. Increasing `NUM_GUESSES` allows the beam search to explore more candidate parameterizations and depth-2 primitive pairs, but the four remaining unsolved tasks fail due to structural bottlenecks rather than candidate budget exhaustion.

| Task ID | Category | Bottleneck Level | Concrete Architectural Cause |
|---|---|---|---|
| `d037b0a7` | movement_gravity_extrusion | Level 1: Missing Primitive | Requires downward column extrusion (forward fill from non-zero cells to grid bottom). Existing `gravity` drops pixels like falling blocks rather than extruding lines. |
| `009d5c81` | object_relational | Level 2: Search Depth Exhaustion | Requires 3-step composition (`segment_objects -> find_adjacent_neighbors -> recolor_by_mask`). Depth-2 beam search truncated before depth 3. |
| `00dbd492` | complex_structural | Level 2: Search Depth Exhaustion | Requires multi-region topological partitioning with conditional flood-fills. Exceeds depth-2 composition. |
| `445eab21` | object_abstraction_dimension_reduction | Level 1 & 2: Macro-Abstraction | Requires extracting nested rectangular frames, ranking by area, and projecting dominant color into a fixed 2x2 grid. |

---

## 2. Evidence-Based Categorization of Unsolved Tasks

### Task `d037b0a7` (Column Downward Extrusion)
- **Observations:** In each column, non-zero values project downward to fill all cells below them down to the bottom border.
- **Diagnostics:**
  - Evaluated candidates included `complete_symmetry_vertical`, `gravity`, and `rot90`.
  - `gravity` fails because it shifts `[[0, 2, 0], [7, 0, 8], [0, 0, 0]]` into `[[0, 0, 0], [0, 0, 0], [7, 2, 8]]` (Tetris dropping), whereas the target is `[[0, 2, 0], [7, 2, 8], [7, 2, 8]]`.
- **Classification:** **Level 1 (Missing Primitive: `extrude_downwards`)**.
- **Remediation:** Introduce a principled directional projection primitive in a future version.

### Task `009d5c81` (Relational Color Rule)
- **Observations:** Multi-color $14	imes 14$ grid where isolated pixels are recolored based on spatial adjacency to primary foreground structures.
- **Diagnostics:**
  - Best candidate found: `keep_largest_object(g)` (Pixel accuracy: 0.826).
  - Failed exact match on all training demonstrations.
  - Required program is a composite relational pipeline:
    1. Detect connected components with 8-way connectivity.
    2. Compute Euclidean adjacency between distinct color components.
    3. Conditionally re-color adjacent pixels.
- **Classification:** **Level 2 (Search Depth Exhaustion: Depth $\ge 3$ Required)**.
- **Remediation:** Neural guidance from Qwen2.5-Coder to propose macro-relational hypotheses, or expanding search depth to 3 with aggressive constraint pruning.

### Task `00dbd492` (Complex Multi-Region Structural Filling)
- **Observations:** Complex $15	imes 15$ maze-like structure with multiple partitioned cavities.
- **Diagnostics:**
  - Best candidate found: `fill_enclosed(g, fill_color=4)` (Pixel accuracy: 0.742).
  - Failed because only specific enclosed cavities are filled, conditioned on cavity boundary colors.
- **Classification:** **Level 2 (Search Depth Exhaustion: Conditioned Topological Filling)**.

### Task `445eab21` (Object Abstraction & Dimension Reduction)
- **Observations:** Input grids ($10	imes 10$) contain nested rectangular frames of varying colors. Output is a $2	imes 2$ block of the color belonging to the frame with the maximum bounding box area.
- **Diagnostics:**
  - Dimension changes from $10	imes 10 	o 2	imes 2$.
  - Candidate generator prioritized dimensional changers (`crop_nonzero`, `scale_grid`, `rot90`), but no single primitive or depth-2 pair expresses "find largest bbox color and create 2x2 solid grid".
- **Classification:** **Level 1/2 (Macro-Abstraction / Dimension Reduction)**.

---

## 3. Conclusions on Budget Scaling
1. **Saturation at Depth 2:** For depth-1 and standard depth-2 primitives, a candidate budget of 20 is already sufficient to discover and verify all expressible single-operation and dual-operation solutions.
2. **Search Depth vs. Candidate Budget:** Expanding `NUM_GUESSES` from 20 to 60 within depth-2 does not solve tasks requiring depth-3 composition. To solve `009d5c81` and `00dbd492`, depth must be increased to 3, requiring tighter semantic pruning or neural proposal guidance to avoid combinatorial explosion.
