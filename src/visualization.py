"""
Visualization tools for ARC-AGI-2 grids and tasks using standard ARC color palettes.

Adheres strictly to the canonical 10-color ARC palette introduced by François Chollet:
0: Black (#000000)
1: Blue (#0074D9)
2: Red (#FF4136)
3: Green (#2ECC40)
4: Yellow (#FFDC00)
5: Grey (#AAAAAA)
6: Magenta (#F012BE)
7: Orange (#FF851B)
8: Teal (#7FDBFF)
9: Maroon (#870C25)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Canonical ARC 10-color hex codes and names
ARC_HEX_COLORS = [
    "#000000",  # 0: Black
    "#0074D9",  # 1: Blue
    "#FF4136",  # 2: Red
    "#2ECC40",  # 3: Green
    "#FFDC00",  # 4: Yellow
    "#AAAAAA",  # 5: Grey
    "#F012BE",  # 6: Magenta
    "#FF851B",  # 7: Orange
    "#7FDBFF",  # 8: Teal
    "#870C25",  # 9: Maroon
]

ARC_COLOR_NAMES = [
    "Black", "Blue", "Red", "Green", "Yellow",
    "Grey", "Magenta", "Orange", "Teal", "Maroon"
]

# Matplotlib Colormap and Normalization for integer values 0..9
ARC_CMAP = mcolors.ListedColormap(ARC_HEX_COLORS)
ARC_NORM = mcolors.Normalize(vmin=0, vmax=9)
ARC_COLORS = ARC_HEX_COLORS


def plot_grid(
    grid: Union[np.ndarray, List[List[int]]],
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show_grid_lines: bool = True,
    show_dimensions: bool = True,
) -> plt.Axes:
    """
    Renders a single ARC grid with crisp cell borders and canonical colors.

    Args:
        grid: 2D array or list of lists containing integer colors 0-9.
        ax: Optional matplotlib Axes. If None, creates a new figure.
        title: Optional title string.
        show_grid_lines: If True, draws thin borders between cells.
        show_dimensions: If True, appends (H × W) to title.

    Returns:
        The matplotlib Axes object.
    """
    grid_arr = np.array(grid, dtype=np.int32) if not isinstance(grid, np.ndarray) else grid
    h, w = grid_arr.shape

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(w * 0.35, 2.0), max(h * 0.35, 2.0)))
        created_fig = True

    ax.imshow(grid_arr, cmap=ARC_CMAP, norm=ARC_NORM, interpolation="nearest")

    if show_grid_lines:
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which="minor", color="#444444", linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", size=0)

    ax.set_xticks([])
    ax.set_yticks([])

    full_title = title or ""
    if show_dimensions:
        dim_str = f"({h}×{w})"
        full_title = f"{full_title} {dim_str}".strip()

    if full_title:
        ax.set_title(full_title, fontsize=10, pad=6)

    return ax


def plot_task(
    task: Any,
    ground_truth: Optional[List[Union[np.ndarray, List[List[int]]]]] = None,
    show_solutions: bool = False,
    max_train_examples: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualizes an entire ARC task, displaying all training demonstration pairs and test inputs.

    SAFETY WARNING:
    Test solutions are ONLY rendered if `show_solutions=True` AND `ground_truth` is explicitly supplied.
    A prominent banner will indicate when evaluation ground-truth is on display.

    Args:
        task: ArcTask instance or dict with 'train' and 'test'.
        ground_truth: Optional list of solution grids for test inputs.
        show_solutions: If True, displays test solutions alongside test inputs.
        max_train_examples: Optional cap on displayed training examples.
        figsize: Optional (width, height) figure size override.
        save_path: Optional file path to save figure.

    Returns:
        The matplotlib Figure object.
    """
    # Extract training pairs and test inputs
    if hasattr(task, "train_pairs"):
        task_id = task.task_id
        split = task.dataset_split
        train_pairs = [(p.input, p.output) for p in task.train_pairs]
        test_inputs = list(task.test_inputs)
    else:
        task_id = task.get("id", "Unknown")
        split = "Custom"
        train_pairs = [(np.array(p["input"]), np.array(p["output"])) for p in task.get("train", [])]
        test_inputs = [np.array(t["input"]) for t in task.get("test", [])]

    if max_train_examples:
        train_pairs = train_pairs[:max_train_examples]

    num_train = len(train_pairs)
    num_test = len(test_inputs)

    # Determine layout: each row represents a train pair or test item
    total_rows = num_train + num_test
    cols_per_row = 2  # Input on left, Output/Prediction on right

    if figsize is None:
        fig_w = 7.0
        fig_h = max(total_rows * 2.5, 3.5)
        figsize = (fig_w, fig_h)

    fig, axes = plt.subplots(total_rows, cols_per_row, figsize=figsize, squeeze=False)

    fig.suptitle(
        f"ARC Task: {task_id} (Split: {split}) | {num_train} Train Pairs, {num_test} Test Input(s)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )

    # Plot training pairs
    for i, (inp, out) in enumerate(train_pairs):
        plot_grid(inp, ax=axes[i, 0], title=f"Train {i+1} Input")
        plot_grid(out, ax=axes[i, 1], title=f"Train {i+1} Output")

    # Plot test inputs
    for j, test_inp in enumerate(test_inputs):
        row_idx = num_train + j
        plot_grid(test_inp, ax=axes[row_idx, 0], title=f"Test {j+1} Input")

        if show_solutions and ground_truth and j < len(ground_truth):
            sol = ground_truth[j]
            plot_grid(sol, ax=axes[row_idx, 1], title=f"Test {j+1} Solution [GROUND TRUTH]")
            # Highlight border to emphasize evaluation ground truth
            for spine in axes[row_idx, 1].spines.values():
                spine.set_edgecolor("#FF4136")
                spine.set_linewidth(2.0)
        else:
            axes[row_idx, 1].text(
                0.5, 0.5,
                "Prediction Required\n(Unseen Test Output)",
                ha="center", va="center", fontsize=9, color="#666666", style="italic"
            )
            axes[row_idx, 1].set_xticks([])
            axes[row_idx, 1].set_yticks([])
            axes[row_idx, 1].set_title(f"Test {j+1} Target")
            for spine in axes[row_idx, 1].spines.values():
                spine.set_linestyle(":")
                spine.set_color("#888888")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_color_palette(ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Renders the reference 10-color ARC palette with color numbers and names."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 1.5))

    palette_arr = np.arange(10).reshape(1, 10)
    ax.imshow(palette_arr, cmap=ARC_CMAP, norm=ARC_NORM, aspect="auto")

    for c in range(10):
        text_color = "white" if c in [0, 1, 2, 9] else "black"
        ax.text(c, 0, f"{c}\n{ARC_COLOR_NAMES[c]}", ha="center", va="center", color=text_color, fontsize=8, fontweight="bold")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Canonical ARC-AGI 10-Color Palette", fontsize=10, pad=6)
    return ax
