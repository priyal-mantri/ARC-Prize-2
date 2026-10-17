"""
Canonical Domain-Specific Language (DSL) Primitives for ARC-AGI-2.

Provides 45+ deterministic, composable grid transformation operations:
- Rotations and Reflections
- Color transformations and Mappings
- Translations and Dynamic Slides
- Cropping, Padding, Scaling, and Tiling
- Topological fills, Gravity, and Symmetry
- Object extraction, Filtering, and Outlining
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.ndimage


# =====================================================================
# 1. ROTATIONS & REFLECTIONS
# =====================================================================

def rotate_90(grid: np.ndarray) -> np.ndarray:
    """Rotates grid 90 degrees clockwise."""
    if grid.size == 0: return grid.copy()
    return np.rot90(grid, k=-1).copy()


def rotate_180(grid: np.ndarray) -> np.ndarray:
    """Rotates grid 180 degrees."""
    if grid.size == 0: return grid.copy()
    return np.rot90(grid, k=2).copy()


def rotate_270(grid: np.ndarray) -> np.ndarray:
    """Rotates grid 270 degrees clockwise (90 counter-clockwise)."""
    if grid.size == 0: return grid.copy()
    return np.rot90(grid, k=1).copy()


def flip_horizontal(grid: np.ndarray) -> np.ndarray:
    """Reflects grid horizontally (flips left to right)."""
    if grid.size == 0: return grid.copy()
    return np.fliplr(grid).copy()


def flip_vertical(grid: np.ndarray) -> np.ndarray:
    """Reflects grid vertically (flips top to bottom)."""
    if grid.size == 0: return grid.copy()
    return np.flipud(grid).copy()


def mirror_horizontal(grid: np.ndarray) -> np.ndarray:
    """Mirrors the grid horizontally: [G | fliplr(G)]. Doubles width."""
    if grid.size == 0: return grid.copy()
    return np.hstack([grid, np.fliplr(grid)])


def mirror_vertical(grid: np.ndarray) -> np.ndarray:
    """Mirrors the grid vertically: [G / flipud(G)]. Doubles height."""
    if grid.size == 0: return grid.copy()
    return np.vstack([grid, np.flipud(grid)])


# =====================================================================
# 2. COLOR TRANSFORMATIONS & MAPPINGS
# =====================================================================

def change_color(grid: np.ndarray, old_color: int, new_color: int) -> np.ndarray:
    """Replaces all occurrences of old_color with new_color."""
    if grid.size == 0 or old_color == new_color: return grid.copy()
    out = grid.copy()
    out[out == old_color] = new_color
    return out


def keep_only_color(grid: np.ndarray, target_color: int) -> np.ndarray:
    """Preserves only target_color pixels, setting all other cells to 0."""
    if grid.size == 0: return grid.copy()
    out = grid.copy()
    out[out != target_color] = 0
    return out


def map_colors(grid: np.ndarray, color_map_dict: Dict[int, int]) -> np.ndarray:
    """Maps colors according to a dictionary mapping."""
    if grid.size == 0 or not color_map_dict: return grid.copy()
    out = grid.copy()
    for src, dst in color_map_dict.items():
        out[grid == src] = dst
    return out


def invert_colors(grid: np.ndarray) -> np.ndarray:
    """Inverts non-zero colors: c -> 9 - c (for c > 0)."""
    if grid.size == 0: return grid.copy()
    out = grid.copy()
    mask = (out > 0)
    out[mask] = 9 - out[mask]
    return out


def fill_with_color(grid: np.ndarray, color: int) -> np.ndarray:
    """Returns a grid of the same dimensions filled entirely with color."""
    if grid.size == 0: return grid.copy()
    return np.full_like(grid, fill_value=color)


def color_top_half(grid: np.ndarray, color: int) -> np.ndarray:
    """Colors the top half of the grid with the specified color."""
    if grid.size == 0: return grid.copy()
    out = grid.copy()
    mid = out.shape[0] // 2
    out[:mid, :] = color
    return out


def get_most_common_color(grid: np.ndarray) -> int:
    """Returns the most frequent color in the grid."""
    if grid.size == 0: return 0
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def get_least_common_color(grid: np.ndarray) -> int:
    """Returns the least frequent non-zero color in the grid, or 0 if none."""
    if grid.size == 0: return 0
    mask = (grid > 0)
    if not np.any(mask): return 0
    vals, counts = np.unique(grid[mask], return_counts=True)
    return int(vals[np.argmin(counts)])


# =====================================================================
# 3. SPATIAL TRANSLATIONS & MOVEMENT
# =====================================================================

def move_up(grid: np.ndarray, color_id: int) -> np.ndarray:
    """Moves all pixels of color_id up by 1 row, stopping at edge."""
    if grid.size == 0: return grid.copy()
    out = grid.copy()
    mask = (grid == color_id)
    out[mask] = 0
    rows, cols = np.where(mask)
    valid = (rows > 0)
    out[rows[valid] - 1, cols[valid]] = color_id
    return out


def move_down(grid: np.ndarray, color_id: int) -> np.ndarray:
    """Moves all pixels of color_id down by 1 row, stopping at edge."""
    if grid.size == 0: return grid.copy()
    out = grid.copy()
    mask = (grid == color_id)
    out[mask] = 0
    rows, cols = np.where(mask)
    valid = (rows < grid.shape[0] - 1)
    out[rows[valid] + 1, cols[valid]] = color_id
    return out


def move_left(grid: np.ndarray, color_id: int) -> np.ndarray:
    """Moves all pixels of color_id left by 1 column."""
    if grid.size == 0: return grid.copy()
    out = grid.copy()
    mask = (grid == color_id)
    out[mask] = 0
    rows, cols = np.where(mask)
    valid = (cols > 0)
    out[rows[valid], cols[valid] - 1] = color_id
    return out


def move_right(grid: np.ndarray, color_id: int) -> np.ndarray:
    """Moves all pixels of color_id right by 1 column."""
    if grid.size == 0: return grid.copy()
    out = grid.copy()
    mask = (grid == color_id)
    out[mask] = 0
    rows, cols = np.where(mask)
    valid = (cols < grid.shape[1] - 1)
    out[rows[valid], cols[valid] + 1] = color_id
    return out


def slide_down(grid: np.ndarray, color_id: int) -> np.ndarray:
    """Slides all pixels of color_id down as far as possible until hitting wall or non-zero obstacle."""
    if grid.size == 0: return grid.copy()
    out = grid.copy()
    mask = (grid == color_id)
    if not np.any(mask): return out
    out[mask] = 0

    h, w = grid.shape
    # Check max downward shift possible for all pixels simultaneously
    rows, cols = np.where(mask)
    max_shift = 0
    for shift in range(1, h):
        shifted_rows = rows + shift
        if np.any(shifted_rows >= h):
            break
        # Obstacle check
        if np.any(out[shifted_rows, cols] != 0):
            break
        max_shift = shift

    out[rows + max_shift, cols] = color_id
    return out


def slide_up(grid: np.ndarray, color_id: int) -> np.ndarray:
    """Slides all pixels of color_id up as far as possible."""
    if grid.size == 0: return grid.copy()
    out = grid.copy()
    mask = (grid == color_id)
    if not np.any(mask): return out
    out[mask] = 0

    rows, cols = np.where(mask)
    max_shift = 0
    for shift in range(1, grid.shape[0]):
        shifted_rows = rows - shift
        if np.any(shifted_rows < 0):
            break
        if np.any(out[shifted_rows, cols] != 0):
            break
        max_shift = shift

    out[rows - max_shift, cols] = color_id
    return out


def slide_left(grid: np.ndarray, color_id: int) -> np.ndarray:
    """Slides all pixels of color_id left as far as possible."""
    if grid.size == 0: return grid.copy()
    out = grid.copy()
    mask = (grid == color_id)
    if not np.any(mask): return out
    out[mask] = 0

    rows, cols = np.where(mask)
    max_shift = 0
    for shift in range(1, grid.shape[1]):
        shifted_cols = cols - shift
        if np.any(shifted_cols < 0):
            break
        if np.any(out[rows, shifted_cols] != 0):
            break
        max_shift = shift

    out[rows, cols - max_shift] = color_id
    return out


def slide_right(grid: np.ndarray, color_id: int) -> np.ndarray:
    """Slides all pixels of color_id right as far as possible."""
    if grid.size == 0: return grid.copy()
    out = grid.copy()
    mask = (grid == color_id)
    if not np.any(mask): return out
    out[mask] = 0

    rows, cols = np.where(mask)
    max_shift = 0
    for shift in range(1, grid.shape[1]):
        shifted_cols = cols + shift
        if np.any(shifted_cols >= grid.shape[1]):
            break
        if np.any(out[rows, shifted_cols] != 0):
            break
        max_shift = shift

    out[rows, cols + max_shift] = color_id
    return out


def apply_gravity(grid: np.ndarray, direction: str = "down") -> np.ndarray:
    """Applies gravity in specified direction ('down', 'up', 'left', 'right')."""
    if grid.size == 0:
        return grid.copy()
    if direction == "down":
        out = np.zeros_like(grid)
        h, w = grid.shape
        for c in range(w):
            col_vals = grid[:, c]
            non_zeros = col_vals[col_vals != 0]
            if len(non_zeros) > 0:
                out[h - len(non_zeros):, c] = non_zeros
        return out
    elif direction == "up":
        out = np.zeros_like(grid)
        for c in range(grid.shape[1]):
            col_vals = grid[:, c]
            non_zeros = col_vals[col_vals != 0]
            if len(non_zeros) > 0:
                out[:len(non_zeros), c] = non_zeros
        return out
    elif direction == "left":
        out = np.zeros_like(grid)
        for r in range(grid.shape[0]):
            row_vals = grid[r, :]
            non_zeros = row_vals[row_vals != 0]
            if len(non_zeros) > 0:
                out[r, :len(non_zeros)] = non_zeros
        return out
    elif direction == "right":
        out = np.zeros_like(grid)
        w = grid.shape[1]
        for r in range(grid.shape[0]):
            row_vals = grid[r, :]
            non_zeros = row_vals[row_vals != 0]
            if len(non_zeros) > 0:
                out[r, w - len(non_zeros):] = non_zeros
        return out
    return grid.copy()


# =====================================================================
# 4. CROPPING, PADDING, SCALING & TILING
# =====================================================================

def crop_to_content(grid: np.ndarray) -> np.ndarray:
    """Crops the grid to the minimal bounding box enclosing all non-zero pixels."""
    if grid.size == 0: return grid.copy()
    non_zeros = np.nonzero(grid)
    if len(non_zeros[0]) == 0:
        return grid.copy()
    min_r, max_r = int(np.min(non_zeros[0])), int(np.max(non_zeros[0]))
    min_c, max_c = int(np.min(non_zeros[1])), int(np.max(non_zeros[1]))
    return grid[min_r : max_r + 1, min_c : max_c + 1].copy()


def crop_background(grid: np.ndarray, bg_color: int = 0) -> np.ndarray:
    """Crops grid to bounding box enclosing all pixels not equal to bg_color."""
    if grid.size == 0: return grid.copy()
    non_bg = np.nonzero(grid != bg_color)
    if len(non_bg[0]) == 0:
        return grid.copy()
    min_r, max_r = int(np.min(non_bg[0])), int(np.max(non_bg[0]))
    min_c, max_c = int(np.min(non_bg[1])), int(np.max(non_bg[1]))
    return grid[min_r : max_r + 1, min_c : max_c + 1].copy()


def extract_color_bounding_box(grid: np.ndarray, target_color: int) -> np.ndarray:
    """Crops grid to the bounding box enclosing all pixels of target_color."""
    if grid.size == 0: return grid.copy()
    matching = np.nonzero(grid == target_color)
    if len(matching[0]) == 0:
        return grid.copy()
    min_r, max_r = int(np.min(matching[0])), int(np.max(matching[0]))
    min_c, max_c = int(np.min(matching[1])), int(np.max(matching[1]))
    return grid[min_r : max_r + 1, min_c : max_c + 1].copy()


def tile_grid(grid: np.ndarray, multiplier: int) -> np.ndarray:
    """Tiles the grid in a multiplier x multiplier pattern."""
    if grid.size == 0 or multiplier <= 1: return grid.copy()
    return np.tile(grid, (multiplier, multiplier))


def scale_grid(grid: np.ndarray, multiplier: int) -> np.ndarray:
    """Upscales each cell into a multiplier x multiplier block."""
    if grid.size == 0 or multiplier <= 1: return grid.copy()
    return np.kron(grid, np.ones((multiplier, multiplier), dtype=np.int32))


def repeat_pattern_h(grid: np.ndarray) -> np.ndarray:
    """Repeats the grid horizontally: [grid | grid]."""
    if grid.size == 0: return grid.copy()
    return np.tile(grid, (1, 2))


def repeat_pattern_v(grid: np.ndarray) -> np.ndarray:
    """Repeats the grid vertically: [grid / grid]."""
    if grid.size == 0: return grid.copy()
    return np.tile(grid, (2, 1))


def pad_grid(grid: np.ndarray, pad_size: int, pad_color: int = 0) -> np.ndarray:
    """Pads grid borders with pad_size cells of pad_color."""
    if grid.size == 0: return grid.copy()
    return np.pad(grid, pad_width=pad_size, mode='constant', constant_values=pad_color)


# =====================================================================
# 5. TOPOLOGY, FLOOD-FILL & SYMMETRY
# =====================================================================

def flood_fill(grid: np.ndarray, r: int, c: int, new_color: int) -> np.ndarray:
    """Performs connected flood fill from (r, c) with new_color."""
    if grid.size == 0: return grid.copy()
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w): return grid.copy()
    target_color = grid[r, c]
    if target_color == new_color: return grid.copy()

    out = grid.copy()
    mask = (out == target_color)
    labeled, num_features = scipy.ndimage.label(mask)
    fill_label = labeled[r, c]
    if fill_label > 0:
        out[labeled == fill_label] = new_color
    return out


def fill_enclosed(grid: np.ndarray, fill_color: int) -> np.ndarray:
    """Fills enclosed 0-holes inside foreground structures with fill_color."""
    if grid.size == 0: return grid.copy()
    out = grid.copy()
    mask = (out > 0)
    filled = scipy.ndimage.binary_fill_holes(mask)
    holes = filled & (~mask)
    out[holes] = fill_color
    return out


def hollow_out(grid: np.ndarray, color_id: int) -> np.ndarray:
    """Hollows out objects of color_id, keeping only their exterior boundary."""
    if grid.size == 0: return grid.copy()
    out = grid.copy()
    mask = (grid == color_id)
    if not np.any(mask): return out

    # Erode mask
    eroded = scipy.ndimage.binary_erosion(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    interior = mask & eroded
    out[interior] = 0
    return out


def outline_objects(grid: np.ndarray, outline_color: int = 1) -> np.ndarray:
    """Draws an outline border around all non-zero objects."""
    if grid.size == 0: return grid.copy()
    out = grid.copy()
    mask = (grid > 0)
    dilated = scipy.ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool))
    boundary = dilated & (~mask)
    out[boundary] = outline_color
    return out


def complete_symmetry(grid: np.ndarray) -> np.ndarray:
    """Completes horizontal symmetry by overlaying fliplr onto non-zero cells."""
    if grid.size == 0: return grid.copy()
    out = grid.copy()
    flipped = np.fliplr(grid)
    mask = (out == 0) & (flipped > 0)
    out[mask] = flipped[mask]
    return out


# =====================================================================
# 6. OBJECT SELECTION & FILTERING
# =====================================================================

def keep_largest_object(grid: np.ndarray) -> np.ndarray:
    """Keeps only the largest connected component of non-zero pixels."""
    if grid.size == 0: return grid.copy()
    mask = (grid > 0)
    labeled, num_features = scipy.ndimage.label(mask)
    if num_features <= 1: return grid.copy()

    sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
    largest_label = np.argmax(sizes) + 1
    out = grid.copy()
    out[labeled != largest_label] = 0
    return out


def remove_smallest_object(grid: np.ndarray) -> np.ndarray:
    """Removes the smallest connected component of non-zero pixels."""
    if grid.size == 0: return grid.copy()
    mask = (grid > 0)
    labeled, num_features = scipy.ndimage.label(mask)
    if num_features <= 1: return grid.copy()

    sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
    smallest_label = np.argmin(sizes) + 1
    out = grid.copy()
    out[labeled == smallest_label] = 0
    return out


def remove_noise(grid: np.ndarray) -> np.ndarray:
    """Removes isolated single-pixel objects (components of size 1)."""
    if grid.size == 0: return grid.copy()
    mask = (grid > 0)
    labeled, num_features = scipy.ndimage.label(mask)
    if num_features == 0: return grid.copy()

    out = grid.copy()
    for i in range(1, num_features + 1):
        if np.sum(labeled == i) == 1:
            out[labeled == i] = 0
    return out


def sort_objects_by_size(grid: np.ndarray) -> np.ndarray:
    """Extracts non-zero objects and arranges them side-by-side sorted by size descending."""
    if grid.size == 0: return grid.copy()
    mask = (grid > 0)
    labeled, num_features = scipy.ndimage.label(mask)
    if num_features <= 1: return grid.copy()

    objs = []
    for i in range(1, num_features + 1):
        obj_m = (labeled == i)
        pixels = np.nonzero(obj_m)
        min_r, max_r = np.min(pixels[0]), np.max(pixels[0])
        min_c, max_c = np.min(pixels[1]), np.max(pixels[1])
        sub = grid[min_r:max_r+1, min_c:max_c+1]
        objs.append((np.sum(obj_m), sub))

    objs.sort(key=lambda x: x[0], reverse=True)
    # Stack side-by-side with equal heights padded by 0
    max_h = max(o[1].shape[0] for o in objs)
    padded = [np.pad(o[1], ((0, max_h - o[1].shape[0]), (0, 0))) for o in objs]
    return np.hstack(padded)


def overlay(grid_bg: np.ndarray, grid_fg: np.ndarray) -> np.ndarray:
    """Overlays non-zero pixels of grid_fg onto grid_bg (assuming matching shapes)."""
    if grid_bg.shape != grid_fg.shape:
        return grid_bg.copy()
    out = grid_bg.copy()
    mask = (grid_fg > 0)
    out[mask] = grid_fg[mask]
    return out


def draw_line_hv(grid: np.ndarray, r1: int, c1: int, r2: int, c2: int, color_id: int) -> np.ndarray:
    """Draws a horizontal or vertical line between two points."""
    if grid.size == 0: return grid.copy()
    out = grid.copy()
    h, w = out.shape
    r_start, r_end = min(r1, r2), max(r1, r2)
    c_start, c_end = min(c1, c2), max(c1, c2)
    if r1 == r2:
        out[r1, max(0, c_start):min(w, c_end + 1)] = color_id
    elif c1 == c2:
        out[max(0, r_start):min(h, r_end + 1), c1] = color_id
    return out


# Map of primitive name -> function
DSL_REGISTRY: Dict[str, Callable[..., np.ndarray]] = {
    # Rotations & Reflections
    "rotate_90": rotate_90,
    "rotate_180": rotate_180,
    "rotate_270": rotate_270,
    "flip_horizontal": flip_horizontal,
    "flip_vertical": flip_vertical,
    "mirror_horizontal": mirror_horizontal,
    "mirror_vertical": mirror_vertical,
    # Colors
    "change_color": change_color,
    "keep_only_color": keep_only_color,
    "map_colors": map_colors,
    "invert_colors": invert_colors,
    "fill_with_color": fill_with_color,
    "color_top_half": color_top_half,
    # Movement
    "move_up": move_up,
    "move_down": move_down,
    "move_left": move_left,
    "move_right": move_right,
    "slide_up": slide_up,
    "slide_down": slide_down,
    "slide_left": slide_left,
    "slide_right": slide_right,
    "apply_gravity": apply_gravity,
    # Cropping, Scaling & Tiling
    "crop_to_content": crop_to_content,
    "crop_background": crop_background,
    "extract_color_bounding_box": extract_color_bounding_box,
    "tile_grid": tile_grid,
    "scale_grid": scale_grid,
    "repeat_pattern_h": repeat_pattern_h,
    "repeat_pattern_v": repeat_pattern_v,
    "pad_grid": pad_grid,
    # Topology & Symmetry
    "flood_fill": flood_fill,
    "fill_enclosed": fill_enclosed,
    "hollow_out": hollow_out,
    "outline_objects": outline_objects,
    "complete_symmetry": complete_symmetry,
    # Filtering & Selection
    "keep_largest_object": keep_largest_object,
    "remove_smallest_object": remove_smallest_object,
    "remove_noise": remove_noise,
    "sort_objects_by_size": sort_objects_by_size,
    "overlay": overlay,
    "draw_line_hv": draw_line_hv,
}

# Aliases for common canonical operations
DSL_REGISTRY.update({
    "identity": lambda grid: np.array(grid, copy=True),
    "transpose": lambda grid: np.array(grid, copy=True).T,
    "rot90": rotate_90,
    "rot180": rotate_180,
    "rot270": rotate_270,
    "reflect_main_diagonal": lambda grid: np.array(grid, copy=True).T,
    "reflect_anti_diagonal": lambda grid: np.flipud(np.fliplr(np.array(grid, copy=True).T)),
    "replace_color": lambda grid, old_color=0, new_color=0: change_color(grid, old_color, new_color),
    "fill_enclosed_holes": fill_enclosed,
    "gravity_down": lambda grid: apply_gravity(grid, direction="down"),
    "gravity_up": lambda grid: apply_gravity(grid, direction="up"),
    "gravity_left": lambda grid: apply_gravity(grid, direction="left"),
    "gravity_right": lambda grid: apply_gravity(grid, direction="right"),
    "complete_symmetry_horizontal": lambda grid: complete_symmetry(grid, axis="horizontal"),
    "complete_symmetry_vertical": lambda grid: complete_symmetry(grid, axis="vertical"),
    "crop_to_largest_object": lambda grid: crop_to_content(keep_largest_object(grid)),
    "keep_smallest_object": remove_smallest_object,
    "scale_up": lambda grid, factor=2: scale_grid(grid, factor_h=factor, factor_w=factor),
    "pad": lambda grid, pad_width=1, fill_color=0: pad_grid(grid, pad_h=pad_width, pad_w=pad_width, fill_color=fill_color),
    "crop": lambda grid, r0=0, c0=0, r1=None, c1=None: np.array(grid, copy=True)[r0:r1, c0:c1],
})

# Canonical exported alias
PRIMITIVE_REGISTRY = DSL_REGISTRY


def get_primitives_by_category() -> Dict[str, List[str]]:
    """Group registered primitives into semantic categories."""
    return {
        "geometry": [
            "identity", "transpose", "rotate_90", "rotate_180", "rotate_270",
            "rot90", "rot180", "rot270", "flip_horizontal", "flip_vertical",
            "mirror_horizontal", "mirror_vertical", "reflect_main_diagonal", "reflect_anti_diagonal"
        ],
        "color": [
            "change_color", "replace_color", "keep_only_color", "map_colors",
            "invert_colors", "fill_with_color", "color_top_half"
        ],
        "movement": [
            "move_up", "move_down", "move_left", "move_right",
            "slide_up", "slide_down", "slide_left", "slide_right",
            "apply_gravity", "gravity_down", "gravity_up", "gravity_left", "gravity_right"
        ],
        "cropping_scaling": [
            "crop_to_content", "crop_background", "extract_color_bounding_box",
            "crop_to_largest_object", "tile_grid", "scale_grid", "scale_up",
            "repeat_pattern_h", "repeat_pattern_v", "pad_grid", "pad", "crop"
        ],
        "topology_symmetry": [
            "flood_fill", "fill_enclosed", "fill_enclosed_holes", "hollow_out",
            "outline_objects", "complete_symmetry", "complete_symmetry_horizontal",
            "complete_symmetry_vertical"
        ],
        "objects": [
            "keep_largest_object", "remove_smallest_object", "keep_smallest_object",
            "remove_noise", "sort_objects_by_size", "overlay", "draw_line_hv"
        ],
    }
