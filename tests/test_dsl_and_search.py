"""Unit tests for DSL Primitives, Programs, Constraint Filtering, Verification, and Beam Search."""

import numpy as np
import pytest

from src.representation.grid import ArcGrid
from src.dsl.primitives import PRIMITIVE_REGISTRY
from src.dsl.program import DSLProgram, PrimitiveCall, LambdaProgram
from src.search.constraint_filter import ConstraintFilter
from src.search.verifier import DemonstrationVerifier, TaskDemonstration
from src.search.repl_refiner import REPLRefiner
from src.search.beam_search import DSLBeamSearch
from src.data.dataset_loader import ArcTask, GridPair


def test_primitive_operations():
    # 1. Rotation and reflection
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    r90 = PRIMITIVE_REGISTRY["rot90"](arr)
    assert np.array_equal(r90, np.array([[3, 1], [4, 2]], dtype=np.int32))

    fliph = PRIMITIVE_REGISTRY["flip_horizontal"](arr)
    assert np.array_equal(fliph, np.array([[2, 1], [4, 3]], dtype=np.int32))

    # 2. Color replacement
    replaced = PRIMITIVE_REGISTRY["replace_color"](arr, old_color=1, new_color=5)
    assert replaced[0, 0] == 5
    assert replaced[0, 1] == 2

    # 3. Gravity
    grav_arr = np.zeros((4, 2), dtype=np.int32)
    grav_arr[0, 0] = 3
    grav_down = PRIMITIVE_REGISTRY["gravity_down"](grav_arr)
    assert grav_down[3, 0] == 3
    assert grav_down[0, 0] == 0


def test_dsl_program_composition_and_execution():
    call1 = PrimitiveCall("rot90")
    call2 = PrimitiveCall("replace_color", {"old_color": 1, "new_color": 8})
    prog = DSLProgram("rot_and_recolor", [call1, call2])

    grid = ArcGrid(np.array([[1, 2], [3, 4]], dtype=np.int32))
    result = prog.execute(grid)
    assert isinstance(result, ArcGrid)
    res_arr = result.to_numpy()
    # rot90 -> [[3, 1], [4, 2]], replace 1->8 -> [[3, 8], [4, 2]]
    assert res_arr[0, 1] == 8
    assert res_arr[0, 0] == 3

    # Serialization check
    p_dict = prog.to_dict()
    restored = DSLProgram.from_dict(p_dict)
    assert restored.name == prog.name
    assert restored.complexity == prog.complexity


def test_demonstration_verifier():
    verifier = DemonstrationVerifier()
    prog = DSLProgram("rot90", [PrimitiveCall("rot90")])

    in_arr = np.array([[1, 0], [0, 0]], dtype=np.int32)
    # rot90 -> [[0, 1], [0, 0]]
    out_arr = np.array([[0, 1], [0, 0]], dtype=np.int32)

    demo = TaskDemonstration(ArcGrid(in_arr), ArcGrid(out_arr))
    vres = verifier.verify(prog, [demo])
    assert vres.is_fully_solved
    assert vres.mean_pixel_accuracy == 1.0


def test_constraint_filtering():
    # Demonstrations where dimensions are preserved
    demos = [
        TaskDemonstration(
            ArcGrid(np.zeros((4, 4), dtype=np.int32)),
            ArcGrid(np.zeros((4, 4), dtype=np.int32)),
        )
    ]
    cf = ConstraintFilter.from_demonstrations(demos)
    assert cf.is_primitive_admissible("rot90")
    # Shape changing primitives must be pruned
    assert not cf.is_primitive_admissible("crop_to_content")
    assert not cf.is_primitive_admissible("tile_grid")


def test_dsl_beam_search_synthetic():
    verifier = DemonstrationVerifier()
    searcher = DSLBeamSearch(verifier=verifier, timeout_seconds=1.0)

    # Synthetic task: rot90
    t_in = np.array([[1, 2], [3, 4]], dtype=np.int32)
    t_out = np.array([[3, 1], [4, 2]], dtype=np.int32)
    task = ArcTask(
        task_id="synthetic_rot90",
        train_pairs=(GridPair(input=t_in, output=t_out),),
        test_inputs=(t_in,),
        dataset_split="training",
    )

    result = searcher.search(task)
    assert result.solved
    assert result.best_program is not None
    assert result.best_program.name == "rotate_90" or result.best_program.name == "rot90"
