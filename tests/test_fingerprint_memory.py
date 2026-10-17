"""Unit tests for Structural Fingerprint, MirrorMatrix Memory, and Concept Library."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.representation.grid import ArcGrid
from src.fingerprint.encoder import FINGERPRINT_DIM_V1, grid_to_barcode, task_to_fingerprint
from src.memory.mirror_matrix import MemoryEntry, MirrorMatrix
from src.dsl.program import DSLProgram, PrimitiveCall
from src.solvers.concept_library import ConceptLibrary


def test_fingerprint_extraction():
    arr = np.zeros((10, 15), dtype=np.int32)
    arr[2:5, 3:6] = 4
    grid = ArcGrid(arr)

    barcode = grid_to_barcode(grid)
    assert len(barcode) == FINGERPRINT_DIM_V1
    assert all(isinstance(v, (float, np.floating)) for v in barcode)
    # barcode[0] is number of foreground objects
    assert barcode[0] == 1.0
    # barcode[15], barcode[16] are height and width
    assert barcode[15] == 10.0
    assert barcode[16] == 15.0


def test_mirror_matrix_memory_retrieval():
    matrix = MirrorMatrix(vector_dim=FINGERPRINT_DIM_V1)

    fp1 = [0.1] * FINGERPRINT_DIM_V1
    prog1 = DSLProgram("rotate_90", [PrimitiveCall("rot90")])
    matrix.add_entry(MemoryEntry(task_id="task_1", fingerprint=fp1, program=prog1))

    fp2 = [0.9] * FINGERPRINT_DIM_V1
    prog2 = DSLProgram("flip_h", [PrimitiveCall("flip_horizontal")])
    matrix.add_entry(MemoryEntry(task_id="task_2", fingerprint=fp2, program=prog2))

    assert len(matrix) == 2

    # Query close to fp1
    query_fp = [0.12] * FINGERPRINT_DIM_V1
    results = matrix.query_similar(query_fp, top_k=1)
    assert len(results) == 1
    best_entry, dist = results[0]
    assert best_entry.task_id == "task_1"
    assert best_entry.program is not None
    assert best_entry.program.name == "rotate_90"

    # Persistence save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_file = Path(tmpdir) / "test_memory.json"
        matrix.save(save_file)
        assert save_file.exists()

        loaded_matrix = MirrorMatrix(storage_path=save_file)
        assert len(loaded_matrix) == 2
        assert loaded_matrix.entries[0].task_id == "task_1"


def test_concept_library_registration_and_retrieval():
    library = ConceptLibrary()
    prog = DSLProgram("crop_largest", [PrimitiveCall("crop_to_content")])

    skill = library.register_skill(
        program=prog,
        task_id="task_alpha",
        preconditions={"requires_same_shape": False},
        expected_effects=["crop_bounding_box"],
    )
    assert skill.skill_id.startswith("skill_")
    assert "task_alpha" in skill.solved_task_ids

    # Query skills matching conditions
    candidates = library.get_candidate_skills_for_task({"same_shape": False})
    assert len(candidates) == 1
    assert candidates[0].program.name == "crop_largest"
