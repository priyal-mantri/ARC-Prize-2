"""LLM (Qwen2.5-Coder) and Symbolic Hybrid Synthesizer Interface.

Coordinates neural program hypothesis generation with deterministic symbolic fallback:
- Discovers and loads local/Kaggle-hosted Qwen2.5-Coder weights offline (no network downloads).
- Formulates structured ARC problem prompts with scene graph facts and relational changes.
- Validates model JSON output against canonical DSL primitive registry and parameter schemas.
- Records full execution telemetry: model ID, source, device, latency, tokens, and fallback usage.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.data.dataset_loader import ArcTask
from src.representation.grid import ArcGrid
from src.dsl.primitives import PRIMITIVE_REGISTRY, PRIMITIVE_SPECS
from src.dsl.program import DSLProgram, PrimitiveCall
from src.correspondence.relational_change import RelationalChangeReport

logger = logging.getLogger(__name__)


# Known candidate Kaggle / local model weight locations
CANDIDATE_MODEL_PATHS = [
    Path(os.environ.get("QWEN_MODEL_PATH", "/nonexistent")),
    Path("/kaggle/input/models/qwen-lm/qwen2.5-coder/transformers/7b-instruct/1"),
    Path("/kaggle/input/models/qwen-lm/qwen2.5-coder/Transformers/7b-instruct/1"),
    Path("/kaggle/input/models/qwen-lm/qwen2.5-coder/transformers/1.5b-instruct/1"),
    Path("/kaggle/input/models/qwen-lm/qwen2.5-coder/Transformers/1.5b-instruct/1"),
    Path("/kaggle/input/qwen2.5-coder/transformers/7b-instruct/1"),
    Path("/kaggle/input/qwen2.5-coder/Transformers/7b-instruct/1"),
    Path("/kaggle/input/qwen2.5-coder/transformers/1.5b-instruct/1"),
    Path("/kaggle/input/qwen2.5-coder/Transformers/1.5b-instruct/1"),
    Path("/kaggle/input/qwen2.5-coder/transformers/3b-instruct/1"),
    Path("/kaggle/input/qwen2.5-coder/Transformers/3b-instruct/1"),
    Path("models/qwen2.5-coder"),
]


@dataclass
class LLMExecutionTelemetry:
    """Detailed record of neural model initialization and inference metrics."""
    runtime_environment: Dict[str, Any] = field(default_factory=dict)
    model_identifier: str = "none"
    model_artifact_source: str = "none"
    model_artifact_path: str = "none"
    quantization_method: str = "none"
    runtime_versions: Dict[str, str] = field(default_factory=dict)
    gpu_device: Optional[str] = None
    model_loading_success: bool = False
    model_load_success: bool = False
    model_load_error: Optional[str] = None
    model_load_seconds: float = 0.0
    inference_success: bool = False
    calls_count: int = 0
    tokens_generated: int = 0
    generated_token_count: int = 0
    inference_latency_sec: float = 0.0
    inference_latency_seconds: float = 0.0
    fallback_usage: bool = False
    symbolic_fallback_used: bool = False
    fallback_reason: Optional[str] = None
    failures: List[str] = field(default_factory=list)
    proposals_count: int = 0
    valid_dsl_programs_count: int = 0
    inference_attempt_count: int = 0
    successful_inference_count: int = 0
    genuine_qwen_proposal_count: int = 0
    valid_qwen_proposal_count: int = 0
    qwen_proposal_verification_count: int = 0
    unsupported_capability_requests: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runtime_environment": self.runtime_environment,
            "model_identifier": self.model_identifier,
            "model_artifact_source": self.model_artifact_source,
            "model_artifact_path": self.model_artifact_source,
            "quantization_method": self.quantization_method,
            "runtime_versions": self.runtime_versions,
            "gpu_device": self.gpu_device,
            "model_loading_success": self.model_loading_success,
            "model_load_success": self.model_loading_success,
            "model_load_error": self.fallback_reason,
            "model_load_seconds": round(self.model_load_seconds, 3),
            "inference_success": self.inference_success,
            "calls_count": self.calls_count,
            "tokens_generated": self.tokens_generated,
            "generated_token_count": self.tokens_generated,
            "inference_latency_sec": round(self.inference_latency_sec, 4),
            "inference_latency_seconds": round(self.inference_latency_sec, 4),
            "fallback_usage": self.fallback_usage,
            "symbolic_fallback_used": self.fallback_usage,
            "fallback_reason": self.fallback_reason,
            "failures": self.failures,
            "proposals_count": self.proposals_count,
            "valid_dsl_programs_count": self.valid_dsl_programs_count,
            "inference_attempt_count": self.inference_attempt_count,
            "successful_inference_count": self.successful_inference_count,
            "genuine_qwen_proposal_count": self.genuine_qwen_proposal_count,
            "valid_qwen_proposal_count": self.valid_qwen_proposal_count,
            "qwen_proposal_verification_count": self.qwen_proposal_verification_count,
            "unsupported_capability_requests": self.unsupported_capability_requests,
        }


class LLMSynthesizerInterface:
    """Neuro-symbolic hypothesis generator with Qwen2.5-Coder and rule-based fallback."""

    # Class-level model/tokenizer cache to prevent redundant reloading across tasks
    _cached_model: Any = None
    _cached_tokenizer: Any = None
    _cached_path: Optional[Path] = None
    _cached_gpu_device: Optional[str] = None
    _cached_load_duration: float = 0.0

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        use_symbolic_fallback: bool = True,
        temperature: float = 0.2,
        max_new_tokens: int = 256,
    ):
        self.requested_model_name = model_name or "Qwen/Qwen2.5-Coder-7B-Instruct"
        self.explicit_path = Path(model_path) if model_path else None
        self.use_symbolic_fallback = use_symbolic_fallback
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.telemetry = LLMExecutionTelemetry()
        self.model = None
        self.tokenizer = None
        self._init_runtime()

    def _init_runtime(self) -> None:
        """Attempt to discover and load local/Kaggle model weights."""
        import platform
        import sys
        self.telemetry.runtime_environment = {
            "os": platform.system(),
            "os_release": platform.release(),
            "python": sys.version.split()[0],
        }

        try:
            import torch
            self.telemetry.runtime_versions["torch"] = str(torch.__version__)
            self.telemetry.runtime_environment["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                self.telemetry.runtime_environment["gpu_name"] = torch.cuda.get_device_name(0)
                self.telemetry.runtime_environment["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        except ImportError:
            self.telemetry.runtime_versions["torch"] = "unavailable"
            self.telemetry.runtime_environment["cuda_available"] = False

        try:
            import transformers
            self.telemetry.runtime_versions["transformers"] = str(transformers.__version__)
        except ImportError:
            self.telemetry.runtime_versions["transformers"] = "unavailable"

        # Check if already cached
        if LLMSynthesizerInterface._cached_model is not None and LLMSynthesizerInterface._cached_tokenizer is not None:
            self.model = LLMSynthesizerInterface._cached_model
            self.tokenizer = LLMSynthesizerInterface._cached_tokenizer
            self.telemetry.model_loading_success = True
            self.telemetry.model_load_success = True
            self.telemetry.model_artifact_source = str(LLMSynthesizerInterface._cached_path)
            self.telemetry.model_artifact_path = str(LLMSynthesizerInterface._cached_path)
            self.telemetry.model_identifier = LLMSynthesizerInterface._cached_path.name if LLMSynthesizerInterface._cached_path else "cached"
            self.telemetry.gpu_device = LLMSynthesizerInterface._cached_gpu_device
            self.telemetry.model_load_seconds = LLMSynthesizerInterface._cached_load_duration
            self.telemetry.quantization_method = "float16/auto"
            logger.info("Reusing cached Qwen2.5-Coder model instance from %s", self.telemetry.model_artifact_source)
            return

        resolved_path: Optional[Path] = None
        if self.explicit_path and self.explicit_path.exists():
            resolved_path = self.explicit_path
        else:
            for cand in CANDIDATE_MODEL_PATHS:
                if cand.exists() and (cand / "config.json").exists():
                    resolved_path = cand
                    break

        if resolved_path is None:
            self.telemetry.model_loading_success = False
            self.telemetry.model_load_success = False
            self.telemetry.fallback_usage = True
            self.telemetry.symbolic_fallback_used = True
            self.telemetry.fallback_reason = "No local Qwen model artifact found at candidate paths"
            self.telemetry.model_load_error = self.telemetry.fallback_reason
            logger.info("Qwen2.5-Coder weights not found locally. Running in symbolic fallback mode.")
            return

        # Check GPU availability
        import torch
        has_cuda = torch.cuda.is_available()
        self.telemetry.model_artifact_source = str(resolved_path)
        self.telemetry.model_artifact_path = str(resolved_path)
        self.telemetry.model_identifier = resolved_path.name

        if not has_cuda:
            self.telemetry.model_loading_success = False
            self.telemetry.model_load_success = False
            self.telemetry.fallback_usage = True
            self.telemetry.symbolic_fallback_used = True
            self.telemetry.fallback_reason = "CUDA GPU device unavailable for offline neural inference"
            self.telemetry.model_load_error = self.telemetry.fallback_reason
            logger.info("CUDA GPU unavailable. Running in symbolic fallback mode.")
            return

        self.telemetry.gpu_device = torch.cuda.get_device_name(0)

        # Attempt offline load
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info("Loading Qwen2.5-Coder from %s...", resolved_path)
            start_load = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(
                resolved_path,
                local_files_only=True,
                trust_remote_code=False,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved_path,
                torch_dtype="auto",
                device_map="auto",
                local_files_only=True,
                trust_remote_code=False,
            )
            load_duration = time.time() - start_load
            self.telemetry.model_loading_success = True
            self.telemetry.model_load_success = True
            self.telemetry.model_load_seconds = load_duration
            self.telemetry.quantization_method = "auto"
            # Cache for future instances
            LLMSynthesizerInterface._cached_model = self.model
            LLMSynthesizerInterface._cached_tokenizer = self.tokenizer
            LLMSynthesizerInterface._cached_path = resolved_path
            LLMSynthesizerInterface._cached_gpu_device = self.telemetry.gpu_device
            LLMSynthesizerInterface._cached_load_duration = load_duration
            logger.info("Successfully loaded Qwen2.5-Coder in %.2fs on %s", load_duration, self.telemetry.gpu_device)
        except Exception as e:
            self.telemetry.model_loading_success = False
            self.telemetry.model_load_success = False
            self.telemetry.fallback_usage = True
            self.telemetry.symbolic_fallback_used = True
            self.telemetry.fallback_reason = f"Model load failed: {str(e)}"
            self.telemetry.model_load_error = str(e)
            self.telemetry.failures.append(str(e))
            logger.warning("Failed to load Qwen2.5-Coder: %s. Using symbolic fallback.", e)

    @property
    def is_model_available(self) -> bool:
        return self.telemetry.model_loading_success and self.model is not None and self.tokenizer is not None

    def generate_hypotheses(
        self,
        task: ArcTask,
        change_report: Optional[RelationalChangeReport] = None,
        max_candidates: int = 5,
        scene_graph_facts: Optional[Dict[str, Any]] = None,
        available_skills: Optional[List[Dict[str, Any]]] = None,
    ) -> List[DSLProgram]:
        """Generate structured candidate DSL programs using Qwen when available, or symbolic fallback."""
        candidates: List[DSLProgram] = []

        if self.is_model_available:
            try:
                candidates = self._generate_with_qwen(task, change_report, max_candidates, scene_graph_facts, available_skills)
                if candidates:
                    self.telemetry.inference_success = True
                    return candidates
            except Exception as e:
                self.telemetry.failures.append(f"Inference error on {task.task_id}: {str(e)}")
                logger.warning("Neural generation error on %s: %s", task.task_id, e)

        # Fallback to symbolic relational generator
        self.telemetry.fallback_usage = True
        return self._symbolic_relational_synthesis(task, change_report, max_candidates, scene_graph_facts)

    def _generate_with_qwen(
        self,
        task: ArcTask,
        change_report: Optional[RelationalChangeReport],
        max_candidates: int,
        scene_graph_facts: Optional[Dict[str, Any]],
        available_skills: Optional[List[Dict[str, Any]]] = None,
    ) -> List[DSLProgram]:
        """Execute local model inference with validated JSON output schema."""
        import torch
        prompt = self._build_prompt(task, change_report, scene_graph_facts, available_skills)
        start_t = time.time()
        self.telemetry.calls_count += 1
        self.telemetry.inference_attempt_count += 1

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][input_len:]
        self.telemetry.tokens_generated += len(gen_tokens)
        self.telemetry.inference_latency_sec += (time.time() - start_t)
        self.telemetry.successful_inference_count += 1
        response_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        return self._parse_and_validate_response(response_text, max_candidates)

    def _build_prompt(
        self,
        task: ArcTask,
        change_report: Optional[RelationalChangeReport],
        scene_graph_facts: Optional[Dict[str, Any]],
        available_skills: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Constructs grounded neuro-symbolic reasoning prompt containing full 2D grid integer arrays."""
        train_summaries = []
        for i, pair in enumerate(task.train_examples):
            inp = pair.input_grid
            out = pair.output_grid
            inp_rows = "\n".join(" ".join(str(int(c)) for c in row) for row in inp)
            out_rows = "\n".join(" ".join(str(int(c)) for c in row) for row in out)
            train_summaries.append(
                f"--- Demonstration {i} ---\n"
                f"Input ({inp.shape[0]}x{inp.shape[1]}):\n{inp_rows}\n"
                f"Output ({out.shape[0]}x{out.shape[1]}):\n{out_rows}"
            )

        sg_str = f"Scene graph facts: {json.dumps(scene_graph_facts)}" if scene_graph_facts else "Scene graph: single grid"
        rel_str = "Relational changes: None detected"
        if change_report:
            rel_str = (
                f"Consistent recolor: {change_report.consistent_recolor_rules}, "
                f"Motion vectors: {change_report.motion_vectors}, "
                f"Creates/destroys: {change_report.always_creates_or_destroys}"
            )

        skills_str = ""
        if available_skills:
            skills_str = f"Known verified skills available for reuse or composition:\n{json.dumps(available_skills, indent=2)}\n"

        primitives_help = (
            "Available core DSL primitives:\n"
            "- fractal_expansion(): Kronecker expansion where pattern layout dictates placement of pattern copies\n"
            "- tile_alternating_h(rep_h=3, rep_w=3): 3x3 repetition with horizontal flips on alternating rows\n"
            "- fill_enclosed(fill_color): Fills interior enclosed 0-holes with fill_color\n"
            "- fill_enclosed_by_size(): Fills enclosed 0-holes conditioned on cavity size (3x3->8, 5x5->4, 7x7->3)\n"
            "- recolor_by_probe_key(probe_color=1, main_color=8): Erases probe and recolors main component by probe shape\n"
            "- replace_color(old_color, new_color): Replaces old_color cells with new_color\n"
            "- crop_to_content(): Minimal bounding box crop of non-zero pixels\n"
            "- complete_symmetry(axis='horizontal'|'vertical'): Completes reflection symmetry\n"
            "- gravity_down(): Pulls non-zero pixels to bottom floor\n"
            "- keep_largest_object(): Preserves only largest connected component"
        )

        return f"""<|im_start|>system
You are an expert neuro-symbolic ARC-AGI-2 reasoning engine.
Analyze the demonstrations, full 2D grids, scene graph facts, and relational changes.
{primitives_help}
{skills_str}Return ONLY valid JSON matching this schema:
{{"programs": [{{"name": "candidate_name", "calls": [{{"primitive": "primitive_name", "params": {{}}}}]}}]}}
<|im_end|>
<|im_start|>user
Task ID: {task.task_id}
Demonstrations:
{chr(10).join(train_summaries)}

{sg_str}
{rel_str}

Propose candidate DSL program sequences explaining all demonstrations:
<|im_end|>
<|im_start|>assistant
"""

    def _parse_and_validate_response(self, text: str, max_candidates: int) -> List[DSLProgram]:
        """Parses model output and validates primitives against formal DSL schema."""
        programs: List[DSLProgram] = []
        try:
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx == -1 or end_idx == -1:
                return []
            data = json.loads(text[start_idx : end_idx + 1])
            raw_progs = data.get("programs", [])
            self.telemetry.proposals_count += len(raw_progs)
            self.telemetry.genuine_qwen_proposal_count += len(raw_progs)

            for prog_dict in raw_progs:
                name = prog_dict.get("name", "qwen_proposal")
                raw_calls = prog_dict.get("calls", [])
                valid_calls: List[PrimitiveCall] = []

                for c in raw_calls:
                    prim_name = c.get("primitive", c.get("name"))
                    params = c.get("params", c.get("kwargs", {}))

                    # Validate primitive existence
                    if prim_name not in PRIMITIVE_REGISTRY or prim_name not in PRIMITIVE_SPECS:
                        if prim_name:
                            self.telemetry.unsupported_capability_requests.append(str(prim_name))
                        continue

                    # Validate required parameters
                    spec = PRIMITIVE_SPECS[prim_name]
                    if not all(k in params for k in spec.required_params):
                        continue

                    valid_calls.append(PrimitiveCall(primitive_name=prim_name, params=params))

                if valid_calls:
                    programs.append(DSLProgram(name=f"qwen_{name}", calls=valid_calls, origin="qwen2.5_coder"))
                    self.telemetry.valid_dsl_programs_count += 1
                    self.telemetry.valid_qwen_proposal_count += 1
                    if len(programs) >= max_candidates:
                        break
        except Exception as e:
            self.telemetry.failures.append(f"JSON parsing error: {e}")

        return programs

    def _symbolic_relational_synthesis(
        self,
        task: ArcTask,
        change_report: Optional[RelationalChangeReport],
        max_candidates: int,
        scene_graph_facts: Optional[Dict[str, Any]] = None,
    ) -> List[DSLProgram]:
        """Synthesize candidate programs directly from relational changes and invariants."""
        candidates: List[DSLProgram] = []

        # 1. Check Scene Graph hints (holes, containment, alignment)
        if scene_graph_facts and scene_graph_facts.get("has_holes"):
            # Enclosed hole filling
            out_colors: Set[int] = set()
            in_colors: Set[int] = set()
            for demo in task.train_examples:
                in_colors.update(np.unique(demo.input_grid).tolist())
                out_colors.update(np.unique(demo.output_grid).tolist())
            new_colors = out_colors - in_colors
            for fc in (new_colors if new_colors else [c for c in out_colors if c != 0]):
                candidates.append(
                    DSLProgram(
                        name=f"relational_fill_enclosed_{fc}",
                        calls=[PrimitiveCall("fill_enclosed", {"fill_color": fc})],
                        origin="scene_graph_relational_inference(fill_enclosed)",
                    )
                )

        if change_report is None:
            return candidates[:max_candidates]

        # 2. Use detected color mappings
        if change_report.consistent_recolor_rules:
            calls = [
                PrimitiveCall("replace_color", {"old_color": old_c, "new_color": new_c})
                for old_c, new_c in change_report.consistent_recolor_rules.items()
            ]
            candidates.append(
                DSLProgram(
                    name="relational_color_map",
                    calls=calls,
                    origin="relational_change_inference(recolor)",
                )
            )

        # 3. Check for object deletion / extraction
        if change_report.always_creates_or_destroys:
            for filter_op in ["keep_largest_object", "keep_smallest_object"]:
                candidates.append(
                    DSLProgram(
                        name=f"relational_{filter_op}",
                        calls=[PrimitiveCall(filter_op)],
                        origin=f"relational_change_inference({filter_op})",
                    )
                )

        # 4. Check for geometric motion / gravity
        if change_report.motion_vectors:
            avg_dr = sum(dr for dr, dc in change_report.motion_vectors) / len(change_report.motion_vectors)
            if avg_dr > 0:
                candidates.append(
                    DSLProgram(
                        name="relational_gravity_down",
                        calls=[PrimitiveCall("gravity_down")],
                        origin="relational_change_inference(gravity_down)",
                    )
                )
            elif avg_dr < 0:
                candidates.append(
                    DSLProgram(
                        name="relational_gravity_up",
                        calls=[PrimitiveCall("gravity_up")],
                        origin="relational_change_inference(gravity_up)",
                    )
                )

        return candidates[:max_candidates]
