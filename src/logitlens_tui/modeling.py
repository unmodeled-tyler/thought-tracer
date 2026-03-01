from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ModelArtifactError(RuntimeError):
    """Raised when local model artifacts are missing or unusable."""


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    lm_head: Any
    final_norm: Any | None
    input_device: Any
    model_path: Path


def ensure_real_weights(model_path: Path) -> None:
    if not (model_path / "config.json").exists():
        raise ModelArtifactError(f"Missing required model file: {model_path / 'config.json'}")

    # Accept either a single model.safetensors or sharded model-0000X-of-0000Y.safetensors
    single = model_path / "model.safetensors"
    sharded = sorted(model_path.glob("model-*.safetensors"))
    if not single.exists() and not sharded:
        raise ModelArtifactError(
            f"No safetensors weights found in {model_path}. "
            "Expected model.safetensors or model-00001-of-*.safetensors files."
        )

    # Check for LFS pointers in whichever weight files exist
    weight_files = [single] if single.exists() else sharded
    for candidate in weight_files:
        if is_lfs_pointer(candidate):
            raise ModelArtifactError(
                f"{candidate} is still a Git LFS pointer. Pull the real weights before running the app."
            )
    # Also check consolidated.safetensors if present
    consolidated = model_path / "consolidated.safetensors"
    if consolidated.exists() and is_lfs_pointer(consolidated):
        raise ModelArtifactError(
            f"{consolidated} is still a Git LFS pointer. Pull the real weights before running the app."
        )


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            header = handle.read(200)
    except OSError as exc:
        raise ModelArtifactError(f"Failed to read {path}: {exc}") from exc
    return header.startswith(b"version https://git-lfs.github.com/spec/v1")


def load_ministral_model(model_path: str | Path, *, dequantize_to_bf16: bool = False) -> LoadedModel:
    model_dir = Path(model_path).expanduser().resolve()
    ensure_real_weights(model_dir)

    try:
        import torch
        from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
    except ImportError as exc:
        raise ModelArtifactError(
            "Missing runtime dependencies. Install this project with `pip install -e .` first."
        ) from exc

    quantization_config = None
    if dequantize_to_bf16:
        try:
            from transformers import FineGrainedFP8Config
        except ImportError as exc:
            raise ModelArtifactError(
                "This transformers build does not expose FineGrainedFP8Config."
            ) from exc
        quantization_config = FineGrainedFP8Config(dequantize=True)

    tokenizer = MistralCommonBackend.from_pretrained(str(model_dir))

    model = Mistral3ForConditionalGeneration.from_pretrained(
        str(model_dir),
        device_map="cpu",
        quantization_config=quantization_config,
    )
    model.eval()

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise ModelArtifactError("Could not locate `lm_head` on the loaded model.")

    final_norm = resolve_final_norm(model)

    if hasattr(torch, "set_grad_enabled"):
        torch.set_grad_enabled(False)

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        lm_head=lm_head,
        final_norm=final_norm,
        input_device=resolve_input_device(model, lm_head),
        model_path=model_dir,
    )


def resolve_final_norm(model: Any) -> Any | None:
    candidates = [
        ("language_model", "model", "norm"),
        ("model", "norm"),
        ("language_model", "norm"),
        ("norm",),
    ]
    for path in candidates:
        current = model
        for attribute in path:
            current = getattr(current, attribute, None)
            if current is None:
                break
        if current is not None:
            return current
    return None


def resolve_input_device(model: Any, lm_head: Any) -> Any:
    import torch
    
    modules = [lm_head, model]
    for module in modules:
        parameters = getattr(module, "parameters", None)
        if parameters is None:
            continue
        try:
            parameter = next(parameters())
        except StopIteration:
            continue
        device = getattr(parameter, "device", None)
        if device is not None:
            return device
    
    return torch.device("cpu")
