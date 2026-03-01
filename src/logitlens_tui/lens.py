from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TopPrediction:
    token_id: int
    token_text: str
    logit: float


@dataclass
class LayerLensResult:
    layer_index: int
    predictions: list[TopPrediction]


@dataclass
class PromptState:
    input_ids: list[int]
    token_texts: list[str]
    next_token_texts: list[str | None]
    hidden_states: tuple[Any, ...]
    content_positions: list[int]  # indices of non-special tokens


def prepare_prompt_state(loaded: Any, *, system_prompt: str | None, user_prompt: str) -> PromptState:
    import torch

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    tokenized = loaded.tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
    )

    tokenized = {name: value.to(loaded.input_device) for name, value in tokenized.items()}

    with torch.inference_mode():
        # Run the inner language model directly to get all layer hidden states.
        # The outer Mistral3ForConditionalGeneration wrapper may not expose
        # all hidden states from the text model.
        language_model = getattr(loaded.model, "language_model", None)
        if language_model is not None:
            inner_model = getattr(language_model, "model", language_model)
            outputs = inner_model(
                **tokenized,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            outputs = loaded.model(
                **tokenized,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )

    input_ids_tensor = tokenized["input_ids"][0]
    input_ids = input_ids_tensor.tolist()
    token_texts = [render_token(loaded.tokenizer, token_id) for token_id in input_ids]
    next_token_texts = token_texts[1:] + [None]

    # Build list of content (non-special) token positions
    special_ids = set()
    if hasattr(loaded.tokenizer, "all_special_ids"):
        special_ids = set(loaded.tokenizer.all_special_ids)
    content_positions = [i for i, tid in enumerate(input_ids) if tid not in special_ids]

    return PromptState(
        input_ids=input_ids,
        token_texts=token_texts,
        next_token_texts=next_token_texts,
        hidden_states=tuple(outputs.hidden_states),
        content_positions=content_positions,
    )


def analyze_position(loaded: Any, prompt_state: PromptState, *, position: int, top_k: int) -> list[LayerLensResult]:
    import torch

    if position < 0 or position >= len(prompt_state.input_ids):
        raise IndexError(f"Token position {position} is out of range.")

    results: list[LayerLensResult] = []
    with torch.inference_mode():
        for layer_index, hidden_state in enumerate(prompt_state.hidden_states[1:], start=0):
            vector = hidden_state[:, position, :]
            if loaded.final_norm is not None:
                vector = loaded.final_norm(vector)
            logits = loaded.lm_head(vector)
            values, indices = logits[0].topk(top_k)
            predictions = [
                TopPrediction(
                    token_id=int(token_id),
                    token_text=render_token(loaded.tokenizer, int(token_id)),
                    logit=float(logit),
                )
                for logit, token_id in zip(values.tolist(), indices.tolist(), strict=True)
            ]
            results.append(LayerLensResult(layer_index=layer_index, predictions=predictions))
    return results


@dataclass
class TokenAccuracy:
    rank: int  # 0-indexed rank of the actual token in the final layer's predictions
    probability: float  # softmax probability assigned to the actual token
    predicted_probability: float  # softmax probability of the model's top-1 prediction


def compute_token_accuracy(
    loaded: Any, prompt_state: PromptState, *, position: int, actual_token_position: int | None = None
) -> TokenAccuracy | None:
    """Compute rank and probability of the actual next token at the final layer.

    Args:
        position: The position to analyze (where logits are computed from).
        actual_token_position: The position of the actual next token in input_ids.
            If None, defaults to position + 1.
    """
    import torch

    if actual_token_position is None:
        actual_token_position = position + 1

    # Need valid next token
    if actual_token_position >= len(prompt_state.input_ids):
        return None

    actual_token_id = prompt_state.input_ids[actual_token_position]

    # Get final layer hidden state and compute logits
    with torch.inference_mode():
        final_hidden = prompt_state.hidden_states[-1]
        vector = final_hidden[:, position, :]
        if loaded.final_norm is not None:
            vector = loaded.final_norm(vector)
        logits = loaded.lm_head(vector)[0]

        # Rank: how many tokens have a higher logit than the actual token
        actual_logit = logits[actual_token_id]
        rank = int((logits > actual_logit).sum().item())

        # Probability via softmax
        probs = torch.softmax(logits, dim=-1)
        probability = float(probs[actual_token_id].item())

        # Top-1 prediction's probability
        predicted_probability = float(probs.max().item())

    return TokenAccuracy(rank=rank, probability=probability, predicted_probability=predicted_probability)


@dataclass
class LayerEntropy:
    layer_index: int
    entropy: float  # Shannon entropy in bits


def compute_layer_entropies(
    loaded: Any, prompt_state: PromptState, *, position: int
) -> list[LayerEntropy]:
    """Compute Shannon entropy of the prediction distribution at each layer."""
    import torch

    results: list[LayerEntropy] = []
    with torch.inference_mode():
        for layer_index, hidden_state in enumerate(prompt_state.hidden_states[1:], start=0):
            vector = hidden_state[:, position, :]
            if loaded.final_norm is not None:
                vector = loaded.final_norm(vector)
            logits = loaded.lm_head(vector)[0]

            # Shannon entropy in bits: -sum(p * log2(p))
            probs = torch.softmax(logits, dim=-1)
            # Clamp to avoid log(0)
            log_probs = torch.log2(probs.clamp(min=1e-10))
            entropy = float(-(probs * log_probs).sum().item())

            results.append(LayerEntropy(layer_index=layer_index, entropy=entropy))

    return results


@dataclass
class HallucinationRisk:
    risk_score: float  # 0-1, higher = more risk
    entropy_score: float
    convergence_score: float
    confidence_score: float
    final_entropy: float
    convergence_layer: int | None
    top1_prob: float


def compute_hallucination_risk(
    loaded: Any, prompt_state: PromptState, *, position: int
) -> HallucinationRisk:
    """Compute hallucination risk score for a single token position."""
    # Layer results for convergence
    layer_results = analyze_position(loaded, prompt_state, position=position, top_k=1)
    final_top1 = layer_results[-1].predictions[0].token_text
    num_layers = len(layer_results)
    convergence_layer: int | None = None
    for i in range(num_layers):
        if all(r.predictions[0].token_text == final_top1 for r in layer_results[i:]):
            convergence_layer = i
            break

    # Final-layer entropy
    entropies = compute_layer_entropies(loaded, prompt_state, position=position)
    final_entropy = entropies[-1].entropy if entropies else 0.0

    # Top-1 confidence
    accuracy = compute_token_accuracy(
        loaded, prompt_state,
        position=position,
        actual_token_position=position + 1,
    )
    top1_prob = accuracy.predicted_probability if accuracy else 0.0

    # Factor 1: entropy
    practical_max_entropy = 8.0
    entropy_score = min(final_entropy / practical_max_entropy, 1.0)

    # Factor 2: convergence depth
    if convergence_layer is not None:
        depth_ratio = convergence_layer / (num_layers - 1) if num_layers > 1 else 0.0
        convergence_score = max(0.0, (depth_ratio - 0.66) / 0.34)
    else:
        convergence_score = 1.0

    # Factor 3: confidence
    if top1_prob >= 0.5:
        confidence_score = 0.0
    elif top1_prob >= 0.1:
        confidence_score = 1.0 - ((top1_prob - 0.1) / 0.4)
    else:
        confidence_score = 1.0

    risk_score = (0.40 * entropy_score) + (0.30 * convergence_score) + (0.30 * confidence_score)

    return HallucinationRisk(
        risk_score=risk_score,
        entropy_score=entropy_score,
        convergence_score=convergence_score,
        confidence_score=confidence_score,
        final_entropy=final_entropy,
        convergence_layer=convergence_layer,
        top1_prob=top1_prob,
    )


@dataclass
class PromptRiskSummary:
    avg_risk: float
    max_risk: float
    max_risk_position: int
    per_token: list[tuple[int, str, float]]  # (position, token_text, risk_score)


def compute_prompt_risk_summary(
    loaded: Any, prompt_state: PromptState
) -> PromptRiskSummary:
    """Compute hallucination risk across all content tokens and aggregate."""
    per_token: list[tuple[int, str, float]] = []
    for pos in prompt_state.content_positions:
        risk = compute_hallucination_risk(loaded, prompt_state, position=pos)
        token_text = prompt_state.token_texts[pos]
        per_token.append((pos, token_text, risk.risk_score))

    scores = [r for _, _, r in per_token]
    avg_risk = sum(scores) / len(scores) if scores else 0.0
    max_idx = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
    max_risk = scores[max_idx] if scores else 0.0
    max_risk_position = per_token[max_idx][0] if per_token else 0

    return PromptRiskSummary(
        avg_risk=avg_risk,
        max_risk=max_risk,
        max_risk_position=max_risk_position,
        per_token=per_token,
    )


def render_token(tokenizer: Any, token_id: int) -> str:
    if hasattr(tokenizer, "decode"):
        text = tokenizer.decode([token_id])
    else:
        text = str(token_id)
    return sanitize_token_text(text)


def sanitize_token_text(text: str) -> str:
    replaced = (
        text.replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("[", r"\[")
        .replace("]", r"\]")
    )
    return replaced if replaced else "<empty>"
