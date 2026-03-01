# Formulas and Methodology

This document describes all mathematical computations used in Thought Tracer's analysis pipeline. All formulas operate on the hidden states of a Ministral transformer model (3B with 26 layers or 8B with 34 layers, vocabulary size 131,072).

---

## 1. Logit Lens (Per-Layer Prediction)

The core technique. At each transformer layer, we project the intermediate hidden state into vocabulary space to see what the model would predict if decoding stopped at that layer.

For a given token position `t` and layer `l`:

```
h_l = hidden_states[l][:, t, :]       # Extract hidden state vector
h_norm = RMSNorm(h_l)                  # Apply the model's final RMS normalization
logits_l = lm_head(h_norm)             # Project to vocabulary space via the language model head
```

The top-k tokens are selected by sorting `logits_l` in descending order.

**Source:** `lens.py:analyze_position()`

---

## 2. Softmax Probability

Converts raw logits into a probability distribution over the vocabulary.

```
P(token_i) = exp(logit_i) / sum_j(exp(logit_j))
```

Where the sum is over all tokens in the vocabulary (131,072 tokens). This is the standard softmax function. Used for both predicted confidence and actual token probability.

**Source:** `lens.py:compute_token_accuracy()`, `lens.py:compute_layer_entropies()`

---

## 3. Token Rank

The 0-indexed rank of a specific token in the model's prediction distribution. Computed by counting how many tokens have a strictly higher logit value.

```
rank(token_i) = |{ j : logit_j > logit_i }|
```

A rank of 0 means the token is the model's top prediction. Displayed as 1-indexed in the UI (rank 0 shows as "#1").

**Source:** `lens.py:compute_token_accuracy()`

---

## 4. Shannon Entropy (Per-Layer)

Measures the uncertainty/spread of the prediction distribution at each layer. Computed in bits (base-2 logarithm).

```
H(l) = -sum_i( P_l(token_i) * log2(P_l(token_i)) )
```

Where `P_l` is the softmax distribution at layer `l`. A small epsilon (1e-10) is used to clamp probabilities before taking the logarithm to avoid `log(0)`.

**Interpretation:**
- Low entropy (~0-2 bits): Model is very confident, probability concentrated on few tokens
- Moderate entropy (~2-5 bits): Some uncertainty, probability spread across several tokens
- High entropy (~5+ bits): Model is uncertain, probability spread widely
- Theoretical maximum: log2(131,072) = 17.0 bits (uniform distribution over entire vocabulary)

**Source:** `lens.py:compute_layer_entropies()`

---

## 5. Layer Agreement (Convergence Detection)

Determines the earliest layer at which the model's top-1 prediction matches the final layer's top-1 prediction and remains stable through all subsequent layers.

```
convergence_layer = min{ l : for all l' >= l, top1(l') == top1(L_final) }
```

If no such layer exists (the prediction "flickers" and never stabilizes), convergence is reported as `None`.

**Source:** `enhanced_app.py:update_agreement_chart()`, `lens.py:compute_hallucination_risk()`

---

## 6. Hallucination Risk Score (Per-Token)

A composite score from 0 to 1 estimating the likelihood that the model's prediction at a single token position is unreliable. Combines three normalized factors with a weighted sum.

### 6.1 Entropy Factor (weight: 0.40)

Normalizes the final layer's Shannon entropy against a practical ceiling of 8 bits, rather than the theoretical maximum of 17 bits. In practice, language model entropy rarely exceeds 8 bits even for highly uncertain predictions.

```
entropy_score = min(H(L_final) / 8.0, 1.0)
```

### 6.2 Convergence Factor (weight: 0.30)

Penalizes late convergence, but only when convergence occurs in the last third of layers. Convergence in the first two-thirds of the network is considered normal and contributes zero risk.

```
depth_ratio = convergence_layer / (num_layers - 1)

if convergence_layer exists:
    convergence_score = max(0, (depth_ratio - 0.66) / 0.34)
else:
    convergence_score = 1.0    # no convergence = maximum risk
```

This maps (example for 8B, 34 layers):
- Layers 0-21 (first 2/3): convergence_score = 0.0
- Layer 25: convergence_score ~ 0.26
- Layer 30: convergence_score ~ 0.70
- Layer 33 (final): convergence_score = 1.0
- No convergence: convergence_score = 1.0

### 6.3 Confidence Factor (weight: 0.30)

Penalizes low top-1 prediction probability with a graduated scale. High confidence (>50%) contributes zero risk; very low confidence (<10%) contributes maximum risk.

```
if top1_prob >= 0.5:
    confidence_score = 0.0
elif top1_prob >= 0.1:
    confidence_score = 1.0 - (top1_prob - 0.1) / 0.4
else:
    confidence_score = 1.0
```

This maps:
- 50%+ probability: confidence_score = 0.0
- 30% probability: confidence_score = 0.5
- 10% probability: confidence_score = 1.0
- <10% probability: confidence_score = 1.0

### 6.4 Composite Score

```
risk_score = (0.40 * entropy_score) + (0.30 * convergence_score) + (0.30 * confidence_score)
```

### 6.5 Risk Level Thresholds

| Score Range | Label       | Interpretation                               |
|-------------|-------------|----------------------------------------------|
| 0.00 - 0.19 | Low         | Model is confident and decisive               |
| 0.20 - 0.39 | Moderate    | Some uncertainty present                      |
| 0.40 - 0.64 | High        | Model shows significant uncertainty           |
| 0.65 - 1.00 | Very High   | Model is highly uncertain — likely hallucination |

**Source:** `lens.py:compute_hallucination_risk()`

---

## 7. Prompt-Level Risk Summary

Aggregates per-token hallucination risk across all content tokens in the prompt to give a whole-prompt risk assessment.

### 7.1 Per-Token Risk

The hallucination risk score from Section 6 is computed independently for each content (non-special) token position in the prompt.

```
per_token_risks = [risk_score(t) for t in content_positions]
```

### 7.2 Average Risk

```
avg_risk = sum(per_token_risks) / len(per_token_risks)
```

The average is classified using the same thresholds as per-token risk (Section 6.5).

### 7.3 Maximum Risk Token

```
max_risk = max(per_token_risks)
max_risk_position = argmax(per_token_risks)
```

Identifies the single token position where the model is most uncertain, which is the most likely site of hallucination.

**Source:** `lens.py:compute_prompt_risk_summary()`

---

## Implementation Notes

- **Model support:** The tool supports both Ministral 3B (26 transformer layers) and Ministral 8B (34 transformer layers), selectable at runtime.
- **Hidden state access:** The tool accesses the inner language model (`model.language_model.model`) directly rather than the outer multimodal wrapper (`Mistral3ForConditionalGeneration`) to ensure all transformer layer hidden states are available.
- **Normalization:** All logit projections apply the model's own final RMSNorm before the language model head, matching the model's actual decoding pipeline.
- **Special token filtering:** Template tokens (`[INST]`, `[/INST]`, `<s>`, etc.) are excluded from navigation and analysis using the tokenizer's `all_special_ids` set, since predictions at template boundaries are not meaningful.
- **Model caching:** Loaded models are cached on the app instance by path, so switching back to a previously loaded model does not require reloading weights.
