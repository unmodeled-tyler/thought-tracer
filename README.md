# Thought Tracer

A terminal-based model interpretability tool that lets you see inside a language model's decision-making process, layer by layer. Built on the **logit lens** technique applied to **Ministral 3B/8B.**

This research tool was built in less than 48 hours at the Mistral Worldwide Hackathon (San Francisco Edition). 

[Link to video demo on YouTube](https://www.youtube.com/watch?v=mcmC3lf-S_o)

[Read the original post on LessWrong that inspired this project](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)

## What It Does

Thought Tracer intercepts the hidden states at every transformer layer and projects them into vocabulary space, revealing what the model would predict if it stopped thinking at each layer. This exposes:

- **Where the model "locks in" its answer** — the convergence point where the top prediction stabilizes
- **How uncertain the model is** — per-layer Shannon entropy showing confidence distribution
- **Hallucination risk** — a composite score estimating prediction reliability based on entropy, convergence depth, and top-1 confidence

## Features

### Layer Predictions
A full table of top-k predictions at every layer (0-25), showing how the model's thinking evolves from early pattern matching to final prediction.

<img width="3384" height="2192" alt="Screenshot From 2026-02-28 18-43-06" src="https://github.com/user-attachments/assets/8609d4b6-dfdd-4287-921b-ff98f414a5b7" />


### Layer Agreement Chart
A visual map of all 26 layers showing which layers agree with the final prediction. Highlights the convergence point — the first layer where the prediction matches the final answer and stays matched through all remaining layers.

### Entropy

<img width="3384" height="2192" alt="Screenshot From 2026-02-28 18-48-31" src="https://github.com/user-attachments/assets/6dbfdfbf-6124-4a9b-b700-8768be1e812e" />


- **Per-layer entropy chart** — Shannon entropy in bits at each layer, color-coded by uncertainty level
- **Hallucination risk score** — a weighted composite of three factors:
- Entropy factor (40%): final layer prediction uncertainty
- Convergence factor (30%): how late the model settled on its answer
- Confidence factor (30%): top-1 prediction probability
- **Accuracy metrics** — rank and probability of the actual next token, plus the model's confidence in its own prediction

### AI Analysis (Mistral API)
Press `a` to send all computed metrics to `mistral-large-latest`, which returns a plain-English interpretation of the model's behavior at the current token position — covering decision-making patterns, entropy/convergence insights, hallucination risk drivers, and notable layer-by-layer transitions. Requires a Mistral API key (see setup below).

### Smart Token Navigation
Template tokens (`[INST]`, `[/INST]`, `<s>`, etc.) are automatically filtered from navigation and display, since predictions at template boundaries are not meaningful. You only navigate through actual content tokens.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

`transformers` support for this model currently comes from the main branch rather than a stable release, so the dependency is pinned to the GitHub source in `pyproject.toml`.

## Mistral API Key (for AI Analysis)

To use the AI Analysis feature, create a `.env` file in the project root with your Mistral API key:

```bash
echo 'MISTRAL_API_KEY=your-key-here' >> .env
```

You can get an API key from [console.mistral.ai](https://console.mistral.ai/). The key is only used when you press `a` — all other features work fully offline.

## Download Models

This tool has a dropdown selector to change the model from Ministral 3B and Ministral 8B - At least *one* of those models is required for Thought Tracer to work. Due to the nature of the tool, the model must be it's original tensorfile format. 

Ministral 3 3B: https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512
Ministral 3 8B: https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512

## Run

```bash
python3 run_enhanced.py
```

The app will prompt for an optional system prompt and a required user prompt, then launch the interactive analysis interface.

## Controls

| Key | Action |
|-----|--------|
| `n` | Next content token |
| `p` | Previous content token |
| `j` | Jump to token index (enter index in the input field) |
| `k` | Change top-k (enter value in the input field) |
| `r` | Refresh current view |
| `a` | Generate AI analysis (requires API key) |
| `Escape` | Return to prompt input |
| `q` | Quit |

## Documentation

See [`FORMULAS.md`](FORMULAS.md) for a complete description of all mathematical computations, including the logit lens projection, entropy calculation, convergence detection, and hallucination risk scoring methodology.

## Disclaimers

- **Hallucination risk score is heuristic.** The composite score uses hand-tuned weights and thresholds chosen for intuitive calibration, not empirically validated against a ground-truth hallucination dataset. It is a principled starting point for surfacing model uncertainty, not a peer-reviewed metric. See `FORMULAS.md` for full transparency on the formula.
- **Logit lens is an approximation.** Projecting intermediate hidden states through the final layer norm and language model head assumes these components behave meaningfully on inputs they weren't designed for. This is a widely-used interpretability technique but has known limitations — intermediate representations may not fully "mean" what the projected tokens suggest. See [nostalgebraist's original blog post](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) for discussion.
- **Entropy normalization is practical, not theoretical.** Per-layer entropy is normalized against 8 bits (a practical ceiling) rather than the theoretical maximum of ~17 bits (log2 of vocabulary size). This was chosen because real-world language model entropy rarely approaches the theoretical max, and normalizing against it would compress all meaningful variation into a narrow band.
- **Single-token analysis.** The tool analyzes one token position at a time. Hallucination risk at a single position does not necessarily reflect the reliability of a full generated sequence.
- **Text-only.** The model is multimodal (Mistral 3 / Pixtral architecture) but this tool analyzes text prompts only. Vision features are not exercised.

## Architecture

```
src/logitlens_tui/
  modeling.py      # Model loading, weight validation, device resolution
  lens.py          # Core analysis: logit lens, entropy, token accuracy
  enhanced_app.py  # Textual TUI: screens, widgets, visualization
```

## Notes

- The app accesses the inner language model (`model.language_model.model`) directly rather than the outer multimodal wrapper to ensure all 26 transformer layer hidden states are available.
- Analysis is performed one token position across all layers at a time to keep memory use controlled.
- Requires Python 3.11+.
