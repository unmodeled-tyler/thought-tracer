"""
Enhanced Thought Tracer UI using Textual framework.
A sophisticated terminal interface for exploring model predictions.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Markdown,
    Select,
    Static,
    DataTable,
    ProgressBar,
    TabbedContent,
    TabPane,
)
from textual.screen import Screen

from .lens import LayerEntropy, LayerLensResult, PromptState, PromptRiskSummary, TokenAccuracy, analyze_position, compute_layer_entropies, compute_prompt_risk_summary, compute_token_accuracy, prepare_prompt_state
from .modeling import LoadedModel, load_ministral_model


class TokenDisplay(Widget):
    """Custom widget for displaying token information."""

    def __init__(self, token_text: str, is_selected: bool = False) -> None:
        super().__init__()
        self.token_text = token_text
        self.is_selected = is_selected

    def render(self) -> str:
        if self.is_selected:
            return f"[bold #e8e8e8 on #3a5f8a]{self.token_text}[/]"
        else:
            return f"[#8fa8c8]{self.token_text}[/]"


class PredictionTable(DataTable):
    """Enhanced table for showing layer predictions."""

    def __init__(self, top_k: int = 5) -> None:
        super().__init__()
        self.top_k = top_k
        self.cursor_type = "row"
        self.zebra_stripes = True

    def update_predictions(self, results: list[LayerLensResult]) -> None:
        """Update the table with new prediction data."""
        self.clear(columns=True)

        # Add columns
        self.add_column("Layer", key="layer", width=8)
        for i in range(self.top_k):
            self.add_column(f"Top {i+1}", key=f"pred_{i}", width=20)

        # Add rows
        for result in results:
            row_data = [str(result.layer_index)]
            for pred in result.predictions:
                row_data.append(f"{pred.token_text} ({pred.logit:.2f})")
            self.add_row(*row_data, key=str(result.layer_index))


class ThoughtTracerScreen(Screen):
    """Main screen for the Thought Tracer application."""

    BINDINGS = [
        ("n", "next_token", "Next Token"),
        ("p", "prev_token", "Previous Token"),
        ("j", "jump_token", "Jump to Token"),
        ("k", "change_k", "Change Top-K"),
        ("r", "refresh", "Refresh View"),
        ("a", "ai_analysis", "AI Analysis"),
        ("escape", "new_prompt", "New Prompt"),
        ("q", "quit", "Quit"),
    ]

    top_k = reactive(5)
    selected_position = reactive(0)
    _content_index: int = 0  # index into content_positions list

    def __init__(self, loaded: LoadedModel, system_prompt: str | None, user_prompt: str) -> None:
        super().__init__()
        self.loaded = loaded
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.prompt_state: PromptState | None = None
        self.layer_results: list[LayerLensResult] = []
        self.prompt_risk_summary: PromptRiskSummary | None = None

    def compose(self) -> ComposeResult:
        """Compose the screen layout."""
        yield Header(show_clock=True)

        with Container(id="main-container"):
            # Prompt info section
            with Container(id="prompt-info"):
                yield Label("Prompt Analysis", id="prompt-title")
                yield Static(id="system-prompt-display")
                yield Static(id="user-prompt-display")
                yield ProgressBar(id="token-progress", total=100)

            # Token navigation
            with Container(id="token-nav"):
                yield Button("< Previous", id="prev-btn", variant="default")
                yield Static(id="token-position")
                yield Button("Next >", id="next-btn", variant="default")

            # Token display
            yield ScrollableContainer(
                Static("Loading tokens...", id="token-display"),
                id="token-scroll"
            )

            # Predictions tabbed interface
            with TabbedContent(id="predictions-tabs"):
                with TabPane("Layer Predictions", id="layer-preds"):
                    table = PredictionTable(top_k=5)
                    table.id = "prediction-table"
                    yield table
                with TabPane("Layer Agreement", id="attention-viz"):
                    yield ScrollableContainer(
                        Static("", id="agreement-chart"),
                        id="agreement-scroll"
                    )
                with TabPane("Entropy", id="token-stats"):
                    yield ScrollableContainer(
                        Static("", id="stats-chart"),
                        id="stats-scroll"
                    )
                with TabPane("AI Analysis", id="ai-analysis"):
                    yield ScrollableContainer(
                        Markdown("*Press 'a' to generate AI analysis...*", id="ai-summary"),
                        id="ai-scroll"
                    )

            # Control panel
            with Container(id="controls"):
                yield Label("Controls")
                yield Button("Refresh", id="refresh-btn", variant="default")
                yield Button("New Prompt", id="new-prompt-btn", variant="default")
                yield Input(placeholder="Jump to token index...", id="jump-input")
                yield Input(placeholder="Change top-k...", id="topk-input")
                yield Static(id="status-bar")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the screen when mounted."""
        # Prepare the prompt state
        self.prompt_state = prepare_prompt_state(
            self.loaded,
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt
        )

        # Update UI with prompt info
        sys_display = self.query_one("#system-prompt-display", Static)
        user_display = self.query_one("#user-prompt-display", Static)

        if self.system_prompt:
            sys_display.update(f"[bold #a0a0a0]SYSTEM[/]  {self.system_prompt}")
        else:
            sys_display.update("[dim #707070]SYSTEM  (none)[/]")

        user_display.update(f"[bold #a0a0a0]USER[/]    {self.user_prompt}")

        # Initialize token display — start at the last content token
        self._content_index = len(self.prompt_state.content_positions) - 1
        self.selected_position = self.prompt_state.content_positions[self._content_index]
        self.update_token_display()
        self.update_predictions()

        # Compute aggregate prompt risk in background
        self.run_worker(self._compute_prompt_risk(), exclusive=True, group="prompt_risk")

    async def _compute_prompt_risk(self) -> None:
        """Compute aggregate risk across all content tokens in a background worker."""
        import asyncio

        if not self.prompt_state:
            return

        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(
            None, compute_prompt_risk_summary, self.loaded, self.prompt_state
        )
        self.prompt_risk_summary = summary
        # Re-render the stats tab to include the aggregate
        self.update_token_stats()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "prev-btn":
            self.action_prev_token()
        elif event.button.id == "next-btn":
            self.action_next_token()
        elif event.button.id == "refresh-btn":
            self.action_refresh()
        elif event.button.id == "new-prompt-btn":
            self.action_new_prompt()

    def update_token_display(self) -> None:
        """Update the token display with current selection."""
        if not self.prompt_state:
            return

        token_display = self.query_one("#token-display", Static)
        position_display = self.query_one("#token-position", Static)
        progress_bar = self.query_one("#token-progress", ProgressBar)

        # Update position info
        num_content = len(self.prompt_state.content_positions)
        position_display.update(
            f"Token {self._content_index + 1} of {num_content}"
        )

        # Update progress bar
        progress_bar.total = num_content - 1
        progress_bar.progress = self._content_index

        # Build token visualization — only show content tokens
        content_set = set(self.prompt_state.content_positions)
        tokens_html = ""
        for i, token_text in enumerate(self.prompt_state.token_texts):
            if i not in content_set:
                continue
            if i == self.selected_position:
                tokens_html += f"[bold #e8e8e8 on #3a5f8a]{token_text}[/] "
            else:
                tokens_html += f"[#8fa8c8]{token_text}[/] "

        # Add next token info — find next content token, not just next raw token
        current_token = self.prompt_state.token_texts[self.selected_position]
        if self._content_index + 1 < len(self.prompt_state.content_positions):
            next_pos = self.prompt_state.content_positions[self._content_index + 1]
            next_token = self.prompt_state.token_texts[next_pos]
        else:
            next_token = "<END>"

        tokens_html += f"\n\n[bold #a0a0a0]Selected:[/] {current_token}\n"
        tokens_html += f"[bold #a0a0a0]Actual Next:[/] {next_token}"

        token_display.update(tokens_html)

    def update_predictions(self) -> None:
        """Update the predictions table."""
        if not self.prompt_state:
            return

        # Analyze current position
        self.layer_results = analyze_position(
            self.loaded,
            self.prompt_state,
            position=self.selected_position,
            top_k=self.top_k
        )

        # Update table
        table = self.query_one("#prediction-table", PredictionTable)
        table.top_k = self.top_k
        table.update_predictions(self.layer_results)

        # Update agreement chart
        self.update_agreement_chart()

        # Update token statistics
        self.update_token_stats()

    def update_agreement_chart(self) -> None:
        """Update the layer-over-layer agreement chart."""
        if not self.layer_results:
            return

        chart = self.query_one("#agreement-chart", Static)

        # Get the final layer's top-1 token as the target
        final_top1 = self.layer_results[-1].predictions[0].token_text
        num_layers = len(self.layer_results)

        # Determine convergence point: first layer where top-1 matches final
        # and stays matched through all subsequent layers
        convergence_layer: int | None = None
        for i in range(num_layers):
            all_match = all(
                r.predictions[0].token_text == final_top1
                for r in self.layer_results[i:]
            )
            if all_match:
                convergence_layer = i
                break

        # Build the chart
        bar_width = 10
        lines: list[str] = []
        for result in self.layer_results:
            layer_idx = result.layer_index
            top1 = result.predictions[0].token_text
            matches = top1 == final_top1

            # Layer number (right-aligned)
            layer_label = f"{layer_idx:>2}"

            # Bar segment
            bar_char = "\u2588" * bar_width
            if matches:
                bar = f"[#5a8fba]{bar_char}[/]"
            else:
                bar = f"[#2c3e6b]{bar_char}[/]"

            # Token text
            token_part = f"[#8fa8c8]{top1}[/]"

            # Convergence marker
            marker = ""
            if convergence_layer is not None and layer_idx == convergence_layer:
                marker = "  [bold #e8e8e8]* converges here[/]"
            elif layer_idx == num_layers - 1:
                marker = "  [#8fa8c8](final prediction)[/]"

            lines.append(f"{layer_label}  {bar}  {token_part}{marker}")

        # Summary line
        if convergence_layer is not None:
            summary = f"\n[bold #c8d6e5]Converges at layer {convergence_layer} / {num_layers - 1}[/]"
        else:
            summary = f"\n[bold #c8d6e5]No stable convergence[/]"
        lines.append(summary)

        # Actual next token comparison and accuracy metrics
        # Find the next content token position for comparison
        if self.prompt_state and self._content_index + 1 < len(self.prompt_state.content_positions):
            next_content_pos = self.prompt_state.content_positions[self._content_index + 1]
            actual_next = self.prompt_state.token_texts[next_content_pos]
        else:
            actual_next = None
        if actual_next is not None:
            if actual_next == final_top1:
                lines.append(f"[bold #c8d6e5]Actual next token:[/] [bold #5a8fba]{actual_next}[/] [bold #5a8fba](matches prediction)[/]")
            else:
                lines.append(f"[bold #c8d6e5]Actual next token:[/] [bold #e8e8e8]{actual_next}[/]  [#8fa8c8]vs predicted[/] [bold #e8e8e8]{final_top1}[/]")

            # Compute rank and probability using the next content token
            accuracy = compute_token_accuracy(
                self.loaded, self.prompt_state,
                position=self.selected_position,
                actual_token_position=next_content_pos,
            )
            if accuracy is not None:
                rank_display = f"#{accuracy.rank + 1}"
                prob_pct = accuracy.probability * 100

                # Color rank based on how good it is
                if accuracy.rank == 0:
                    rank_color = "#5a8fba"
                    rank_label = "exact match"
                elif accuracy.rank < 5:
                    rank_color = "#5a8fba"
                    rank_label = "close"
                elif accuracy.rank < 50:
                    rank_color = "#c8d6e5"
                    rank_label = "moderate"
                else:
                    rank_color = "#e74c3c"
                    rank_label = "far off"

                # Color probability based on confidence
                if prob_pct >= 50:
                    prob_color = "#5a8fba"
                elif prob_pct >= 10:
                    prob_color = "#c8d6e5"
                else:
                    prob_color = "#e74c3c"

                # Model's confidence in its own prediction
                pred_pct = accuracy.predicted_probability * 100
                if pred_pct >= 50:
                    pred_color = "#5a8fba"
                elif pred_pct >= 10:
                    pred_color = "#c8d6e5"
                else:
                    pred_color = "#e74c3c"

                lines.append(
                    f"[bold #c8d6e5]Predicted confidence:[/] [bold {pred_color}]{pred_pct:.1f}%[/] [#8fa8c8]for[/] [bold #e8e8e8]{final_top1}[/]"
                )
                lines.append(
                    f"[bold #c8d6e5]Actual token rank:[/] [bold {rank_color}]{rank_display}[/] [#8fa8c8]({rank_label})[/]"
                    f"    [bold #c8d6e5]Actual token prob:[/] [bold {prob_color}]{prob_pct:.1f}%[/]"
                )
        else:
            lines.append(f"[bold #c8d6e5]Actual next token:[/] [dim #707070]<END>[/]")

        chart.update("\n".join(lines))

    def update_token_stats(self) -> None:
        """Update the token statistics tab with entropy and hallucination risk."""
        if not self.layer_results or not self.prompt_state:
            return

        stats = self.query_one("#stats-chart", Static)

        # Compute per-layer entropy
        entropies = compute_layer_entropies(
            self.loaded, self.prompt_state, position=self.selected_position
        )

        # Max possible entropy for reference (log2 of vocab size)
        import math
        vocab_size = len(self.loaded.tokenizer)
        max_entropy = math.log2(vocab_size)

        # --- Entropy sparkline chart ---
        lines: list[str] = []
        lines.append("[bold #c8d6e5]Entropy by Layer[/] [#8fa8c8](bits)[/]\n")

        max_ent = max(e.entropy for e in entropies) if entropies else 1.0
        bar_max_width = 30
        for entry in entropies:
            layer_label = f"{entry.layer_index:>2}"
            # Scale bar width relative to max observed entropy
            bar_width = int((entry.entropy / max_ent) * bar_max_width) if max_ent > 0 else 0
            bar_width = max(bar_width, 1)

            # Color based on entropy level (relative to max possible)
            ent_ratio = entry.entropy / max_entropy
            if ent_ratio < 0.15:
                color = "#5a8fba"  # low entropy — confident
            elif ent_ratio < 0.35:
                color = "#c8d6e5"  # moderate
            elif ent_ratio < 0.55:
                color = "#e0a458"  # elevated
            else:
                color = "#e74c3c"  # high entropy — uncertain

            bar = "\u2588" * bar_width
            lines.append(f"{layer_label}  [{color}]{bar}[/]  [#8fa8c8]{entry.entropy:.2f}[/]")

        # --- Summary stats ---
        final_entropy = entropies[-1].entropy if entropies else 0.0
        min_entropy = min(e.entropy for e in entropies) if entropies else 0.0
        avg_entropy = sum(e.entropy for e in entropies) / len(entropies) if entropies else 0.0

        lines.append(f"\n[bold #c8d6e5]Final layer entropy:[/] [#8fa8c8]{final_entropy:.2f} bits[/]")
        lines.append(f"[bold #c8d6e5]Min / Avg entropy:[/] [#8fa8c8]{min_entropy:.2f} / {avg_entropy:.2f} bits[/]")
        lines.append(f"[bold #c8d6e5]Max possible:[/] [#8fa8c8]{max_entropy:.1f} bits[/] [dim #707070](vocab={vocab_size:,})[/]")

        # --- Hallucination risk assessment ---
        # Compute convergence depth from agreement chart data
        final_top1 = self.layer_results[-1].predictions[0].token_text
        num_layers = len(self.layer_results)
        convergence_layer: int | None = None
        for i in range(num_layers):
            if all(r.predictions[0].token_text == final_top1 for r in self.layer_results[i:]):
                convergence_layer = i
                break

        # Get predicted confidence (top-1 probability at final layer)
        accuracy = compute_token_accuracy(
            self.loaded, self.prompt_state,
            position=self.selected_position,
            actual_token_position=self.selected_position + 1,  # just need predicted_probability
        )
        top1_prob = accuracy.predicted_probability if accuracy else 0.0

        # Composite risk score (0-1, higher = more risk)

        # Factor 1: Final entropy — normalize against practical range, not
        # theoretical max.  Entropy above ~8 bits is very uncertain for a
        # language model; below ~2 bits is quite confident.
        practical_max_entropy = 8.0
        entropy_score = min(final_entropy / practical_max_entropy, 1.0)

        # Factor 2: Convergence depth — only penalize when convergence is
        # in the last third of layers.  Converging in the first 2/3 is
        # normal and shouldn't inflate risk.
        if convergence_layer is not None:
            depth_ratio = convergence_layer / (num_layers - 1)
            # Map: 0-0.66 → 0, 0.66-1.0 → 0-1 (only late convergence matters)
            convergence_score = max(0.0, (depth_ratio - 0.66) / 0.34)
        else:
            convergence_score = 1.0  # no convergence = max risk

        # Factor 3: Low confidence — only penalize when top-1 prob is weak.
        # Above 50% is fine; below 10% is very concerning.
        if top1_prob >= 0.5:
            confidence_score = 0.0
        elif top1_prob >= 0.1:
            confidence_score = 1.0 - ((top1_prob - 0.1) / 0.4)
        else:
            confidence_score = 1.0

        # Weighted combination
        risk_score = (0.40 * entropy_score) + (0.30 * convergence_score) + (0.30 * confidence_score)

        # Map to risk level
        if risk_score < 0.20:
            risk_label = "Low"
            risk_color = "#5a8fba"
            risk_desc = "Model is confident and decisive"
        elif risk_score < 0.40:
            risk_label = "Moderate"
            risk_color = "#c8d6e5"
            risk_desc = "Some uncertainty present"
        elif risk_score < 0.65:
            risk_label = "High"
            risk_color = "#e0a458"
            risk_desc = "Model shows significant uncertainty"
        else:
            risk_label = "Very High"
            risk_color = "#e74c3c"
            risk_desc = "Model is highly uncertain — likely hallucination"

        lines.append(f"\n[bold #c8d6e5]{'─' * 40}[/]")
        lines.append(f"[bold #c8d6e5]Hallucination Risk:[/] [bold {risk_color}]{risk_label}[/] [#8fa8c8]({risk_score:.0%})[/]")
        lines.append(f"[#8fa8c8]{risk_desc}[/]")
        convergence_display = f"layer {convergence_layer}" if convergence_layer is not None else "none"
        lines.append("")
        lines.append(f"  [#8fa8c8]Entropy factor:[/]     [{risk_color}]{entropy_score:.0%}[/]  [dim #707070]({final_entropy:.2f} bits)[/]")
        lines.append(f"  [#8fa8c8]Convergence factor:[/] [{risk_color}]{convergence_score:.0%}[/]  [dim #707070]({convergence_display})[/]")
        lines.append(f"  [#8fa8c8]Confidence factor:[/]  [{risk_color}]{confidence_score:.0%}[/]  [dim #707070](top-1 prob: {top1_prob:.1%})[/]")

        # --- Aggregate prompt-level risk ---
        lines.append(f"\n[bold #c8d6e5]{'─' * 40}[/]")
        if self.prompt_risk_summary is not None:
            s = self.prompt_risk_summary

            # Color the average
            if s.avg_risk < 0.20:
                avg_color = "#5a8fba"
                avg_label = "Low"
            elif s.avg_risk < 0.40:
                avg_color = "#c8d6e5"
                avg_label = "Moderate"
            elif s.avg_risk < 0.65:
                avg_color = "#e0a458"
                avg_label = "High"
            else:
                avg_color = "#e74c3c"
                avg_label = "Very High"

            # Color the max
            if s.max_risk < 0.20:
                max_color = "#5a8fba"
            elif s.max_risk < 0.40:
                max_color = "#c8d6e5"
            elif s.max_risk < 0.65:
                max_color = "#e0a458"
            else:
                max_color = "#e74c3c"

            max_token = ""
            for pos, tok, _ in s.per_token:
                if pos == s.max_risk_position:
                    max_token = tok
                    break

            lines.append(f"[bold #c8d6e5]Prompt-Level Risk:[/] [bold {avg_color}]{avg_label}[/] [#8fa8c8]({s.avg_risk:.0%} avg across {len(s.per_token)} tokens)[/]")
            lines.append(f"  [#8fa8c8]Highest risk token:[/] [bold {max_color}]{max_token}[/] [#8fa8c8]({s.max_risk:.0%})[/]")

            # Mini bar chart of per-token risk
            lines.append("")
            lines.append("[bold #c8d6e5]Per-Token Risk:[/]")
            bar_max = 20
            for pos, tok, score in s.per_token:
                if score < 0.20:
                    tc = "#5a8fba"
                elif score < 0.40:
                    tc = "#c8d6e5"
                elif score < 0.65:
                    tc = "#e0a458"
                else:
                    tc = "#e74c3c"
                bar_w = max(1, int(score * bar_max))
                bar = "\u2588" * bar_w
                marker = " <" if pos == self.selected_position else ""
                lines.append(f"  [{tc}]{tok:<12s}[/] [{tc}]{bar}[/] [#8fa8c8]{score:.0%}[/]{marker}")
        else:
            lines.append("[dim #707070]Computing prompt-level risk...[/]")

        stats.update("\n".join(lines))

    # Action handlers
    def action_next_token(self) -> None:
        """Move to next content token."""
        if self.prompt_state:
            if self._content_index + 1 < len(self.prompt_state.content_positions):
                self._content_index += 1
                self.selected_position = self.prompt_state.content_positions[self._content_index]
                self.update_token_display()
                self.update_predictions()

    def action_prev_token(self) -> None:
        """Move to previous content token."""
        if self.prompt_state:
            if self._content_index > 0:
                self._content_index -= 1
                self.selected_position = self.prompt_state.content_positions[self._content_index]
                self.update_token_display()
                self.update_predictions()

    def action_refresh(self) -> None:
        """Refresh the current view."""
        self.update_token_display()
        self.update_predictions()

    def action_jump_token(self) -> None:
        """Jump to specific content token index."""
        jump_input = self.query_one("#jump-input", Input)
        try:
            index = int(jump_input.value)
            if self.prompt_state and 0 <= index < len(self.prompt_state.content_positions):
                self._content_index = index
                self.selected_position = self.prompt_state.content_positions[index]
                self.update_token_display()
                self.update_predictions()
                jump_input.value = ""
        except ValueError:
            pass

    def action_new_prompt(self) -> None:
        """Return to the prompt input screen."""
        self.app.pop_screen()

    def action_change_k(self) -> None:
        """Change top-k value."""
        topk_input = self.query_one("#topk-input", Input)
        try:
            new_k = int(topk_input.value)
            if 1 <= new_k <= 20:
                self.top_k = new_k
                self.update_predictions()
                topk_input.value = ""
        except ValueError:
            pass

    def action_ai_analysis(self) -> None:
        """Trigger AI-powered analysis of the current token position."""
        self.run_worker(self._run_ai_analysis(), exclusive=True, group="ai_analysis")

    async def _run_ai_analysis(self) -> None:
        """Run the Mistral API call in a background worker."""
        import os
        import math

        summary_widget = self.query_one("#ai-summary", Markdown)
        summary_widget.update("*Generating AI analysis... (calling Mistral API)*")

        # Switch to AI Analysis tab
        tabs = self.query_one("#predictions-tabs", TabbedContent)
        tabs.active = "ai-analysis"

        # Check for API key
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            summary_widget.update(
                "## Missing API Key\n\n"
                "Set `MISTRAL_API_KEY` in your `.env` file:\n\n"
                "```bash\necho 'MISTRAL_API_KEY=your-key-here' >> .env\n```"
            )
            return

        if not self.prompt_state or not self.layer_results:
            summary_widget.update("**No analysis data available.** Navigate to a token first.")
            return

        # --- Collect all metrics ---

        # Layer-by-layer top-1 predictions
        layer_predictions = []
        for result in self.layer_results:
            top1 = result.predictions[0]
            layer_predictions.append(
                f"Layer {result.layer_index}: {top1.token_text!r} (logit={top1.logit:.2f})"
            )

        # Convergence layer
        final_top1 = self.layer_results[-1].predictions[0].token_text
        num_layers = len(self.layer_results)
        convergence_layer = None
        for i in range(num_layers):
            if all(r.predictions[0].token_text == final_top1 for r in self.layer_results[i:]):
                convergence_layer = i
                break

        # Per-layer entropy
        entropies = compute_layer_entropies(
            self.loaded, self.prompt_state, position=self.selected_position
        )
        entropy_lines = [
            f"Layer {e.layer_index}: {e.entropy:.2f} bits" for e in entropies
        ]
        final_entropy = entropies[-1].entropy if entropies else 0.0

        # Current token and context
        current_token = self.prompt_state.token_texts[self.selected_position]
        content_positions = self.prompt_state.content_positions
        # Gather a few surrounding content tokens for context
        ctx_start = max(0, self._content_index - 5)
        ctx_end = min(len(content_positions), self._content_index + 5)
        context_tokens = [
            self.prompt_state.token_texts[content_positions[i]]
            for i in range(ctx_start, ctx_end)
        ]

        # Actual next token
        if self._content_index + 1 < len(content_positions):
            next_pos = content_positions[self._content_index + 1]
            actual_next = self.prompt_state.token_texts[next_pos]
        else:
            actual_next = "<END>"

        # Token rank and probability
        if self._content_index + 1 < len(content_positions):
            next_pos = content_positions[self._content_index + 1]
            accuracy = compute_token_accuracy(
                self.loaded, self.prompt_state,
                position=self.selected_position,
                actual_token_position=next_pos,
            )
        else:
            accuracy = None
        rank_str = f"#{accuracy.rank + 1}" if accuracy else "N/A"
        prob_str = f"{accuracy.probability * 100:.1f}%" if accuracy else "N/A"
        pred_prob_str = f"{accuracy.predicted_probability * 100:.1f}%" if accuracy else "N/A"

        # Hallucination risk factors
        vocab_size = len(self.loaded.tokenizer)
        practical_max_entropy = 8.0
        entropy_score = min(final_entropy / practical_max_entropy, 1.0)

        if convergence_layer is not None:
            depth_ratio = convergence_layer / (num_layers - 1)
            convergence_score = max(0.0, (depth_ratio - 0.66) / 0.34)
        else:
            convergence_score = 1.0

        top1_prob = accuracy.predicted_probability if accuracy else 0.0
        if top1_prob >= 0.5:
            confidence_score = 0.0
        elif top1_prob >= 0.1:
            confidence_score = 1.0 - ((top1_prob - 0.1) / 0.4)
        else:
            confidence_score = 1.0

        risk_score = (0.40 * entropy_score) + (0.30 * convergence_score) + (0.30 * confidence_score)

        # --- Build prompt ---
        user_message = f"""Here are the interpretability metrics for a transformer model (Ministral 3B) at a specific token position:

**Current Token:** {current_token!r}
**Context (surrounding tokens):** {"".join(context_tokens)}
**Actual Next Token:** {actual_next!r}
**Model's Predicted Next Token:** {final_top1!r}
**Predicted Token Confidence:** {pred_prob_str}
**Actual Token Rank:** {rank_str}
**Actual Token Probability:** {prob_str}

**Convergence Layer:** {f"layer {convergence_layer} out of {num_layers - 1}" if convergence_layer is not None else "No stable convergence"}

**Hallucination Risk Score:** {risk_score:.0%}
  - Entropy factor: {entropy_score:.0%} (final entropy: {final_entropy:.2f} bits)
  - Convergence factor: {convergence_score:.0%}
  - Confidence factor: {confidence_score:.0%}

**Layer-by-layer top-1 predictions:**
{chr(10).join(layer_predictions)}

**Per-layer entropy (bits):**
{chr(10).join(entropy_lines)}

Please provide:
1. A summary of the model's decision-making process for this token
2. Interpretation of the entropy and convergence patterns
3. Assessment of the hallucination risk and what's driving it
4. Any notable layer-by-layer behavior (e.g., interesting transitions, competing predictions)
"""

        system_message = (
            "You are a transformer interpretability analyst. You analyze logit lens "
            "outputs and internal model metrics to explain how a language model arrives "
            "at its predictions. Be concise, insightful, and use plain English. "
            "Highlight the most interesting patterns. Format your response with clear "
            "sections using markdown headers."
        )

        # --- Call Mistral API ---
        try:
            from mistralai import Mistral

            client = Mistral(api_key=api_key)
            response = await client.chat.complete_async(
                model="mistral-large-latest",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
            )
            result_text = response.choices[0].message.content
            summary_widget.update(result_text)
        except Exception as e:
            summary_widget.update(
                f"## API Error\n\n`{type(e).__name__}: {e}`"
            )


_SHARED_CSS = """
Screen {
    background: #1a1a2e;
}

Header {
    background: #16213e;
    color: #c8d6e5;
}

Footer {
    background: #16213e;
    color: #8fa8c8;
}

#main-container {
    width: 100%;
    height: 100%;
    layout: vertical;
    background: #1a1a2e;
}

#prompt-info {
    border: solid #2c3e6b;
    padding: 1 2;
    margin: 1 0;
    background: #16213e;
}

#prompt-title {
    color: #c8d6e5;
    text-style: bold;
}

#token-nav {
    layout: horizontal;
    padding: 1;
    margin: 1 0;
    background: #1a1a2e;
}

#token-scroll {
    border: solid #2c3e6b;
    height: 10;
    margin: 1 0;
    background: #16213e;
}

#predictions-tabs {
    height: 1fr;
    min-height: 30;
    margin: 1 0;
}

#agreement-scroll {
    height: 1fr;
}

#agreement-chart, #stats-chart {
    height: auto;
}

#ai-summary {
    height: auto;
    margin: 1 2;
    color: #c8d6e5;
}

#stats-scroll {
    height: 1fr;
}

#ai-scroll {
    height: 1fr;
}

#controls {
    border: solid #2c3e6b;
    padding: 1;
    margin: 1 0;
    background: #16213e;
}

Button {
    width: 20;
    background: #2c3e6b;
    color: #c8d6e5;
    border: none;
}

Button:hover {
    background: #3a5f8a;
    color: #e8e8e8;
}

Button:focus {
    background: #3a5f8a;
    color: #e8e8e8;
}

#token-position {
    width: 1fr;
    text-align: center;
    color: #8fa8c8;
}

DataTable {
    background: #16213e;
}

DataTable > .datatable--header {
    background: #2c3e6b;
    color: #c8d6e5;
    text-style: bold;
}

DataTable > .datatable--cursor {
    background: #3a5f8a;
    color: #e8e8e8;
}

DataTable > .datatable--even-row {
    background: #1a1a2e;
}

DataTable > .datatable--odd-row {
    background: #16213e;
}

TabbedContent {
    background: #1a1a2e;
}

ContentSwitcher {
    background: #16213e;
}

TabPane {
    background: #16213e;
}

Tab {
    background: #16213e;
    color: #8fa8c8;
}

Tab.-active {
    background: #2c3e6b;
    color: #e8e8e8;
}

Tabs {
    background: #16213e;
}

Input {
    background: #16213e;
    border: solid #2c3e6b;
    color: #c8d6e5;
}

Input:focus {
    border: solid #3a5f8a;
}

ProgressBar {
    padding: 0 1;
}

Bar > .bar--bar {
    color: #3a5f8a;
}

Bar > .bar--complete {
    color: #5a8fba;
}

Label {
    color: #c8d6e5;
}

Static {
    color: #8fa8c8;
}

#title {
    text-align: center;
    text-style: bold;
    color: #c8d6e5;
    padding: 2 0;
}

#instructions {
    text-align: center;
    color: #8fa8c8;
    padding: 1 0;
}

#model-label, #system-label, #user-label {
    color: #a0a0a0;
    padding: 1 0 0 0;
}

Select {
    width: 60;
}

SelectCurrent {
    background: #16213e;
    border: solid #2c3e6b;
    color: #c8d6e5;
}

SelectCurrent:focus {
    border: solid #3a5f8a;
}
"""


class EnhancedLogitLensApp(App):
    """Enhanced Thought Tracer application with Textual UI."""

    SCREENS = {}

    CSS = _SHARED_CSS

    def __init__(self, loaded: LoadedModel, system_prompt: str | None, user_prompt: str) -> None:
        super().__init__()
        self.loaded = loaded
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def on_mount(self) -> None:
        """Initialize the app."""
        self.push_screen(
            ThoughtTracerScreen(
                self.loaded,
                self.system_prompt,
                self.user_prompt
            )
        )


class PromptInputScreen(Screen):
    """Screen for collecting user prompts."""

    def __init__(self, model_choices: list[tuple[str, str]]) -> None:
        super().__init__()
        self.model_choices = model_choices

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Thought Tracer  |  Logit Lens", id="title")
        yield Static("Enter prompts to analyze model predictions.", id="instructions")

        yield Label("Model:", id="model-label")
        yield Select(
            [(name, path) for name, path in self.model_choices],
            value=self.model_choices[0][1],
            id="model-select",
        )

        yield Label("System Prompt (optional):", id="system-label")
        yield Input(placeholder="You are a helpful assistant...", id="system-input")

        yield Label("User Prompt (required):", id="user-label")
        yield Input(placeholder="Enter text to analyze...", id="user-input")

        yield Button("Analyze", id="submit-btn", variant="primary")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "submit-btn":
            self.start_analysis()

    def start_analysis(self) -> None:
        """Start the analysis with user inputs."""
        system_input = self.query_one("#system-input", Input)
        user_input = self.query_one("#user-input", Input)
        model_select = self.query_one("#model-select", Select)

        system_prompt = system_input.value.strip() or None
        user_prompt = user_input.value.strip()
        model_path = model_select.value

        if model_path is Select.BLANK:
            self.notify("Please select a model.", severity="warning")
            return

        if not user_prompt:
            self.notify("Please enter a user prompt.", severity="warning")
            return

        # Check model cache on app
        model_cache: dict[str, LoadedModel] = self.app._model_cache  # type: ignore[attr-defined]
        if model_path in model_cache:
            self.app.push_screen(
                ThoughtTracerScreen(
                    model_cache[model_path],
                    system_prompt,
                    user_prompt,
                )
            )
        else:
            # Find display name for the selected model
            model_name = model_path
            for name, path in self.model_choices:
                if path == model_path:
                    model_name = name
                    break
            self.notify(f"Loading {model_name}...")
            self.run_worker(
                self._load_and_launch(model_path, system_prompt, user_prompt),
                exclusive=True,
                group="model_load",
            )

    async def _load_and_launch(
        self, model_path: str, system_prompt: str | None, user_prompt: str
    ) -> None:
        """Load the model in a worker thread, then push the analysis screen."""
        import asyncio

        loop = asyncio.get_event_loop()
        loaded = await loop.run_in_executor(None, load_ministral_model, model_path)

        # Cache on the app instance
        self.app._model_cache[model_path] = loaded  # type: ignore[attr-defined]

        self.app.push_screen(
            ThoughtTracerScreen(loaded, system_prompt, user_prompt)
        )


class PromptInputApp(App):
    """App that starts with the prompt input screen."""

    TITLE = "Thought Tracer"
    CSS = _SHARED_CSS

    def __init__(self, model_choices: list[tuple[str, str]]) -> None:
        super().__init__()
        self.model_choices = model_choices
        self._model_cache: dict[str, LoadedModel] = {}

    def on_mount(self) -> None:
        """Push the prompt input screen on mount."""
        self.push_screen(PromptInputScreen(self.model_choices))


class EnhancedAppLauncher:
    """Launcher for the enhanced Thought Tracer application."""

    def __init__(self, model_choices: list[tuple[str, str]]) -> None:
        self.model_choices = model_choices

    def run(self) -> None:
        """Run the enhanced application."""
        app = PromptInputApp(self.model_choices)
        app.run()
