from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .lens import LayerLensResult, PromptState, analyze_position, prepare_prompt_state
from .modeling import LoadedModel


@dataclass
class SessionState:
    prompt_state: PromptState
    selected_position: int
    top_k: int


class LogitLensApp:
    def __init__(self, loaded: LoadedModel, console: Console | None = None) -> None:
        self.loaded = loaded
        self.console = console or Console()

    def run(self) -> None:
        system_prompt = self.console.input("System prompt (optional): ").strip() or None
        user_prompt = self.console.input("User prompt: ").strip()
        if not user_prompt:
            raise SystemExit("A user prompt is required.")

        prompt_state = prepare_prompt_state(
            self.loaded,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        session = SessionState(
            prompt_state=prompt_state,
            selected_position=max(0, len(prompt_state.input_ids) - 1),
            top_k=5,
        )

        while True:
            self.render(session)
            command = self.console.input("[bold cyan]Command[/bold cyan] (`n`, `p`, `j <idx>`, `k <count>`, `r`, `q`): ").strip()
            if not command or command in {"n", "next"}:
                session.selected_position = min(
                    len(session.prompt_state.input_ids) - 1,
                    session.selected_position + 1,
                )
                continue
            if command in {"p", "prev"}:
                session.selected_position = max(0, session.selected_position - 1)
                continue
            if command in {"q", "quit", "exit"}:
                return
            if command in {"r", "refresh"}:
                continue
            if command.startswith("j "):
                session.selected_position = clamp_int(
                    command[2:].strip(),
                    lower=0,
                    upper=len(session.prompt_state.input_ids) - 1,
                    fallback=session.selected_position,
                )
                continue
            if command.startswith("k "):
                session.top_k = clamp_int(
                    command[2:].strip(),
                    lower=1,
                    upper=20,
                    fallback=session.top_k,
                )
                continue
            self.console.print("[yellow]Unrecognized command.[/yellow]")

    def render(self, session: SessionState) -> None:
        results = analyze_position(
            self.loaded,
            session.prompt_state,
            position=session.selected_position,
            top_k=session.top_k,
        )

        header = self.build_header(session)
        tokens = self.build_token_table(session)
        predictions = self.build_predictions_table(results, session.top_k)

        self.console.clear()
        self.console.print(Group(header, tokens, predictions))

    def build_header(self, session: SessionState) -> Panel:
        current_token = session.prompt_state.token_texts[session.selected_position]
        actual_next = session.prompt_state.next_token_texts[session.selected_position] or "<end>"
        body = Text()
        body.append(f"Selected position: {session.selected_position}\n", style="bold")
        body.append(f"Current token: {current_token}\n")
        body.append(f"Actual next token: {actual_next}\n")
        body.append(f"Top-k: {session.top_k}\n")
        body.append(f"Sequence length: {len(session.prompt_state.input_ids)}")
        return Panel(body, title="Prompt State", border_style="blue")

    def build_token_table(self, session: SessionState) -> Panel:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Pos", justify="right")
        table.add_column("Token")
        table.add_column("Actual Next")
        for index, token_text in enumerate(session.prompt_state.token_texts):
            style = "bold green" if index == session.selected_position else ""
            table.add_row(
                str(index),
                token_text,
                session.prompt_state.next_token_texts[index] or "<end>",
                style=style,
            )
        return Panel(table, title="Tokens", border_style="magenta")

    def build_predictions_table(self, results: list[LayerLensResult], top_k: int) -> Panel:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Layer", justify="right")
        for rank in range(top_k):
            table.add_column(f"Top {rank + 1}")
        for result in results:
            row = [str(result.layer_index)]
            for prediction in result.predictions:
                row.append(f"{prediction.token_text} ({prediction.logit:.2f})")
            table.add_row(*row)
        return Panel(table, title="Layer Predictions", border_style="cyan")


def clamp_int(raw: str, *, lower: int, upper: int, fallback: int) -> int:
    try:
        value = int(raw)
    except ValueError:
        return fallback
    return max(lower, min(upper, value))
