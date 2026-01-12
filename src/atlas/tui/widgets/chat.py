"""Chat widget for conversation display."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rich.markdown import Markdown
from rich.text import Text
from textual.widgets import RichLog


@dataclass
class ChatMessage:
    """A single chat message."""
    role: Literal["user", "assistant", "system"]
    content: str


class ChatWidget(RichLog):
    """Conversation display with markdown rendering and streaming support."""

    DEFAULT_CSS = """
    ChatWidget {
        background: $surface;
        border: solid $primary;
        padding: 1;
    }
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            highlight=True,
            markup=True,
            wrap=True,
            **kwargs,
        )
        self._streaming = False
        self._stream_buffer = ""

    def add_message(self, role: str, content: str) -> None:
        """Add a complete message to the chat."""
        if role == "user":
            prefix = Text("You: ", style="bold cyan")
        elif role == "assistant":
            prefix = Text("Atlas: ", style="bold green")
        else:
            prefix = Text("System: ", style="bold yellow")

        self.write(prefix)
        self.write(Markdown(content))
        self.write("")  # Empty line between messages

    def start_stream(self, role: str = "assistant") -> None:
        """Start a streaming response."""
        self._streaming = True
        self._stream_buffer = ""

        if role == "assistant":
            prefix = Text("Atlas: ", style="bold green")
        else:
            prefix = Text(f"{role}: ", style="bold")

        self.write(prefix)

    def stream_token(self, token: str) -> None:
        """Append a token to the current streaming response."""
        if not self._streaming:
            return

        self._stream_buffer += token
        # For streaming, we write raw text (markdown rendered on end)
        self.write(token, expand=True, scroll_end=True)

    def end_stream(self) -> str:
        """End the streaming response and return the full content."""
        self._streaming = False
        content = self._stream_buffer
        self._stream_buffer = ""
        self.write("")  # Empty line after message
        return content

    def add_status(self, message: str, style: str = "dim") -> None:
        """Add a status message (e.g., 'Thinking...', 'Running backtest...')."""
        self.write(Text(f"  {message}", style=style))

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.write(Text(f"Error: {message}", style="bold red"))

    def clear_chat(self) -> None:
        """Clear all messages."""
        self.clear()
