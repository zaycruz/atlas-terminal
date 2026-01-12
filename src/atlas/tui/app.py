"""Main Textual application for Atlas AI Research Platform."""

from __future__ import annotations

import asyncio
from pathlib import Path
from datetime import datetime

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Static

from .widgets import ChatWidget, JobStatusWidget, SituationMonitor
from .widgets.monitor import Alert
from .widgets.jobs import JobInfo
from ..orchestrator.events import EventBus, Event, EventType, get_event_bus
from ..orchestrator.router import OrchestrationRouter, get_router
from ..agents import get_agent_registry, BacktestAgent
from ..agents.analysis import ProbabilisticAgent, RegressionAgent, ForecastingAgent
from ..data import get_feed_manager
from ..environment import load_dotenv


STYLES_PATH = Path(__file__).parent / "styles" / "atlas.tcss"


class RecentResearchWidget(Static):
    """Widget showing recent research items."""

    DEFAULT_CSS = """
    RecentResearchWidget {
        background: $surface;
        border: solid $primary;
        padding: 1;
        height: auto;
    }

    RecentResearchWidget .widget-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("RECENT RESEARCH", classes="widget-title")
        yield Static("No research yet", classes="dim")


class AtlasApp(App):
    """Main Textual application for Atlas."""

    TITLE = "Atlas AI Research Platform"
    CSS_PATH = STYLES_PATH

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+k", "command_palette", "Commands", show=False),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+r", "refresh_feeds", "Refresh"),
        Binding("f1", "help", "Help"),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        load_dotenv()

        self._event_bus = get_event_bus()
        self._router: OrchestrationRouter | None = None
        self._feed_manager = get_feed_manager()
        self._setup_event_handlers()
        self._register_agents()

    def _register_agents(self) -> None:
        """Register all available agents."""
        registry = get_agent_registry()
        registry.register(BacktestAgent())
        registry.register(ProbabilisticAgent())
        registry.register(RegressionAgent())
        registry.register(ForecastingAgent())

    def _setup_event_handlers(self) -> None:
        """Subscribe to event bus for reactive updates."""
        self._event_bus.subscribe(EventType.AGENT_UPDATE, self._on_agent_update)
        self._event_bus.subscribe(EventType.ALERT, self._on_alert)
        self._event_bus.subscribe(EventType.LLM_STREAM, self._on_llm_stream)

    async def _on_agent_update(self, event: Event) -> None:
        """Handle agent status updates."""
        payload = event.payload
        jobs_widget = self.query_one("#jobs", JobStatusWidget)

        job_info = JobInfo(
            id=payload.get("job_id", "unknown"),
            name=payload.get("agent_name", "Agent"),
            status=payload.get("status", "running"),
            progress=payload.get("progress"),
            message=payload.get("message", ""),
        )
        jobs_widget.add_job(job_info)

        # Also show status in chat
        chat = self.query_one("#chat", ChatWidget)
        if payload.get("status") == "running":
            chat.add_status(payload.get("message", "Working..."))
        elif payload.get("status") == "failed":
            chat.add_error(payload.get("error", "Unknown error"))

    async def _on_alert(self, event: Event) -> None:
        """Handle new alerts."""
        payload = event.payload
        monitor = self.query_one("#monitor", SituationMonitor)

        alert = Alert(
            id=payload.get("id", ""),
            type=payload.get("type", "alert"),
            title=payload.get("title", ""),
            timestamp=datetime.now(),
            priority=payload.get("priority", 0),
            url=payload.get("url"),
        )
        monitor.add_alert(alert)

    async def _on_llm_stream(self, event: Event) -> None:
        """Handle LLM streaming tokens."""
        chat = self.query_one("#chat", ChatWidget)
        payload = event.payload

        if payload.get("start"):
            chat.start_stream()
        elif payload.get("token"):
            chat.stream_token(payload["token"])
        elif payload.get("end"):
            chat.end_stream()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="chat-area"):
                yield ChatWidget(id="chat")
                yield Input(
                    placeholder="Type your message... (Enter to send)",
                    id="input",
                )
            with Vertical(id="sidebar"):
                yield SituationMonitor(id="monitor")
                yield JobStatusWidget(id="jobs")
                yield RecentResearchWidget(id="recent")
        yield Footer()

    async def on_mount(self) -> None:
        """Called when app is mounted."""
        chat = self.query_one("#chat", ChatWidget)
        chat.add_message(
            "assistant",
            "Welcome to **Atlas AI Research Platform**.\n\n"
            "I can help you with:\n"
            "- **Backtesting**: 'Backtest AAPL with EMA crossover strategy'\n"
            "- **Analysis**: 'Run Monte Carlo simulation on my portfolio'\n"
            "- **Forecasting**: 'Forecast SPY for the next month'\n"
            "- **Research**: 'What are the latest Fed announcements?'\n\n"
            "What would you like to explore today?"
        )

        # Focus the input
        self.query_one("#input", Input).focus()

        # Start feed manager in background
        try:
            await self._feed_manager.start()
            # Initial feed refresh
            asyncio.create_task(self._initial_feed_refresh())
        except Exception:
            pass  # Continue without feeds if they fail

    async def _initial_feed_refresh(self) -> None:
        """Perform initial feed refresh after startup."""
        await asyncio.sleep(2)  # Wait for UI to settle
        try:
            items = await self._feed_manager.fetch_all()
            # Show first few items
            monitor = self.query_one("#monitor", SituationMonitor)
            for item in items[:5]:
                monitor.add_alert(Alert(
                    id=item.id,
                    type=item.feed_type.value,
                    title=item.title,
                    timestamp=item.timestamp,
                    priority=item.priority,
                    url=item.url,
                ))
        except Exception:
            pass

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        user_message = event.value.strip()
        if not user_message:
            return

        # Clear input
        input_widget = self.query_one("#input", Input)
        input_widget.value = ""

        # Add user message to chat
        chat = self.query_one("#chat", ChatWidget)
        chat.add_message("user", user_message)

        # Initialize router lazily (needs LLM client which may need API key)
        if self._router is None:
            try:
                self._router = get_router()
            except ValueError as e:
                chat.add_error(f"LLM not configured: {e}")
                chat.add_message(
                    "system",
                    "Please set the `ZAI_API_KEY` environment variable to use the AI features."
                )
                return

        # Route through orchestrator
        try:
            async for event in self._router.route(user_message):
                # Events are handled by subscriptions, but we can track completion
                pass
        except Exception as e:
            chat.add_error(str(e))

    def action_clear_chat(self) -> None:
        """Clear the chat history."""
        chat = self.query_one("#chat", ChatWidget)
        chat.clear_chat()
        if self._router:
            self._router.clear_history()

    async def action_refresh_feeds(self) -> None:
        """Manually refresh data feeds."""
        chat = self.query_one("#chat", ChatWidget)
        chat.add_status("Refreshing feeds...")
        try:
            items = await self._feed_manager.refresh()
            chat.add_status(f"Refreshed: {len(items)} new items")
        except Exception as e:
            chat.add_error(f"Feed refresh failed: {e}")

    def action_help(self) -> None:
        """Show help."""
        chat = self.query_one("#chat", ChatWidget)
        chat.add_message(
            "system",
            "**Keyboard Shortcuts:**\n\n"
            "- `Ctrl+Q` - Quit\n"
            "- `Ctrl+L` - Clear chat\n"
            "- `Ctrl+R` - Refresh feeds\n"
            "- `F1` - Show this help\n"
            "- `Enter` - Send message\n\n"
            "**Available Agents:**\n\n"
            "- `backtest` - Run trading strategy backtests\n"
            "- `probabilistic` - Monte Carlo, VaR analysis\n"
            "- `regression` - ML regression and prediction\n"
            "- `forecasting` - Time series forecasting\n\n"
            "**Examples:**\n\n"
            "- 'Backtest AAPL with 20/50 EMA crossover from 2022 to 2024'\n"
            "- 'Run Monte Carlo simulation with 1000 iterations'\n"
            "- 'Forecast NVDA stock for the next 30 days'\n"
            "- 'What factors predict SPY returns?'"
        )

    async def on_unmount(self) -> None:
        """Called when app is unmounting."""
        await self._feed_manager.close()


def main() -> None:
    """Entry point for the Atlas TUI."""
    app = AtlasApp()
    app.run()


if __name__ == "__main__":
    main()
