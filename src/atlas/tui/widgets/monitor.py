"""Situation monitor widget for alerts and news."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static, ListView, ListItem
from textual.containers import Vertical


@dataclass
class Alert:
    """A situation alert or news item."""
    id: str
    type: str  # "news", "fed", "cot", "geo", "market"
    title: str
    timestamp: datetime
    priority: int = 0  # Higher = more important
    url: str | None = None


class SituationMonitor(Static):
    """Real-time situation awareness widget."""

    DEFAULT_CSS = """
    SituationMonitor {
        background: $surface;
        border: solid $primary;
        padding: 1;
        height: auto;
        max-height: 15;
    }

    SituationMonitor .widget-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    SituationMonitor ListView {
        height: auto;
        max-height: 10;
    }

    SituationMonitor ListItem {
        padding: 0 1;
    }
    """

    alerts: reactive[List[Alert]] = reactive([], always_update=True)

    def compose(self):
        yield Static("SITUATION MONITOR", classes="widget-title")
        yield ListView(id="alert-list")

    def watch_alerts(self, alerts: List[Alert]) -> None:
        """React to new alerts."""
        list_view = self.query_one("#alert-list", ListView)
        list_view.clear()

        if not alerts:
            list_view.append(
                ListItem(Static(Text("No alerts", style="dim")))
            )
            return

        # Sort by priority (desc) then timestamp (desc)
        sorted_alerts = sorted(
            alerts,
            key=lambda a: (a.priority, a.timestamp.timestamp()),
            reverse=True,
        )

        for alert in sorted_alerts[:10]:  # Show last 10
            icon = self._get_type_icon(alert.type)
            style = "bold" if alert.priority > 0 else ""

            alert_text = Text()
            alert_text.append(f"{icon} ", style=self._get_type_style(alert.type))
            alert_text.append(alert.title[:50], style=style)

            if len(alert.title) > 50:
                alert_text.append("...", style="dim")

            list_view.append(ListItem(Static(alert_text)))

    def _get_type_icon(self, alert_type: str) -> str:
        return {
            "news": "📰",
            "fed": "🏛️",
            "cot": "📊",
            "geo": "🌍",
            "market": "📈",
            "alert": "⚠️",
        }.get(alert_type, "•")

    def _get_type_style(self, alert_type: str) -> str:
        return {
            "news": "blue",
            "fed": "yellow",
            "cot": "cyan",
            "geo": "magenta",
            "market": "green",
            "alert": "red",
        }.get(alert_type, "")

    def add_alert(self, alert: Alert) -> None:
        """Add a new alert."""
        current = list(self.alerts)
        current.append(alert)
        # Keep only last 50 alerts
        self.alerts = current[-50:]

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts = []
