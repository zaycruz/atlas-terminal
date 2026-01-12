# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Atlas is an AI-native financial research platform with a Textual TUI. It provides natural language interfaces for backtesting, analysis, and market research through specialized agents.

## Development Commands

```bash
# Create virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Run the TUI
atlas
# or
python -m atlas

# Run tests
pytest
```

## Architecture

```
src/atlas/
├── tui/                    # Textual TUI layer
│   ├── app.py              # AtlasApp main application
│   ├── widgets/            # ChatWidget, JobStatusWidget, SituationMonitor
│   └── styles/             # atlas.tcss theme
├── llm/                    # LLM provider layer
│   ├── base.py             # LLMProvider protocol, Message, ToolCall
│   └── client.py           # ZAIClient (OpenAI-compatible)
├── orchestrator/           # Agent coordination
│   ├── events.py           # EventBus pub/sub system
│   └── router.py           # NL intent → agent routing
├── agents/                 # Specialized agents
│   ├── base.py             # Agent protocol, AgentRequest, AgentUpdate
│   ├── registry.py         # AgentRegistry
│   ├── backtest.py         # BacktestAgent (Docker code execution)
│   └── analysis/           # ProbabilisticAgent, RegressionAgent, ForecastingAgent
├── data/                   # Data feeds layer
│   ├── base.py             # DataFeed protocol, FeedItem
│   ├── manager.py          # FeedManager (aggregates all feeds)
│   ├── feeds/              # RSSFeed, MarketDataFeed
│   └── events/             # FedEventMonitor, EconomicCalendar
├── mcp/                    # MCP Docker integration
│   └── docker.py           # FastMCPDockerClient
└── environment.py          # Configuration and paths
```

### Key Components

**TUI Layer** (`tui/app.py`):
- Textual application with ChatWidget, JobStatusWidget, SituationMonitor
- Event bus subscriptions for reactive updates
- Keybindings: Ctrl+Q (quit), Ctrl+L (clear), Ctrl+R (refresh), F1 (help)

**LLM Layer** (`llm/`):
- `ZAIClient`: OpenAI-compatible client for zai.org/GLM-4
- Supports streaming and tool calls
- Configure via `ZAI_API_KEY` and `ATLAS_LLM_BASE_URL`

**Orchestrator** (`orchestrator/`):
- `EventBus`: Pub/sub for AGENT_UPDATE, ALERT, LLM_STREAM events
- `OrchestrationRouter`: Routes natural language to appropriate agents via LLM classification

**Agents** (`agents/`):
- `BacktestAgent`: Generates strategy code via LLM, executes in Docker via MCP
- `ProbabilisticAgent`: Monte Carlo, VaR/CVaR, Bayesian analysis
- `RegressionAgent`: ML regression, feature importance
- `ForecastingAgent`: Time series forecasting

**Data Feeds** (`data/`):
- `RSSFeed`: Financial news from Reuters, Bloomberg, CNBC, etc.
- `MarketDataFeed`: Polygon.io market news and quotes
- `FedEventMonitor`: Federal Reserve announcements
- `EconomicCalendar`: Macro events (NFP, CPI, GDP, etc.)

### Environment Configuration

| Variable | Purpose | Default |
|----------|---------|---------|
| `ZAI_API_KEY` | LLM API key | Required |
| `ATLAS_LLM_BASE_URL` | LLM endpoint | https://api.zai.org/v1 |
| `ATLAS_LLM_MODEL` | Model name | GLM-4.7 |
| `ATLAS_MCP_ENDPOINT` | MCP Docker server | http://localhost:3000 |
| `POLYGON_API_KEY` | Market data | Optional |
| `ATLAS_SEARXNG_ENDPOINT` | Web search | http://localhost:8080/search |

Data stored in `~/.atlas/` (config, data, logs, cache, backtests).

### Event-Driven Architecture

```python
from atlas.orchestrator import get_event_bus, Event, EventType

bus = get_event_bus()
bus.subscribe(EventType.AGENT_UPDATE, async_callback)
await bus.publish(Event(type=EventType.ALERT, payload={...}))
```

### Adding a New Agent

1. Create agent in `src/atlas/agents/`:
```python
class MyAgent:
    name = "myagent"
    capabilities = {AgentCapability.DATA_ANALYSIS}
    description = "Does something useful"

    async def execute(self, request: AgentRequest) -> AsyncIterable[AgentUpdate]:
        yield AgentUpdate(self.name, "running", "Working...")
        # ... do work ...
        yield AgentUpdate(self.name, "completed", "Done", artifacts=[...])
```

2. Register in `tui/app.py`:
```python
registry.register(MyAgent())
```

### BacktestAgent Flow

1. Parse strategy from natural language via LLM → `StrategySpec`
2. Generate Python backtest code via LLM
3. Create Docker container via MCP (`create_container`)
4. Install dependencies (`add_dependencies`)
5. Execute script (`execute_python_script`)
6. Parse JSON results → `BacktestMetrics`, equity curve, trade log
7. Save artifacts to `~/.atlas/data/backtests/<job_id>/`
8. Cleanup container
