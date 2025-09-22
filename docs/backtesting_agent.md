# Backtesting Agent Integration Plan

## 1. docker_mcp capabilities snapshot
- **Server role**: Model Context Protocol (MCP) server exposing Docker operations to LLM agents.
- **Runtime isolation**: Each request can spin up a container (any image, e.g. `python:3.11-slim`).
- **Available tools**
  - `list_containers(show_all: bool = True)` – inspect running/stopped containers.
  - `create_container(image, container_name, dependencies=...)` – start container, optional package install.
  - `add_dependencies(container_name, dependencies)` – add packages mid-session (pip/npm/apt/apk based on image).
  - `execute_code(container_name, command)` – run shell commands.
  - `execute_python_script(container_name, script_content, script_args=None)` – send multi-line Python for backtests.
  - `cleanup_container(container_name)` – stop/remove container.
- **Workflow expectations**: start server via `python run_server.py`, connect through MCP client, then issue tool calls.

## 2. Proposed Atlas architecture
```
conversation_agent  ──(request backtest)──► backtesting_dispatch
                                          ▼
                           async BacktestingAgent (MCP client)
                                │   1. ensure container
                                │   2. install deps
                                │   3. generate/run script
                                │   4. stream logs/artifacts
                                ▼
                         result envelope → conversation_agent
```

### Responsibilities
- **Conversation agent**
  - Detect backtesting intent.
  - Enqueue job, notify user (“running backtest…”).
  - Await completion event, surface summary + artifacts.
- **Backtesting agent**
  - Translate user strategy/parameters into runnable Python (prompted via LLM or curated templates).
  - Interact with MCP server for container lifecycle + execution.
  - Stream interim status (e.g. install logs) and final results (metrics, equity curve paths).
  - Tear down container (unless flagged to persist).

## 3. Data contracts
- **Job request**
  ```json
  {
    "strategy": "ema_crossover",
    "symbols": ["AAPL"],
    "timeframe": "1d",
    "from": "2022-01-01",
    "to": "2024-01-01",
    "notes": "Risk-free rate 0.02"
  }
  ```
- **Result envelope**
  ```json
  {
    "status": "completed",
    "summary": "EMA(20/50) on AAPL beat buy-and-hold by 3.2%",
    "metrics": {
      "cagr": 0.12,
      "max_drawdown": -0.08,
      "sharpe": 1.1
    },
    "artifacts": [
      {"type": "plot", "path": "atlas_data/backtests/ema_aapl_equity.png"},
      {"type": "log", "content": "hydra installed numpy pandas\n..."}
    ],
    "container": {"name": "atlas-backtest-aapl", "image": "python:3.11"}
  }
  ```

## 4. Implementation steps
1. **MCP client scaffold**
   - Small wrapper in `src/atlas/mcp/docker.py` with methods mirroring docker_mcp tools.
   - Config via env (`ATLAS_MCP_URL`, `ATLAS_MCP_TOKEN`, default image `ATLAS_BACKTEST_IMAGE`).
2. **Backtesting agent module**
   - Async worker (e.g. `src/atlas/agents/backtester.py`) with queue or task manager.
   - Accepts structured job payload; uses MCP client to run scripts.
   - Emits progress callbacks (log stream, completion).
3. **Conversation agent updates**
   - Detect backtest intent in `atlas.ai.chat` (or separate routing layer).
   - Provide status updates (“job queued/running/finished”).
   - On completion, render summary + attach artifacts (images/logs) in chat/terminal.
4. **Persistence & cleanup**
   - Store artifacts under `APP_DIR/data/backtests/<job_id>/`.
   - Ensure `cleanup_container` runs on success/failure unless `persist: true`.
5. **Testing**
   - Unit: mock MCP responses, ensure retry/backoff.
   - Integration: with local Docker, run sample backtest script to validate flow.

## 5. Open questions / TODOs
- Do we want the backtesting agent to generate code via LLM or pull from templates?
- How do we handle long-running jobs (progress polling vs. push)?
- Should we support multiple concurrent backtests or queue them serially?
- Artifact types beyond plots (CSV exports, JSON strategy configs)?

This document will be the reference while we implement the MCP-powered backtesting workflow.

## 6. Configuration hooks
- `ATLAS_MCP_ENDPOINT` (default `http://localhost:3000`)
- `ATLAS_MCP_TOKEN` (optional auth token)
- `ATLAS_BACKTEST_IMAGE` (default `python:3.11-slim`)


## 7. Search backend
- Atlas uses a self-hosted SearxNG instance for web research.
- Run `docker run -p 8080:8080 searxng/searxng:latest` and expose JSON output.
- Configure via `ATLAS_SEARXNG_ENDPOINT` / `ATLAS_SEARXNG_CATEGORIES`.

- When running SearxNG via Docker, disable bot detection to allow the tool to call it: 
  `docker run -d -e FILTER_HEADERS=false -e BOTDETECTION_ENABLED=false -p 8080:8080 searxng/searxng:latest`


When deploying SearxNG, make sure the JSON API is enabled and bot detection is disabled so the agent can query it:

```
server:
  bind_address: 0.0.0.0
  port: 8080
  filter_headers: false
  method: "GET"

botdetection:
  enabled: false

search:
  formats:
    - html
    - json
```
