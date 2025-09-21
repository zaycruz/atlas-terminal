# Atlas: Minimal Alpaca Terminal

Atlas provides a focused command‑line experience for Alpaca trading. It ships with
scriptable commands (account, positions, orders, buy/sell, quote) plus an
interactive terminal shell that wraps the same broker layer.

## Requirements
- Python >= 3.13
- Poetry (recommended) or pip
- Dependencies: `alpaca-py`, `rich` (installed automatically by Poetry)
- Environment variables: `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`

## Environment setup
```bash
# Ensure the interpreter matches the project target
poetry env use /opt/homebrew/opt/python@3.13/bin/python3.13
poetry install

# Load credentials via .env (auto-read on startup)
cp .env.example .env
# Edit .env and add ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY
```

You can also export credentials directly:
```bash
export ALPACA_API_KEY_ID=your_key
export ALPACA_API_SECRET_KEY=your_secret
```

## Using the CLI
```bash
# Quick account snapshot (defaults to paper)
atlas account

# Specify live/paper explicitly
atlas --env live positions

# Order management
atlas orders --status closed
atlas buy AAPL 1
atlas sell AAPL 0.5
atlas cancel <order_id>

# Market data (requires market data permissions)
atlas quote AAPL
atlas options AAPL --expiration 2024-09-20
atlas option-order AAPL240920C00200000 1 --side buy --intent buy_to_open --type limit --limit 2.50
```

## Interactive terminal
Launch the REPL experience to explore commands without leaving the shell:
```bash
atlas terminal
```
Commands mirror the CLI (`account`, `positions`, `orders`, `buy`, `sell`,
`cancel`, `quote`). The session greets you with the Atlas banner, keeps a
persistent history in `~/.atlas/history.txt`, and uses color cues via
`termcolor`/Rich to highlight status. Type `help` for a list, `env` to see the
current paper/live mode, and `quit` to exit.

## AI chat mode
```bash
atlas ai
# or override the model
atlas ai --model qwen2.5
```
The assistant connects to a local Ollama instance (`OLLAMA_HOST`, default
`http://localhost:11434`) and uses tools to execute broker actions. Provide
Alpaca credentials as usual; optional variables like `ATLAS_AI_MODEL` and
`ATLAS_AI_SYSTEM_PROMPT` customize behavior. Inside the interactive terminal you
can type `ai` to jump into the same experience.

## Notes
- `--env` (or `$ATLAS_ENV`) controls whether calls hit paper or live trading.
- The broker layer lives in `src/atlas/brokers/` and powers both the CLI and
  terminal.
- Errors are surfaced directly from Alpaca; review them before re-running a
  command.
- See `docs/ai_mode.md` for AI-mode architecture details.

Happy trading!
