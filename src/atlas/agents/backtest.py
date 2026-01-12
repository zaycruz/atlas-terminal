"""Backtesting agent for strategy execution with native sandboxed executor."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any, AsyncIterable

from .base import Agent, AgentCapability, AgentRequest, AgentUpdate
from ..llm import ZAIClient, Message
from ..execution import SandboxedExecutor, ExecutionConfig
from ..environment import BACKTESTS_DIR


STRATEGY_PARSER_PROMPT = """You are a trading strategy parser. Extract structured parameters from natural language descriptions.

Output a JSON object with these fields:
- symbol: string (ticker symbol, e.g., "AAPL")
- strategy_type: string (e.g., "ema_crossover", "rsi_mean_reversion", "macd", "bollinger_bands")
- parameters: object (strategy-specific params, e.g., {"fast_period": 20, "slow_period": 50})
- start_date: string (YYYY-MM-DD format, default to "2022-01-01" if not specified)
- end_date: string (YYYY-MM-DD format, default to "2024-01-01" if not specified)
- initial_capital: number (default 100000)

Only output valid JSON, nothing else."""


BACKTEST_CODE_PROMPT = """Generate Python backtest code. Output ONLY valid Python code - no markdown, no explanations.

IMPORTANT: When comparing pandas values, use .iloc[-1] or .item() to get scalar values.

Follow this working template exactly, adapting the strategy logic as needed:

import yfinance as yf
import pandas as pd
import numpy as np
import json

# Parameters - REPLACE THESE
symbol = "AAPL"
start_date = "2023-01-01"
end_date = "2023-06-01"
initial_capital = 10000.0
fast_period = 10
slow_period = 20

# Fetch data
df = yf.download(symbol, start=start_date, end=end_date, progress=False)

# Calculate indicators (adapt for strategy type)
df['fast_ema'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
df['slow_ema'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
df = df.dropna()

# Backtest loop
cash = initial_capital
shares = 0
equity_curve = []
trade_log = []

for i in range(1, len(df)):
    date = str(df.index[i].date())
    price = float(df['Close'].iloc[i])
    fast = float(df['fast_ema'].iloc[i])
    slow = float(df['slow_ema'].iloc[i])
    prev_fast = float(df['fast_ema'].iloc[i-1])
    prev_slow = float(df['slow_ema'].iloc[i-1])

    # Buy signal: fast crosses above slow
    if prev_fast <= prev_slow and fast > slow and shares == 0:
        shares = int(cash / price)
        if shares > 0:
            cash -= shares * price
            trade_log.append({"date": date, "side": "buy", "price": price, "shares": shares})

    # Sell signal: fast crosses below slow
    elif prev_fast >= prev_slow and fast < slow and shares > 0:
        cash += shares * price
        trade_log.append({"date": date, "side": "sell", "price": price, "shares": shares})
        shares = 0

    equity = cash + shares * price
    equity_curve.append({"date": date, "equity": round(equity, 2)})

# Calculate metrics
final_equity = equity_curve[-1]["equity"] if equity_curve else initial_capital
total_return = (final_equity - initial_capital) / initial_capital
returns = pd.Series([e["equity"] for e in equity_curve]).pct_change().dropna()
sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0.0
max_dd = float((pd.Series([e["equity"] for e in equity_curve]) / pd.Series([e["equity"] for e in equity_curve]).cummax() - 1).min()) if equity_curve else 0.0

result = {
    "metrics": {
        "total_return": round(total_return, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "total_trades": len([t for t in trade_log if t["side"] == "buy"])
    },
    "equity_curve": equity_curve[-20:],
    "trade_log": trade_log
}
print(json.dumps(result))"""


@dataclass
class StrategySpec:
    """Parsed strategy specification."""
    symbol: str
    strategy_type: str
    parameters: dict[str, Any]
    start_date: str
    end_date: str
    initial_capital: float = 100000.0


@dataclass
class BacktestMetrics:
    """Standard backtest performance metrics."""
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0


@dataclass
class BacktestResult:
    """Complete backtest output."""
    spec: StrategySpec
    metrics: BacktestMetrics
    equity_curve: list[dict[str, Any]]
    trade_log: list[dict[str, Any]]
    code_executed: str


class BacktestAgent:
    """Agent for generating and executing backtests with native sandboxed execution."""

    def __init__(
        self,
        llm: ZAIClient | None = None,
        executor: SandboxedExecutor | None = None,
        timeout: int = 120,
    ) -> None:
        self._llm = llm
        self._executor = executor
        self._timeout = timeout
        self._lazy_init = llm is None or executor is None

    def _ensure_initialized(self) -> None:
        """Lazy initialization of clients."""
        if self._llm is None:
            from ..llm import get_llm_client
            self._llm = get_llm_client()

        if self._executor is None:
            config = ExecutionConfig(timeout=self._timeout)
            self._executor = SandboxedExecutor(config)

    @property
    def name(self) -> str:
        return "backtest"

    @property
    def capabilities(self) -> set[AgentCapability]:
        return {AgentCapability.CODE_EXECUTION, AgentCapability.DATA_ANALYSIS}

    @property
    def description(self) -> str:
        return "Generates and executes trading strategy backtests. Supports EMA crossover, RSI, MACD, Bollinger Bands, and custom strategies."

    async def execute(self, request: AgentRequest) -> AsyncIterable[AgentUpdate]:
        """Execute a backtest request."""
        self._ensure_initialized()

        job_id = request.request_id

        try:
            # Step 1: Parse strategy
            yield AgentUpdate(
                agent_name=self.name,
                status="running",
                message="Parsing strategy...",
                progress=0.1,
                job_id=job_id,
            )

            spec = await self._parse_strategy(request.intent)

            # Step 2: Generate code
            yield AgentUpdate(
                agent_name=self.name,
                status="running",
                message="Generating backtest code...",
                progress=0.3,
                job_id=job_id,
            )

            code = await self._generate_code(spec)

            # Step 3: Execute in sandbox
            yield AgentUpdate(
                agent_name=self.name,
                status="running",
                message=f"Running backtest for {spec.symbol} ({spec.strategy_type})...",
                progress=0.5,
                job_id=job_id,
            )

            exec_result = await self._executor.execute(code)

            # Check for execution errors
            if not exec_result.success:
                raise RuntimeError(
                    f"Code execution failed: {exec_result.error or exec_result.stderr}"
                )

            # Step 4: Parse results
            yield AgentUpdate(
                agent_name=self.name,
                status="running",
                message="Parsing results...",
                progress=0.9,
                job_id=job_id,
            )

            result = self._parse_output(exec_result.stdout, spec, code)

            # Step 5: Save artifacts
            self._save_artifacts(job_id, result)

            # Step 6: Complete
            yield AgentUpdate(
                agent_name=self.name,
                status="completed",
                message=f"Backtest complete: {result.metrics.total_return:.1%} total return, {result.metrics.total_trades} trades",
                progress=1.0,
                job_id=job_id,
                artifacts=[
                    {"type": "metrics", "data": asdict(result.metrics)},
                    {"type": "equity_curve", "data": result.equity_curve[:100]},  # Limit for display
                    {"type": "trade_log", "data": result.trade_log[:50]},
                    {"type": "code", "data": result.code_executed},
                    {"type": "spec", "data": asdict(result.spec)},
                ],
            )

        except Exception as e:
            yield AgentUpdate(
                agent_name=self.name,
                status="failed",
                message=f"Backtest failed: {str(e)}",
                error=str(e),
                job_id=job_id,
            )

    async def _parse_strategy(self, intent: str) -> StrategySpec:
        """Use LLM to extract strategy parameters."""
        messages = [
            Message(role="system", content=STRATEGY_PARSER_PROMPT),
            Message(role="user", content=intent),
        ]

        response = await self._llm.complete(messages, temperature=0.1)

        # Extract JSON from response
        content = response.content.strip()

        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group()

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse strategy: {e}. Response: {content[:200]}")

        return StrategySpec(
            symbol=data.get("symbol", "SPY"),
            strategy_type=data.get("strategy_type", "ema_crossover"),
            parameters=data.get("parameters", {}),
            start_date=data.get("start_date", "2022-01-01"),
            end_date=data.get("end_date", "2024-01-01"),
            initial_capital=data.get("initial_capital", 100000.0),
        )

    async def _generate_code(self, spec: StrategySpec) -> str:
        """Use LLM to generate executable Python code."""
        prompt = f"""Generate a backtest for:
- Symbol: {spec.symbol}
- Strategy: {spec.strategy_type}
- Parameters: {json.dumps(spec.parameters)}
- Date range: {spec.start_date} to {spec.end_date}
- Initial capital: ${spec.initial_capital:,.0f}

Generate complete, runnable Python code."""

        messages = [
            Message(role="system", content=BACKTEST_CODE_PROMPT),
            Message(role="user", content=prompt),
        ]

        response = await self._llm.complete(messages, max_tokens=4096, temperature=0.2)

        # Extract code from response
        code = response.content.strip()

        # Remove markdown code blocks if present
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]

        if code.endswith("```"):
            code = code[:-3]

        return code.strip()

    def _parse_output(self, output: Any, spec: StrategySpec, code: str) -> BacktestResult:
        """Parse the output from the backtest execution."""
        # Convert output to string if needed
        if isinstance(output, list):
            output = "\n".join(str(item) for item in output)
        output_str = str(output)

        # Try to find JSON in output
        json_match = re.search(r'\{.*\}', output_str, re.DOTALL)

        metrics = BacktestMetrics()
        equity_curve: list[dict] = []
        trade_log: list[dict] = []

        if json_match:
            try:
                data = json.loads(json_match.group())

                if "metrics" in data:
                    m = data["metrics"]
                    metrics = BacktestMetrics(
                        total_return=m.get("total_return", 0),
                        cagr=m.get("cagr", 0),
                        sharpe_ratio=m.get("sharpe_ratio", 0),
                        sortino_ratio=m.get("sortino_ratio", 0),
                        max_drawdown=m.get("max_drawdown", 0),
                        win_rate=m.get("win_rate", 0),
                        profit_factor=m.get("profit_factor", 0),
                        total_trades=m.get("total_trades", 0),
                    )

                equity_curve = data.get("equity_curve", [])
                trade_log = data.get("trade_log", [])

            except json.JSONDecodeError:
                pass

        return BacktestResult(
            spec=spec,
            metrics=metrics,
            equity_curve=equity_curve,
            trade_log=trade_log,
            code_executed=code,
        )

    def _save_artifacts(self, job_id: str, result: BacktestResult) -> None:
        """Save backtest artifacts to disk."""
        job_dir = BACKTESTS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Save spec
        with open(job_dir / "spec.json", "w") as f:
            json.dump(asdict(result.spec), f, indent=2)

        # Save metrics
        with open(job_dir / "metrics.json", "w") as f:
            json.dump(asdict(result.metrics), f, indent=2)

        # Save equity curve
        with open(job_dir / "equity_curve.json", "w") as f:
            json.dump(result.equity_curve, f, indent=2)

        # Save trade log
        with open(job_dir / "trade_log.json", "w") as f:
            json.dump(result.trade_log, f, indent=2)

        # Save code
        with open(job_dir / "code.py", "w") as f:
            f.write(result.code_executed)
