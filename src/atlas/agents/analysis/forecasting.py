"""Time series forecasting agent."""

from __future__ import annotations

from typing import Any, AsyncIterable

from ..base import Agent, AgentCapability, AgentRequest, AgentUpdate
from ...llm import ZAIClient, Message


FORECASTING_PROMPT = """You are a time series analyst specializing in financial forecasting.

For forecasting tasks:
1. Analyze the time series characteristics (trend, seasonality, stationarity)
2. Choose appropriate model (ARIMA, exponential smoothing, Prophet, etc.)
3. Generate forecasts with confidence intervals
4. Evaluate forecast accuracy

Generate Python code using statsmodels or similar libraries.
Output forecasts and analysis as structured data."""


class ForecastingAgent:
    """Agent for time series forecasting."""

    def __init__(self, llm: ZAIClient | None = None) -> None:
        self._llm = llm

    def _ensure_initialized(self) -> None:
        if self._llm is None:
            from ...llm import get_llm_client
            self._llm = get_llm_client()

    @property
    def name(self) -> str:
        return "forecasting"

    @property
    def capabilities(self) -> set[AgentCapability]:
        return {AgentCapability.FORECASTING, AgentCapability.DATA_ANALYSIS}

    @property
    def description(self) -> str:
        return "Performs time series analysis and forecasting using ARIMA, exponential smoothing, and other methods."

    async def execute(self, request: AgentRequest) -> AsyncIterable[AgentUpdate]:
        """Execute forecasting analysis."""
        self._ensure_initialized()
        job_id = request.request_id

        try:
            yield AgentUpdate(
                agent_name=self.name,
                status="running",
                message="Analyzing time series...",
                progress=0.1,
                job_id=job_id,
            )

            messages = [
                Message(role="system", content=FORECASTING_PROMPT),
                Message(role="user", content=f"Perform forecasting: {request.intent}"),
            ]

            yield AgentUpdate(
                agent_name=self.name,
                status="running",
                message="Generating forecasts...",
                progress=0.5,
                job_id=job_id,
            )

            response = await self._llm.complete(messages, max_tokens=4096)

            yield AgentUpdate(
                agent_name=self.name,
                status="completed",
                message="Forecasting complete",
                progress=1.0,
                job_id=job_id,
                artifacts=[
                    {"type": "forecast", "data": response.content},
                ],
            )

        except Exception as e:
            yield AgentUpdate(
                agent_name=self.name,
                status="failed",
                message=f"Forecasting failed: {str(e)}",
                error=str(e),
                job_id=job_id,
            )
