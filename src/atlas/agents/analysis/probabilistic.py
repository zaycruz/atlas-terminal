"""Probabilistic analysis agent - Monte Carlo, Bayesian inference."""

from __future__ import annotations

import json
from typing import Any, AsyncIterable
from uuid import uuid4

from ..base import Agent, AgentCapability, AgentRequest, AgentUpdate
from ...llm import ZAIClient, Message


PROBABILISTIC_PROMPT = """You are a quantitative analyst specializing in probabilistic analysis.

For Monte Carlo simulations, generate Python code that:
1. Simulates many random paths based on historical data
2. Calculates confidence intervals
3. Estimates VaR and CVaR
4. Returns results as JSON

For Bayesian analysis:
1. Define prior distributions
2. Update with observed data
3. Report posterior statistics

Output analysis results as structured JSON."""


class ProbabilisticAgent:
    """Agent for probabilistic analysis - Monte Carlo, Bayesian."""

    def __init__(self, llm: ZAIClient | None = None) -> None:
        self._llm = llm

    def _ensure_initialized(self) -> None:
        if self._llm is None:
            from ...llm import get_llm_client
            self._llm = get_llm_client()

    @property
    def name(self) -> str:
        return "probabilistic"

    @property
    def capabilities(self) -> set[AgentCapability]:
        return {AgentCapability.DATA_ANALYSIS}

    @property
    def description(self) -> str:
        return "Performs probabilistic analysis including Monte Carlo simulations, VaR/CVaR calculations, and Bayesian inference."

    async def execute(self, request: AgentRequest) -> AsyncIterable[AgentUpdate]:
        """Execute probabilistic analysis."""
        self._ensure_initialized()
        job_id = request.request_id

        try:
            yield AgentUpdate(
                agent_name=self.name,
                status="running",
                message="Analyzing request...",
                progress=0.1,
                job_id=job_id,
            )

            # Use LLM to perform analysis
            messages = [
                Message(role="system", content=PROBABILISTIC_PROMPT),
                Message(role="user", content=f"Perform probabilistic analysis: {request.intent}"),
            ]

            yield AgentUpdate(
                agent_name=self.name,
                status="running",
                message="Running analysis...",
                progress=0.5,
                job_id=job_id,
            )

            response = await self._llm.complete(messages, max_tokens=4096)

            yield AgentUpdate(
                agent_name=self.name,
                status="completed",
                message="Analysis complete",
                progress=1.0,
                job_id=job_id,
                artifacts=[
                    {"type": "analysis", "data": response.content},
                ],
            )

        except Exception as e:
            yield AgentUpdate(
                agent_name=self.name,
                status="failed",
                message=f"Analysis failed: {str(e)}",
                error=str(e),
                job_id=job_id,
            )
