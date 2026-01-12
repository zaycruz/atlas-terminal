"""Regression analysis agent - ML regression, feature importance."""

from __future__ import annotations

from typing import Any, AsyncIterable

from ..base import Agent, AgentCapability, AgentRequest, AgentUpdate
from ...llm import ZAIClient, Message


REGRESSION_PROMPT = """You are a machine learning engineer specializing in regression analysis.

For regression tasks:
1. Identify relevant features
2. Choose appropriate model (linear, polynomial, random forest, etc.)
3. Train and evaluate the model
4. Report feature importance
5. Provide predictions with confidence intervals

Generate Python code using scikit-learn when appropriate.
Output results as structured analysis."""


class RegressionAgent:
    """Agent for ML regression and feature analysis."""

    def __init__(self, llm: ZAIClient | None = None) -> None:
        self._llm = llm

    def _ensure_initialized(self) -> None:
        if self._llm is None:
            from ...llm import get_llm_client
            self._llm = get_llm_client()

    @property
    def name(self) -> str:
        return "regression"

    @property
    def capabilities(self) -> set[AgentCapability]:
        return {AgentCapability.DATA_ANALYSIS, AgentCapability.FORECASTING}

    @property
    def description(self) -> str:
        return "Performs ML regression analysis, feature importance, and predictive modeling using various algorithms."

    async def execute(self, request: AgentRequest) -> AsyncIterable[AgentUpdate]:
        """Execute regression analysis."""
        self._ensure_initialized()
        job_id = request.request_id

        try:
            yield AgentUpdate(
                agent_name=self.name,
                status="running",
                message="Setting up regression model...",
                progress=0.1,
                job_id=job_id,
            )

            messages = [
                Message(role="system", content=REGRESSION_PROMPT),
                Message(role="user", content=f"Perform regression analysis: {request.intent}"),
            ]

            yield AgentUpdate(
                agent_name=self.name,
                status="running",
                message="Training model...",
                progress=0.5,
                job_id=job_id,
            )

            response = await self._llm.complete(messages, max_tokens=4096)

            yield AgentUpdate(
                agent_name=self.name,
                status="completed",
                message="Regression analysis complete",
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
