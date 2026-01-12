"""Agent layer for Atlas."""

from .base import Agent, AgentCapability, AgentRequest, AgentUpdate
from .registry import AgentRegistry, get_agent_registry
from .backtest import BacktestAgent
from .analysis import ProbabilisticAgent, RegressionAgent, ForecastingAgent

__all__ = [
    "Agent",
    "AgentCapability",
    "AgentRequest",
    "AgentUpdate",
    "AgentRegistry",
    "get_agent_registry",
    "BacktestAgent",
    "ProbabilisticAgent",
    "RegressionAgent",
    "ForecastingAgent",
]
