"""Steering handlers for agentY Researcher and Brain agents.

These handlers enforce guardrails just-in-time rather than front-loading all
rules into the system prompt, keeping prompts lean and instructions focused.

Usage in agent factories:

    from src.steering import get_brain_steering_handlers, get_researcher_steering_handlers

    plugins = get_brain_steering_handlers()      # list of SteeringHandler instances
    plugins = get_researcher_steering_handlers() # list of SteeringHandler instances
"""

from .brain_handlers import get_brain_steering_handlers
from .researcher_handlers import get_researcher_steering_handlers

__all__ = ["get_brain_steering_handlers", "get_researcher_steering_handlers"]
