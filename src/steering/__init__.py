"""Steering handlers for agentY Query Templates and Assemble Workflow agents.

These handlers enforce guardrails just-in-time rather than front-loading all
rules into the system prompt, keeping prompts lean and instructions focused.

Usage in agent factories:

    from src.steering import get_brain_steering_handlers, get_query_templates_steering_handlers

    plugins = get_brain_steering_handlers()      # list of SteeringHandler instances
    plugins = get_query_templates_steering_handlers() # list of SteeringHandler instances
"""

from .brain_handlers import get_brain_steering_handlers
from .query_templates_handlers import get_query_templates_steering_handlers

__all__ = ["get_brain_steering_handlers", "get_query_templates_steering_handlers"]
