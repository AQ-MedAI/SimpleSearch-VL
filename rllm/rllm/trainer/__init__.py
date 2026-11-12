"""Trainer module for rLLM.

This module contains the training infrastructure for RL training of language agents.
"""

from .env_agent_mappings import *

__all__ = ["AgentTrainer"]


def __getattr__(name):
    if name == "AgentTrainer":
        from .agent_trainer import AgentTrainer

        return AgentTrainer
    raise AttributeError(name)
