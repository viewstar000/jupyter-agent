"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import importlib

from ..bot_outputs import _B
from ..bot_agents.base import BaseChatAgent, AgentOutputFormat, AgentModelType, AgentFactory


class BaseEvaluator(BaseChatAgent):
    """
    Base class for evaluators.
    """

    OUTPUT_FORMAT = AgentOutputFormat.JSON
    MODEL_TYPE = AgentModelType.REASONING
    DISPLAY_REPLY = False

    def on_reply(self, reply):
        _B(reply.model_dump_json(indent=2), title="Evaluator Reply", format="code", code_language="json")
        return reply

    def __call__(self, **kwargs):
        # Ensure BaseChatAgent has a __call__ method, otherwise call a valid method
        result = super().__call__(**kwargs) if hasattr(super(), "__call__") else None
        if result is not None:
            return result[-1]
        raise NotImplementedError("BaseChatAgent does not implement __call__ method.")


class EvaluatorFactory(AgentFactory):

    def get_agent_class(self, agent_class):
        if isinstance(agent_class, str):
            bot_agents = importlib.import_module("..bot_evaluators", __package__)
            agent_class = getattr(bot_agents, agent_class)
        assert issubclass(agent_class, BaseEvaluator), "Unsupported agent class: {}".format(agent_class)
        return agent_class
