"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import json
import importlib

from typing import Tuple, Any
from enum import Enum, unique
from pydantic import BaseModel, Field
from IPython.display import Markdown
from ..bot_outputs import _B
from ..bot_chat import BotChat
from ..utils import no_indent
from ..bot_agents.base import BaseChatAgent, AgentOutputFormat, AgentModelType


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
