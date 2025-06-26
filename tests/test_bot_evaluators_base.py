import pytest
import types
import importlib
from enum import Enum
from unittest.mock import MagicMock, patch
from jupyter_agent.bot_evaluators.base import BaseEvaluator, EvaluatorFactory


class DummyStage(str, Enum):
    START = "start"
    MIDDLE = "middle"
    END = "completed"


class DummyReply:
    def model_dump_json(self, indent=2):
        return '{"result": "ok"}'


class DummyBaseChatAgent(BaseEvaluator):
    def __call__(self, **kwargs):
        return ["first", "second", "third"]


class DummyNotebookContext:
    def __init__(self):
        self.cur_task = MagicMock()
        self.cur_task.cell_idx = 0
        self.cur_task.agent_stage = DummyStage.START
        self.cur_task.update_cell = MagicMock()
        self.cells = []


def test_on_reply(monkeypatch):
    evaluator = BaseEvaluator(DummyNotebookContext(), base_url="", api_key="", model_name="")
    reply = DummyReply()
    called = {}

    def fake_B(msg, title=None, format=None, code_language=None):
        called["msg"] = msg
        called["title"] = title
        called["format"] = format
        called["code_language"] = code_language

    monkeypatch.setattr("jupyter_agent.bot_evaluators.base._B", fake_B)
    result = evaluator.on_reply(reply)
    assert result is reply
    assert called["msg"] == '{"result": "ok"}'
    assert called["title"] == "Evaluator Reply"
    assert called["format"] == "code"
    assert called["code_language"] == "json"


def test___call___with_super(monkeypatch):
    agent = DummyBaseChatAgent(DummyNotebookContext(), base_url="", api_key="", model_name="")
    result = agent()
    assert result == ["first", "second", "third"]


def test_get_agent_class_with_class():
    class DummyEvaluator(BaseEvaluator):
        pass

    factory = EvaluatorFactory(DummyNotebookContext())
    result = factory.get_agent_class(DummyEvaluator)
    assert result is DummyEvaluator


def test_get_agent_class_with_invalid_class():
    class NotEvaluator:
        pass

    factory = EvaluatorFactory(DummyNotebookContext())
    with pytest.raises(AssertionError):
        factory.get_agent_class(NotEvaluator)


def test_get_agent_class_with_string(monkeypatch):
    class DummyEvaluator(BaseEvaluator):
        pass

    dummy_module = types.SimpleNamespace(DummyEvaluator=DummyEvaluator)
    monkeypatch.setattr(importlib, "import_module", lambda name, package=None: dummy_module)

    factory = EvaluatorFactory(DummyNotebookContext())
    result = factory.get_agent_class("DummyEvaluator")
    assert result is DummyEvaluator
