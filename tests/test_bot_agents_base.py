import pytest
from unittest.mock import MagicMock, patch
import sys
import types

from jupyter_agent.bot_agents.base import (
    BaseAgent,
    BaseChatAgent,
    AgentFactory,
    AgentOutputFormat,
    AgentCombineReply,
    AgentModelType,
)


class DummyNotebookContext:
    def __init__(self):
        self.cur_task = "dummy_task"
        self.cells = [
            {
                "type": "task",
                "task_id": 1,
                "source": "print(1)",
                "outputs": ["1"],
                "subject": "Test task",
                "context": "",
            }
        ]
        self.merged_important_infos = []
        self.merged_user_supply_infos = []


class DummyBotChat:
    def __init__(self, base_url, api_key, model_name):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.display_message = False

    def create_messages(self, contexts, templates=None):
        class DummyMessages:
            def __init__(self, prompt="prompt"):
                self.prompt = prompt

            def add(self, prompt):
                self.prompt = prompt

            def get(self):
                return [{"role": "user", "content": self.prompt}]

        return DummyMessages()

    def chat(self, messages, display_reply=True):
        # Simulate different reply types for combine tests
        return [
            {"type": "text", "content": "Hello", "raw": "Hello"},
            {"type": "code", "lang": "python", "content": "print(1)", "raw": "print(1)"},
            {"type": "code", "lang": "json", "content": '{"a": 1}', "raw": '{"a": 1}'},
        ]


@pytest.fixture
def notebook_context():
    return DummyNotebookContext()


@pytest.fixture
def base_chat_agent(monkeypatch, notebook_context):
    # Patch BotChat in BaseChatAgent to DummyBotChat
    with patch("jupyter_agent.bot_agents.base.BotChat", DummyBotChat):

        class TestChatAgent(BaseChatAgent):
            PROMPT = "Test prompt"
            OUTPUT_FORMAT = AgentOutputFormat.RAW
            OUTPUT_CODE_LANG = "python"
            OUTPUT_JSON_SCHEMA = None
            DISPLAY_REPLY = False
            COMBINE_REPLY = AgentCombineReply.MERGE
            ACCEPT_EMPYT_REPLY = True
            MODEL_TYPE = AgentModelType.REASONING

            def on_reply(self, reply):
                return reply

        return TestChatAgent(
            notebook_context=notebook_context, base_url="http://test", api_key="key", model_name="model"
        )


def test_base_agent_properties(notebook_context):
    agent = BaseAgent(notebook_context)
    assert agent.task == "dummy_task"
    assert agent.cells == notebook_context.cells


def test_base_chat_agent_prepare_contexts(base_chat_agent):
    ctx = base_chat_agent.prepare_contexts(extra="value")
    assert "cells" in ctx
    assert "task" in ctx
    assert ctx["output_format"] == AgentOutputFormat.RAW
    assert ctx["output_code_lang"] == "python"
    assert ctx["extra"] == "value"


def test_base_chat_agent_create_messages(base_chat_agent):
    contexts = base_chat_agent.prepare_contexts()
    messages = base_chat_agent.create_messages(contexts)
    assert hasattr(messages, "get")
    assert hasattr(messages, "add")


def test_base_chat_agent_combine_raw_replies(base_chat_agent):
    replies = [
        {"raw": "first"},
        {"raw": "second"},
        {"raw": "third"},
    ]
    base_chat_agent.COMBINE_REPLY = AgentCombineReply.FIRST
    assert base_chat_agent.combine_raw_replies(replies) == "first"
    base_chat_agent.COMBINE_REPLY = AgentCombineReply.LAST
    assert base_chat_agent.combine_raw_replies(replies) == "third"
    base_chat_agent.COMBINE_REPLY = AgentCombineReply.MERGE
    assert base_chat_agent.combine_raw_replies(replies) == "firstsecondthird"


def test_base_chat_agent_combine_code_replies(base_chat_agent):
    replies = [
        {"type": "code", "lang": "python", "content": "a=1"},
        {"type": "code", "lang": "python", "content": "b=2"},
    ]
    base_chat_agent.COMBINE_REPLY = AgentCombineReply.FIRST
    assert base_chat_agent.combine_code_replies(replies) == "a=1"
    base_chat_agent.COMBINE_REPLY = AgentCombineReply.LAST
    assert base_chat_agent.combine_code_replies(replies) == "b=2"
    base_chat_agent.COMBINE_REPLY = AgentCombineReply.MERGE
    assert base_chat_agent.combine_code_replies(replies) == "a=1\nb=2"


def test_base_chat_agent_combine_json_replies(base_chat_agent):
    replies = [
        {"type": "code", "lang": "json", "content": '{"a": 1}'},
        {"type": "code", "lang": "json", "content": '{"b": 2}'},
    ]
    base_chat_agent.COMBINE_REPLY = AgentCombineReply.FIRST
    assert base_chat_agent.combine_json_replies(replies) == {"a": 1}
    base_chat_agent.COMBINE_REPLY = AgentCombineReply.LAST
    assert base_chat_agent.combine_json_replies(replies) == {"b": 2}
    base_chat_agent.COMBINE_REPLY = AgentCombineReply.LIST
    assert base_chat_agent.combine_json_replies(replies) == [{"a": 1}, {"b": 2}]
    base_chat_agent.COMBINE_REPLY = AgentCombineReply.MERGE
    assert base_chat_agent.combine_json_replies(replies) == {"a": 1, "b": 2}


def test_base_chat_agent_combine_text_replies(base_chat_agent):
    replies = [
        {"type": "text", "content": "foo"},
        {"type": "text", "content": "bar"},
    ]
    base_chat_agent.COMBINE_REPLY = AgentCombineReply.FIRST
    assert base_chat_agent.combine_text_replies(replies) == "foo"
    base_chat_agent.COMBINE_REPLY = AgentCombineReply.LAST
    assert base_chat_agent.combine_text_replies(replies) == "bar"
    base_chat_agent.COMBINE_REPLY = AgentCombineReply.MERGE
    assert base_chat_agent.combine_text_replies(replies) == "foobar"


def test_base_chat_agent_combine_replies_formats(base_chat_agent):
    replies = [
        {"type": "text", "content": "foo", "raw": "foo"},
        {"type": "code", "lang": "python", "content": "print(1)", "raw": "print(1)"},
        {"type": "code", "lang": "json", "content": '{"a": 1}', "raw": '{"a": 1}'},
    ]
    base_chat_agent.OUTPUT_FORMAT = AgentOutputFormat.RAW
    base_chat_agent.COMBINE_REPLY = AgentCombineReply.MERGE
    assert base_chat_agent.combine_replies(replies) == 'foo\nprint(1){"a": 1}'.replace("\n", "")

    base_chat_agent.OUTPUT_FORMAT = AgentOutputFormat.TEXT
    assert base_chat_agent.combine_replies(replies) == "foo"

    base_chat_agent.OUTPUT_FORMAT = AgentOutputFormat.CODE
    assert base_chat_agent.combine_replies(replies) == "print(1)"

    base_chat_agent.OUTPUT_FORMAT = AgentOutputFormat.JSON
    assert base_chat_agent.combine_replies(replies) == {"a": 1}


def test_base_chat_agent_call(monkeypatch, base_chat_agent):
    # Patch chat to return a single reply
    base_chat_agent.chat = MagicMock(return_value=[{"type": "text", "content": "reply", "raw": "reply"}])
    base_chat_agent.combine_replies = MagicMock(return_value="reply")
    base_chat_agent.on_reply = MagicMock(return_value="reply")
    result = base_chat_agent()
    assert result == (False, "reply")


def test_agent_factory(monkeypatch, notebook_context):
    class DummyAgent(BaseAgent):
        pass

    class DummyChatAgent(BaseChatAgent):
        MODEL_TYPE = AgentModelType.REASONING

        def __init__(self, notebook_context, base_url, api_key, model_name):
            pass

    factory = AgentFactory(notebook_context=notebook_context)
    factory.config_model(AgentModelType.DEFAULT, "http://test", "key", "model")

    # Patch bot_agents to contain DummyChatAgent and DummyAgent
    dummy_module = types.SimpleNamespace()
    dummy_module.DummyChatAgent = DummyChatAgent
    dummy_module.DummyAgent = DummyAgent
    monkeypatch.setitem(sys.modules, "jupyter_agent.bot_agents.bot_agents", dummy_module)
    # monkeypatch.setattr("jupyter_agent.bot_agents.base.bot_agents", dummy_module)

    # Test with class
    agent = factory(DummyAgent)
    assert isinstance(agent, DummyAgent)

    # Test with chat agent class
    agent = factory(DummyChatAgent)
    assert isinstance(agent, DummyChatAgent)

    # Test with string
    # agent = factory("DummyChatAgent")
    # assert isinstance(agent, DummyChatAgent)
