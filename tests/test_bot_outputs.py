import pytest
import types
import json
import time

import jupyter_agent.bot_outputs as bot_outputs


class DummyDisplay:
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.updated = False

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self

    def update(self, *args, **kwargs):
        self.updated = True
        self.calls.append(("update", args, kwargs))


@pytest.fixture(autouse=True)
def patch_display(monkeypatch):
    dummy = DummyDisplay()
    monkeypatch.setattr(bot_outputs, "display", dummy)
    yield dummy


def test_agent_output_init_defaults():
    ao = bot_outputs.AgentOutput()
    assert ao.title is None
    assert ao.collapsed is False
    assert ao.logging_level == 20  # INFO
    assert isinstance(ao.jinja_env, type(bot_outputs.jinja2.Environment()))
    assert ao._contents == {}
    assert ao._active_stage is None


def test_agent_output_metadata_and_agent_data():
    ao = bot_outputs.AgentOutput()
    assert ao.metadata["reply_type"] == "AgentOutput"
    ao.output_agent_data(foo="bar")
    meta = ao.metadata
    assert meta["jupyter-agent-data-store"] is True
    assert "jupyter-agent-data-timestamp" in meta
    assert meta["jupyter-agent-data"]["foo"] == "bar"


def test_output_block_and_content_rendering(patch_display):
    ao = bot_outputs.AgentOutput()
    ao.output_block("block content", title="Test Block", collapsed=False, stage="Stage1", format="markdown")
    assert "Test Block" in ao.content
    assert "block content" in ao.content
    assert patch_display.calls  # display should have been called


def test_output_text_merges_same_language(patch_display):
    ao = bot_outputs.AgentOutput()
    ao.output_text("line1", stage="Stage2", code_language="python")
    ao.output_text("line2", stage="Stage2", code_language="python")
    contents = ao._contents["Stage2"]
    assert len(contents) == 1
    assert "line1" in contents[0]["content"]
    assert "line2" in contents[0]["content"]


def test_output_text_different_language(patch_display):
    ao = bot_outputs.AgentOutput()
    ao.output_text("py", stage="Stage3", code_language="python")
    ao.output_text("js", stage="Stage3", code_language="javascript")
    contents = ao._contents["Stage3"]
    assert len(contents) == 2
    assert contents[0]["code_language"] == "python"
    assert contents[1]["code_language"] == "javascript"


def test_output_markdown(patch_display):
    ao = bot_outputs.AgentOutput()
    ao.output_markdown("**bold**", stage="Stage4")
    assert any("**bold**" in str(c["content"]) for c in ao._contents["Stage4"])


def test_clear_and_clear_metadata(patch_display):
    ao = bot_outputs.AgentOutput()
    ao.output_text("abc", stage="Stage5")
    ao.output_agent_data(foo="bar")
    ao.clear(stage="Stage5", clear_metadata=True)
    assert ao._contents["Stage5"] == []
    assert ao._agent_data == {}


def test_log_and_logging_levels(patch_display):
    ao = bot_outputs.AgentOutput(logging_level="DEBUG")
    ao.log("debug msg", level="DEBUG")
    ao.log("info msg", level="INFO")
    ao.log("warn msg", level="WARN")
    logs = ao._logging_records
    assert any("debug msg" in l["content"] for l in logs)
    assert any("info msg" in l["content"] for l in logs)
    assert any("warn msg" in l["content"] for l in logs)
    # Should merge consecutive logs of same level
    ao.log("another debug", level="DEBUG")
    assert "another debug" in logs[-1]["content"]


def test_get_output_and_reset_output(monkeypatch):
    bot_outputs.reset_output()  # reset singleton using public API
    ao1 = bot_outputs.get_output()
    ao2 = bot_outputs.get_output()
    assert ao1 is ao2
    ao3 = bot_outputs.reset_output(title="New", collapsed=True, stage="Stage6", logging_level="ERROR")
    assert ao3 is not ao1
    assert ao3.title == "New"
    assert ao3.collapsed is True
    assert ao3.logging_level == 40  # ERROR


def test_setters_and_stage_switching():
    ao = bot_outputs.reset_output()
    bot_outputs.set_title("MyTitle")
    assert ao.title == "MyTitle"
    bot_outputs.set_collapsed(True)
    assert ao.collapsed is True
    bot_outputs.set_logging_level("WARN")
    assert ao.logging_level == 30
    bot_outputs.set_stage("Stage7")
    assert ao._active_stage == "Stage7"


def test_module_level_helpers(monkeypatch):
    ao = bot_outputs.reset_output()
    bot_outputs.output_block("block", title="T", collapsed=False, stage="S", format="markdown")
    assert ao._contents["S"][0]["title"] == "T"
    bot_outputs.output_text("txt", stage="S2", code_language="python")
    assert ao._contents["S2"][0]["content"] == "txt"
    bot_outputs.output_markdown("md", stage="S3")
    assert ao._contents["S3"][0]["content"] == "md"
    bot_outputs.output_agent_data(foo="bar2")
    assert ao._agent_data["foo"] == "bar2"
    bot_outputs.clear_output(stage="S3", clear_metadata=True)
    assert ao._contents["S3"] == []
    assert ao._agent_data == {}


def test_agent_display_and_aliases(monkeypatch):
    called = {}

    def fake_display(obj, metadata=None, **kwargs):
        called["obj"] = obj
        called["metadata"] = metadata
        called["kwargs"] = kwargs
        return "displayed"

    monkeypatch.setattr(bot_outputs, "display", fake_display)
    res = bot_outputs.agent_display("OBJ", reply_type="RT", exclude_from_context=True, foo="bar")
    assert res == "displayed"
    assert called["metadata"]["reply_type"] == "RT"
    assert called["metadata"]["exclude_from_context"] is True
    # Test _O, _C
    assert bot_outputs._O("OBJ2", reply_type="RT2") == "displayed"
    assert bot_outputs._C("OBJ3", reply_type="RT3") == "displayed"
