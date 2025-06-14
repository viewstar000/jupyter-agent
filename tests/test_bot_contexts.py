import pytest
from jupyter_agent.bot_contexts import TaskCellContext
from jupyter_agent.bot_contexts import TaskCellContext, UserPromptResponse
import yaml


class DummyDebugMixin:
    def __init__(self, debug_level=0):
        pass


def test_format_options_basic():
    ctx = TaskCellContext(cur_line="", cur_content="", notebook_path=None)
    cell_options = {
        "id": "ABC123",
        "subject": "Test Subject",
        "coding_prompt": "Write code",
        "verify_prompt": "Verify code",
        "summary_prompt": "Summarize",
        "result": "Success",
        "issues": "None",
        "important_infos": {"info": "value"},
        "supply_infos": [UserPromptResponse(prompt="foo", response="bar")],
        "confirm_infos": [],
    }
    formatted = ctx.format_options(cell_options)
    assert "## Task Options:" in formatted
    assert "# id: ABC123" in formatted
    assert "# subject: Test Subject" in formatted
    assert "# coding_prompt: Write code" in formatted
    assert "# verify_prompt: Verify code" in formatted
    assert "# summary_prompt: Summarize" in formatted
    assert "# result: Success" in formatted
    assert "# issues: None" in formatted
    assert "# important_infos:" in formatted
    assert "## ---" in formatted
    ctx = TaskCellContext(cur_line="", cur_content=formatted, notebook_path=None)
    assert ctx.task_id == "ABC123"
    assert ctx.task_subject == "Test Subject"
    assert ctx.task_coding_prompt == "Write code"
    assert ctx.task_verify_prompt == "Verify code"
    assert ctx.task_summary_prompt == "Summarize"
    assert ctx.task_result == "Success"
    assert ctx.task_issue == "None"
    assert ctx.task_important_infos == {"info": "value"}
    assert ctx.task_supply_infos == [UserPromptResponse(prompt="foo", response="bar")]
    assert ctx.task_confirm_infos == []


def check_yaml_loaded(formatted):

    yaml_str = ""
    for line in formatted.split("\n"):
        if line.startswith("# "):
            yaml_str += line[2:] + "\n"
    return yaml.safe_load(yaml_str)


def test_format_options_multiline_string():
    ctx = TaskCellContext(cur_line="", cur_content="", notebook_path=None)
    cell_options = {"description": "Line1\nLine2\nLine3"}
    formatted = ctx.format_options(cell_options)
    assert "# description: |-" in formatted
    assert "#     Line1" in formatted
    assert "#     Line2" in formatted
    assert "#     Line3" in formatted
    loaded = check_yaml_loaded(formatted)
    assert loaded["description"] == "Line1\nLine2\nLine3"


def test_format_options_nested_dict_and_list():
    ctx = TaskCellContext(cur_line="", cur_content="", notebook_path=None)
    cell_options = {"nested": {"a": 1, "b": [2, 3, {"c": 4}]}}
    formatted = ctx.format_options(cell_options)
    assert "# nested:" in formatted
    assert "#     a: 1" in formatted
    assert "#     b:" in formatted
    assert "#         - 2" in formatted
    assert "#         - 3" in formatted
    assert "#         - \n#             c: 4" in formatted
    loaded = check_yaml_loaded(formatted)
    assert loaded["nested"] == {"a": 1, "b": [2, 3, {"c": 4}]}


def test_format_options_with_special_characters():
    ctx = TaskCellContext(cur_line="", cur_content="", notebook_path=None)
    cell_options = {"special": "value:with:colon \"and quotes\" and 'single quotes'"}
    formatted = ctx.format_options(cell_options)
    assert "# special: 'value:with:colon \"and quotes\" and ''single quotes'''" in formatted
    loaded = check_yaml_loaded(formatted)
    assert loaded["special"] == "value:with:colon \"and quotes\" and 'single quotes'"


def test_format_options_empty_dict():
    ctx = TaskCellContext(cur_line="", cur_content="", notebook_path=None)
    cell_options = {}
    formatted = ctx.format_options(cell_options)
    assert "## Task Options:" in formatted
    assert "## ---" in formatted
    loaded = check_yaml_loaded(formatted)
    assert loaded == None


def make_cell_content_with_options(options_dict, code="print('hello')"):
    yaml_str = yaml.safe_dump(options_dict, allow_unicode=True, sort_keys=False).strip()
    lines = ["## Task Options:"]
    for line in yaml_str.splitlines():
        lines.append(f"# {line}")
    lines.append("## ---")
    lines.append(code)
    return "\n".join(lines)


def test_parse_bot_cell_basic_options():
    options = {
        "id": "XYZ789",
        "subject": "Parse Test",
        "coding_prompt": "Do something",
        "verify_prompt": "Check it",
        "summary_prompt": "Summarize it",
        "result": "OK",
        "issues": "None",
        "important_infos": {"foo": "bar"},
        "supply_infos": [],
        "confirm_infos": [],
    }
    content = make_cell_content_with_options(options)
    ctx = TaskCellContext(cur_line="", cur_content=content, notebook_path=None)
    ctx.parse_bot_cell()
    assert ctx.task_id == "XYZ789"
    assert ctx.task_subject == "Parse Test"
    assert ctx.task_coding_prompt == "Do something"
    assert ctx.task_verify_prompt == "Check it"
    assert ctx.task_summary_prompt == "Summarize it"
    assert ctx.task_result == "OK"
    assert ctx.task_issue == "None"
    assert ctx.task_important_infos == {"foo": "bar"}
    assert ctx.task_supply_infos == []
    assert ctx.task_confirm_infos == []
    assert ctx.cell_code == "print('hello')"


def test_parse_bot_cell_with_stage_in_line():
    options = {"id": "STAGE1"}
    content = make_cell_content_with_options(options)
    ctx = TaskCellContext(cur_line="-s teststage", cur_content=content, notebook_path=None)
    ctx.parse_bot_cell()
    assert ctx.task_stage == "teststage"
    assert ctx.task_id == "STAGE1"


def test_parse_bot_cell_with_stage_in_options_only():
    options = {"id": "STAGE2", "stage": "stage_from_options"}
    content = make_cell_content_with_options(options)
    ctx = TaskCellContext(cur_line="", cur_content=content, notebook_path=None)
    ctx.parse_bot_cell()
    assert ctx.task_stage == "stage_from_options"
    assert ctx.task_id == "STAGE2"


def test_parse_bot_cell_with_supply_and_confirm_infos():
    options = {
        "supply_infos": [{"prompt": "foo", "response": "bar"}],
        "confirm_infos": [{"prompt": "baz", "response": "qux"}],
    }
    content = make_cell_content_with_options(options)
    ctx = TaskCellContext(cur_line="", cur_content=content, notebook_path=None)
    ctx.parse_bot_cell()
    assert isinstance(ctx.task_supply_infos, list)
    assert isinstance(ctx.task_confirm_infos, list)
    assert isinstance(ctx.task_supply_infos[0], UserPromptResponse)
    assert ctx.task_supply_infos[0].prompt == "foo"
    assert ctx.task_supply_infos[0].response == "bar"
    assert ctx.task_confirm_infos[0].prompt == "baz"
    assert ctx.task_confirm_infos[0].response == "qux"


def test_parse_bot_cell_without_options():
    code = "print('no options')"
    ctx = TaskCellContext(cur_line="", cur_content=code, notebook_path=None)
    ctx.parse_bot_cell()
    assert ctx.cell_code == "print('no options')"
    # All task_* fields should be default
    assert ctx.task_id
    assert ctx.task_subject == ""
    assert ctx.task_coding_prompt == ""
    assert ctx.task_verify_prompt == ""
    assert ctx.task_summary_prompt == ""
    assert ctx.task_result == ""
    assert ctx.task_issue == ""
    assert ctx.task_important_infos == {}
    assert ctx.task_supply_infos == []
    assert ctx.task_confirm_infos == []


def test_parse_bot_cell_stops_options_on_non_comment():
    # Should stop parsing options if a non-# line appears in options section
    content = "## Task Options:\n# id: STOP\nnot_a_comment\n## ---\nprint('done')"
    ctx = TaskCellContext(cur_line="", cur_content=content, notebook_path=None)
    ctx.parse_bot_cell()
    assert ctx.task_id == "STOP"
    assert "not_a_comment" in ctx.cell_code
