import pytest
import types
import tempfile
import os
import nbformat
from jupyter_agent import bot_contexts as bc


class DummyLogger:
    def __call__(self, *args, **kwargs):
        pass


# Patch logging functions to avoid side effects during tests
bc._D = DummyLogger()
bc._I = DummyLogger()
bc._W = DummyLogger()
bc._E = DummyLogger()
bc._F = DummyLogger()
bc._A = DummyLogger()


def make_code_cell(source, outputs=None, metadata=None):
    return {
        "cell_type": "code",
        "source": source,
        "outputs": outputs or [],
        "metadata": metadata or {},
        "id": "testid",
    }


def make_markdown_cell(source, metadata=None):
    return {"cell_type": "markdown", "source": source, "metadata": metadata or {}, "id": "testid"}


def test_cell_context_tags():
    cell = make_code_cell("# BOT_CONTEXT:foo,bar\nprint('hi')", metadata={"tags": ["baz"]})
    ctx = bc.CellContext(0, cell)
    assert "CTX_FOO" in ctx.cell_tags
    assert "CTX_BAR" in ctx.cell_tags
    assert "baz" in ctx.cell_tags
    assert ctx.cell_source == "print('hi')"


def test_cell_context_is_code_context():
    cell = make_code_cell("print('hi')")
    ctx = bc.CellContext(0, cell)
    assert ctx.is_code_context


def test_cell_context_is_task_context():
    cell = make_markdown_cell("Some text", metadata={"tags": ["CTX_TASK"]})
    ctx = bc.CellContext(0, cell)
    assert ctx.is_task_context


def test_code_cell_context_output_truncation():
    cell = make_code_cell("print('hi')")
    ctx = bc.CodeCellContext(0, cell)
    long_output = "a" * (ctx.max_output_size + 10)
    ctx.cell_output = long_output
    out = ctx.cell_output
    assert len(out) <= ctx.max_output_size + 3  # for "..."


def test_code_cell_context_error_truncation():
    cell = make_code_cell("print('hi')")
    ctx = bc.CodeCellContext(0, cell)
    long_error = "e" * (ctx.max_error_size + 10)
    ctx.cell_error = long_error
    err = ctx.cell_error
    assert len(err) <= ctx.max_error_size + 3


def test_code_cell_context_load_cell_outputs_stream():
    outputs = [{"output_type": "stream", "name": "stdout", "text": "hello\n"}]
    cell = make_code_cell("print('hi')", outputs=outputs)
    ctx = bc.CodeCellContext(0, cell)
    ctx.load_cell_outputs(cell)
    assert "stdout" in ctx.cell_output
    assert "hello" in ctx.cell_output


def test_code_cell_context_load_cell_outputs_error():
    outputs = [{"output_type": "error", "ename": "ValueError", "evalue": "bad", "traceback": ["line1", "line2"]}]
    cell = make_code_cell("raise ValueError()", outputs=outputs)
    ctx = bc.CodeCellContext(0, cell)
    ctx.load_cell_outputs(cell)
    assert "ValueError" in ctx.cell_error
    assert "bad" in ctx.cell_error
    assert "Traceback" in ctx.cell_error


def test_agent_data_default():
    d = bc.AgentData.default()
    assert isinstance(d, dict)


def test_agent_cell_context_parse_magic_argv():
    cell = make_code_cell("%%bot -P -f planning -s stage1\nprint('hi')")
    ctx = bc.AgentCellContext(0, cell)
    assert ctx.agent_flow == "planning"
    assert ctx.agent_stage == "stage1"
    assert ctx.cell_type == bc.CellType.PLANNING


def test_agent_cell_context_load_data_from_source_yaml():
    yaml_block = (
        "%%bot\n" "## Task Options:\n" "# task_id: testid\n" "# subject: test subject\n" "## ---\n" "print('hi')"
    )
    cell = make_code_cell(yaml_block)
    ctx = bc.AgentCellContext(0, cell)
    assert ctx._agent_data["task_id"] == "testid"
    assert ctx._agent_data["subject"] == "test subject"


def test_agent_cell_context_format_magic_line():
    cell = make_code_cell("%%bot -s stage1 -f flow1\nprint('hi')")
    ctx = bc.AgentCellContext(0, cell)
    line = ctx.format_magic_line()
    assert "%%bot" in line
    assert "-s" in line
    assert "-f" in line


def test_notebook_context_cells_and_cur_task(tmp_path):
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("print('a')"),
        nbformat.v4.new_code_cell("%%bot\nprint('b')"),
        nbformat.v4.new_markdown_cell("Some text"),
    ]
    nb_path = tmp_path / "testnb.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    ctx = bc.NotebookContext("%%bot", "print('b')", str(nb_path))
    cells = ctx.cells
    assert isinstance(cells, list)
    cur_task = ctx.cur_task
    assert cur_task is None or isinstance(cur_task, bc.AgentCellContext)


def test_format_cell_options_basic_fields():
    cell = make_code_cell("%%bot\nprint('hi')")
    ctx = bc.AgentCellContext(0, cell)
    ctx._agent_data["task_id"] = "tid"
    ctx._agent_data["subject"] = "subj"
    options_str = ctx.format_cell_options()
    assert "task_id: tid" in options_str
    assert "subject: subj" in options_str
    assert "## Task Options:" in options_str
    assert "## ---" in options_str
    assert "update_time" in options_str


def test_format_cell_options_empty_returns_empty_string():
    cell = make_code_cell("%%bot\nprint('hi')")
    ctx = bc.AgentCellContext(0, cell)
    # Clear all fields
    for k in ctx._agent_data:
        ctx._agent_data[k] = ""
    options_str = ctx.format_cell_options()
    assert options_str == ""


def test_format_cell_options_json_field_serialization():
    cell = make_code_cell("%%bot\nprint('hi')")
    ctx = bc.AgentCellContext(0, cell)
    # important_infos is a JSON field
    ctx._agent_data["important_infos"] = {"foo": "bar"}
    options_str = ctx.format_cell_options()
    assert "important_infos:" in options_str
    assert '"foo": "bar"' in options_str or "foo: bar" in options_str


def test_format_cell_options_handles_list_field():
    cell = make_code_cell("%%bot\nprint('hi')")
    ctx = bc.AgentCellContext(0, cell)
    ctx._agent_data["important_infos"] = [{"a": 1}, {"b": 2}]
    options_str = ctx.format_cell_options()
    assert "important_infos:" in options_str
    assert '"a": 1' in options_str or '"b": 2' in options_str


def test_format_cell_options_skips_none_and_falsey():
    cell = make_code_cell("%%bot\nprint('hi')")
    ctx = bc.AgentCellContext(0, cell)
    ctx._agent_data["task_id"] = None
    ctx._agent_data["subject"] = ""
    ctx._agent_data["coding_prompt"] = False
    options_str = ctx.format_cell_options()
    # None, empty string, and False should not appear
    assert "task_id:" not in options_str
    assert "subject:" not in options_str
    assert "coding_prompt:" not in options_str
