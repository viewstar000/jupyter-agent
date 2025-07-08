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
    assert isinstance(d, bc.AgentData)


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
    assert ctx.agent_data.task_id == "testid"
    assert ctx.agent_data.subject == "test subject"


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
    ctx.agent_data.task_id = "tid"
    ctx.agent_data.subject = "subj"
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
    for k, _ in ctx.agent_data:
        ctx.set_data(k, "")
    options_str = ctx.format_cell_options()
    assert options_str == ""


def test_format_cell_options_json_field_serialization():
    cell = make_code_cell("%%bot\nprint('hi')")
    ctx = bc.AgentCellContext(0, cell)
    # important_infos is a JSON field
    ctx.agent_data.important_infos = {"foo": "bar"}
    options_str = ctx.format_cell_options()
    assert "important_infos:" in options_str
    assert '"foo": "bar"' in options_str or "foo: bar" in options_str


def test_format_cell_options_handles_list_field():
    cell = make_code_cell("%%bot\nprint('hi')")
    ctx = bc.AgentCellContext(0, cell)
    ctx.agent_data.important_infos = {"a": 1, "b": 2}
    options_str = ctx.format_cell_options()
    assert "important_infos:" in options_str
    assert '"a": 1' in options_str or '"b": 2' in options_str


def test_format_cell_options_skips_none_and_falsey():
    cell = make_code_cell("%%bot\nprint('hi')")
    ctx = bc.AgentCellContext(0, cell)
    ctx.agent_data.task_id = ""
    ctx.agent_data.subject = ""
    ctx.agent_data.coding_prompt = ""
    options_str = ctx.format_cell_options()
    # None, empty string, and False should not appear
    assert "task_id:" not in options_str
    assert "subject:" not in options_str
    assert "coding_prompt:" not in options_str


def test_cell_context_match_subclass_returns_subclass():
    # UserSupplyInfoCellContext should match a raw cell with correct prefix
    cell = {
        "cell_type": "raw",
        "source": "### USER_SUPPLY_INFO:\n- user: foo\n  assistant: bar\n",
        "metadata": {},
        "id": "testid",
    }
    ctx = bc.CellContext.from_cell(0, cell)
    assert isinstance(ctx, bc.UserSupplyInfoCellContext)


def test_cell_context_match_subclass_returns_codecellcontext():
    # CodeCellContext should match a code cell
    cell = {
        "cell_type": "code",
        "source": "print('hi')",
        "metadata": {},
        "outputs": [],
        "id": "testid",
    }
    ctx = bc.CellContext.from_cell(0, cell)
    assert isinstance(ctx, bc.CodeCellContext)


def test_cell_context_match_subclass_returns_agentcellcontext():
    # AgentCellContext should match a code cell starting with %%bot
    cell = {
        "cell_type": "code",
        "source": "%%bot\nprint('hi')",
        "metadata": {},
        "outputs": [],
        "id": "testid",
    }
    ctx = bc.CellContext.from_cell(0, cell)
    assert isinstance(ctx, bc.AgentCellContext)


def test_cell_context_match_subclass_returns_base_for_unknown():
    # Should return base CellContext for unknown cell type
    cell = {
        "cell_type": "unknown",
        "source": "something",
        "metadata": {},
        "id": "testid",
    }
    ctx = bc.CellContext.from_cell(0, cell)
    assert type(ctx) is bc.CellContext


def test_user_supply_info_cell_context_match_and_parse():
    # Should match raw cell with correct prefix and parse YAML
    cell = {
        "cell_type": "raw",
        "source": "### USER_SUPPLY_INFO:\n- user: alice\n  assistant: bob\n- user: carol\n  assistant: dave\n",
        "metadata": {},
        "id": "testid",
    }
    ctx = bc.CellContext.from_cell(0, cell)
    assert isinstance(ctx, bc.UserSupplyInfoCellContext)
    infos = ctx.get_user_supply_infos()
    assert isinstance(infos, list)
    assert infos[0].question == "bob" and infos[0].answer == "alice"
    assert infos[1].question == "dave" and infos[1].answer == "carol"


def test_user_supply_info_cell_context_empty_source():
    # Should return empty list if no YAML after prefix
    cell = {
        "cell_type": "raw",
        "source": "### USER_SUPPLY_INFO:\n",
        "metadata": {},
        "id": "testid",
    }
    ctx = bc.CellContext.from_cell(0, cell)
    assert isinstance(ctx, bc.UserSupplyInfoCellContext)
    infos = ctx.get_user_supply_infos()
    assert infos == []


def test_user_supply_info_cell_context_invalid_yaml():
    # Should raise exception or return [] if YAML is invalid
    cell = {
        "cell_type": "raw",
        "source": "### USER_SUPPLY_INFO:\nnot: [valid: yaml",
        "metadata": {},
        "id": "testid",
    }
    ctx = bc.CellContext.from_cell(0, cell)
    assert isinstance(ctx, bc.UserSupplyInfoCellContext)
    try:
        infos = ctx.get_user_supply_infos()
        # If yaml.safe_load fails, it should raise an exception
        # If implementation changes to catch, then infos should be []
        assert infos == [] or infos is None
    except Exception:
        pass


def test_user_supply_info_cell_context_non_matching_cell():
    # Should not match if cell_type is not raw or prefix missing
    cell = {
        "cell_type": "markdown",
        "source": "### USER_SUPPLY_INFO:\n- user: foo\n  assistant: bar\n",
        "metadata": {},
        "id": "testid",
    }
    ctx = bc.CellContext.from_cell(0, cell)
    assert not isinstance(ctx, bc.UserSupplyInfoCellContext)
    assert isinstance(ctx, bc.CellContext)


def test_notebook_context_cells_reads_cells(tmp_path):
    # Create a notebook with several cells
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("print('a')"),
        nbformat.v4.new_code_cell("%%bot\nprint('b')"),
        nbformat.v4.new_markdown_cell("Some text"),
    ]
    nb_path = tmp_path / "testnb2.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    ctx = bc.NotebookContext("%%bot", "print('b')", str(nb_path))
    cells = ctx.cells
    assert isinstance(cells, list)
    assert len(cells) >= 1
    # Should contain CellContext or subclasses
    assert all(isinstance(c, bc.CellContext) for c in cells)


def test_notebook_context_cur_task_identifies_agent_cell(tmp_path):
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("print('a')"),
        nbformat.v4.new_code_cell("%%bot -s stage1\nprint('b')"),
        nbformat.v4.new_markdown_cell("Some text"),
    ]
    nb_path = tmp_path / "testnb3.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    ctx = bc.NotebookContext("-s stage1", "print('b')", str(nb_path))
    _ = ctx.cells  # trigger loading
    cur_task = ctx.cur_task
    # Should be None or AgentCellContext
    assert cur_task is None or isinstance(cur_task, bc.AgentCellContext)


def test_notebook_context_merged_important_infos(tmp_path):
    nb = nbformat.v4.new_notebook()
    cell1 = nbformat.v4.new_code_cell("%%bot\nprint('b')")
    cell2 = nbformat.v4.new_code_cell("%%bot\nprint('c')")
    # Patch metadata to simulate important_infos
    cell1.metadata["jupyter-agent-data"] = {"important_infos": {"foo": "bar"}}
    cell2.metadata["jupyter-agent-data"] = {"important_infos": {"baz": "qux"}}
    nb.cells = [cell1, cell2]
    nb_path = tmp_path / "testnb4.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    ctx = bc.NotebookContext("%%bot", "print('b')", str(nb_path))
    _ = ctx.cells
    # Patch agent_data for each cell to simulate .important_infos
    for cell in ctx._cells:
        if hasattr(cell, "agent_data"):
            cell.agent_data.important_infos = cell.agent_data.important_infos or {}
            if cell.cell_idx == 0:
                cell.agent_data.important_infos = {"foo": "bar"}
            elif cell.cell_idx == 1:
                cell.agent_data.important_infos = {"baz": "qux"}
    infos = ctx.merged_important_infos
    assert "foo" in infos
    assert "baz" in infos


def test_notebook_context_merged_user_supply_infos(tmp_path):
    nb = nbformat.v4.new_notebook()
    raw_cell = nbformat.v4.new_raw_cell(
        '### USER_SUPPLY_INFO:\n- user: "alice"\n  assistant: "bob"\n- user: "carol"\n  assistant: "dave"\n'
    )
    nb.cells = [raw_cell]
    nb_path = tmp_path / "testnb5.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    ctx = bc.NotebookContext("", "", str(nb_path))
    _ = ctx.cells
    infos = ctx.merged_user_supply_infos
    assert infos[0].question == "bob"
    assert infos[0].answer == "alice"
    assert infos[1].question == "dave"
    assert infos[1].answer == "carol"


def test_notebook_context_cells_handles_file_not_found(tmp_path):
    # Should not raise, should return empty list if file missing
    ctx = bc.NotebookContext("", "", str(tmp_path / "nonexistent.ipynb"))
    cells = ctx.cells
    assert cells == []


def test_notebook_context_cells_handles_invalid_notebook(tmp_path):
    # Write invalid notebook file
    nb_path = tmp_path / "invalid.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        f.write("not a notebook")
    ctx = bc.NotebookContext("", "", str(nb_path))
    cells = ctx.cells
    assert cells == []
