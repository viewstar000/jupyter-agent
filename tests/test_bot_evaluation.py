import os
import tempfile
import time
import nbformat
import pytest
import uuid

from pathlib import Path
from unittest.mock import MagicMock, patch
from jupyter_agent import bot_evaluation


class DummyAction(bot_evaluation.ActionBase):
    action: str = "dummy"
    timestamp: float = time.time()


@pytest.fixture
def sample_notebook(tmp_path):
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(source="print('Hello')"))
    nb_path = tmp_path / "test.ipynb"
    with open(nb_path, "w") as f:
        nbformat.write(nb, f)
    return nb_path


def test_notebook_runner_init_paths(sample_notebook, tmp_path):
    runner = bot_evaluation.NotebookRunner(str(sample_notebook))
    assert runner.input_path.exists()
    assert Path(runner.output_path).suffix == ".ipynb"
    assert Path(runner.evaluate_path).suffix == ".jsonl"


def test_save_evaluation_record(tmp_path):
    nb_path = tmp_path / "test.ipynb"
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(source="print('hi')"))
    nbformat.write(nb, nb_path)
    eval_path = tmp_path / "eval.jsonl"
    runner = bot_evaluation.NotebookRunner(str(nb_path), evaluate_path=eval_path)
    record = bot_evaluation.BaseEvaluationRecord(
        timestamp=time.time(),
        notebook_name="test",
        evaluator="tester",
        cell_index=0,
        is_success=True,
        correct_score=1.0,
    )
    runner.save_evaluation_record(record)
    with open(eval_path) as f:
        lines = f.readlines()
    assert len(lines) == 1
    assert '"evaluator":"tester"' in lines[0]


def test_handle_cell_payloads(sample_notebook):
    runner = bot_evaluation.NotebookRunner(str(sample_notebook))
    runner.notebook.cells.append(nbformat.v4.new_code_cell(source="old"))
    payloads = [{"source": "set_next_input", "replace": True, "text": "new"}]
    runner.handle_cell_payloads(1, payloads)
    assert runner.notebook.cells[1].source == "new"


def test_handle_jupyter_agent_data(sample_notebook):
    runner = bot_evaluation.NotebookRunner(str(sample_notebook))
    runner.notebook.cells.append(nbformat.v4.new_code_cell(source="cell"))
    cell_meta = {}
    output_metas = [
        {"jupyter-agent-data-store": True, "jupyter-agent-data-timestamp": 2, "jupyter-agent-data": {"foo": "bar"}}
    ]
    runner.handle_jupyter_agent_data(1, cell_meta, output_metas)
    meta = runner.notebook.cells[1].metadata
    assert meta["jupyter-agent-data-store"]
    assert meta["jupyter-agent-data"]["foo"] == "bar"


def test_handle_evaluation_record_notebook(sample_notebook, tmp_path):
    eval_path = tmp_path / "eval.jsonl"
    runner = bot_evaluation.NotebookRunner(str(sample_notebook), evaluate_path=eval_path)
    output_metas = [
        {
            "jupyter-agent-evaluation-records": [
                {
                    "eval_type": "NOTEBOOK",
                    "timestamp": time.time(),
                    "notebook_name": "test",
                    "evaluator": "bot",
                    "cell_index": 0,
                    "is_success": True,
                    "execution_duration": 1.0,
                    "correct_score": 1.0,
                    "flow_count": 1,
                    "planning_score": 1.0,
                    "coding_score": 1.0,
                    "important_score": 1.0,
                    "user_supply_score": 1.0,
                }
            ]
        }
    ]
    runner.handle_evaluation_record(0, output_metas)
    with open(eval_path) as f:
        lines = f.readlines()
    assert any('"eval_type":"NOTEBOOK"' in line for line in lines)


def test_handle_set_next_cell_replace(sample_notebook):
    runner = bot_evaluation.NotebookRunner(str(sample_notebook))
    params = bot_evaluation.SetCellContentParams(index=0, source="replaced", metadata={}, tags=[], type="code")
    action = bot_evaluation.ActionSetCellContent(params=params)
    idx = runner.handle_set_next_cell(0, action)
    assert runner.notebook.cells[0].source == "replaced"
    assert idx == 0


def test_handle_set_next_cell_insert(sample_notebook):
    runner = bot_evaluation.NotebookRunner(str(sample_notebook))
    params = bot_evaluation.SetCellContentParams(index=1, source="inserted", metadata={}, tags=[], type="markdown")
    action = bot_evaluation.ActionSetCellContent(params=params)
    idx = runner.handle_set_next_cell(0, action)
    assert runner.notebook.cells[1].source == "inserted"
    assert idx == 0


def test_on_cell_executed_triggers_save(tmp_path):
    nb_path = tmp_path / "test.ipynb"
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(source="print('hi')"))
    nbformat.write(nb, nb_path)
    runner = bot_evaluation.NotebookRunner(str(nb_path))
    cell = runner.notebook.cells[0]
    execute_reply = {"content": {"payload": []}}
    with patch("nbformat.write") as mock_write:
        runner.on_cell_executed(0, cell, execute_reply)
        mock_write.assert_called_once()


def test_on_notebook_start_inserts_header(sample_notebook):
    runner = bot_evaluation.NotebookRunner(str(sample_notebook))
    runner.notebook.cells[0].source = "not a header"
    runner.on_notebook_start(runner.notebook)
    assert runner.notebook.cells[0].source.startswith("# -*- Jupyter Agent Evaluation Notebook -*-")


def test_on_notebook_complete_appends_eval(tmp_path):
    nb_path = tmp_path / "test.ipynb"
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(source="print('hi')"))
    nbformat.write(nb, nb_path)
    runner = bot_evaluation.NotebookRunner(str(nb_path))
    runner.is_global_finished = False
    with patch.object(runner, "save_evaluation_record") as mock_save:
        with patch("nbformat.write"):
            runner.on_notebook_complete(runner.notebook)
            mock_save.assert_called_once()
