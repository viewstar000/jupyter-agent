"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import os
import time
import json
import argparse
import nbformat

from pathlib import Path
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field
from nbclient.client import NotebookClient
from .bot_actions import ActionBase, ActionSetNextCell, SetNextCellParams, get_action_class


class BaseEvaluationRecord(BaseModel):
    timestamp: float = 0
    notebook_name: str = ""
    evaluator: str = ""
    eval_type: str = "BASE"
    cell_index: int = -1
    execution_duration: float = 0.0
    is_success: bool = False
    correct_score: float = 0.0


class StageEvaluationRecord(BaseEvaluationRecord):
    eval_type: str = "STAGE"
    flow: str = ""
    stage: str = ""
    agent: str = ""


class FlowEvaluationRecord(BaseEvaluationRecord):
    eval_type: str = "FLOW"
    flow: str = ""
    stage_count: int = 0
    planning_score: float = 0.0
    reasoning_score: float = 0.0
    coding_score: float = 0.0
    important_score: float = 0.0
    user_supply_score: float = 0.0


class NotebookEvaluationRecord(BaseEvaluationRecord):
    eval_type: str = "NOTEBOOK"
    flow_count: int = 0
    planning_score: float = 0.0
    coding_score: float = 0.0
    important_score: float = 0.0
    user_supply_score: float = 0.0


class NotebookRunner:

    def __init__(
        self,
        input_path: str | Path,
        output_path: str | Path = "",
        evaluate_path: str | Path = "",
        timeout: int = -1,
        startup_timeout: int = 60,
        allow_errors: bool = False,
        skip_cells_with_tag: str = "skip-execution",
        **kwargs,
    ):
        self.input_path = Path(input_path).with_suffix(".ipynb")
        self.output_path = output_path
        self.evaluate_path = evaluate_path
        self.start_time = 0
        self.is_global_finished = False

        if not self.output_path:
            self.output_path = self.input_path.parent.joinpath(f"{self.input_path.with_suffix('').name}_eval.ipynb")
        if not self.evaluate_path:
            self.evaluate_path = self.input_path.parent.joinpath(f"{self.input_path.with_suffix('').name}_eval.jsonl")
        self.output_path = Path(self.output_path).absolute()
        self.evaluate_path = Path(self.evaluate_path).absolute()

        with self.input_path.open() as f:
            print("Opening notebook:", input_path)
            self.notebook = nbformat.read(f, as_version=4)

        self.client = NotebookClient(
            self.notebook,
            timeout=timeout,
            startup_timeout=startup_timeout,
            skip_cells_with_tag=skip_cells_with_tag,
            allow_errors=allow_errors,
            resources={"metadata": {"path": self.input_path.parent.absolute()}},
            on_notebook_start=self.on_notebook_start,
            on_notebook_complete=self.on_notebook_complete,
            on_cell_executed=self.on_cell_executed,
            **kwargs,
        )

    def save_evaluation_record(self, record: BaseEvaluationRecord):

        if isinstance(record, FlowEvaluationRecord):
            eval_source = record.flow + "-" + record.evaluator
        elif isinstance(record, StageEvaluationRecord):
            eval_source = record.flow + "-" + record.stage + "-" + record.agent + "-" + record.evaluator
        else:
            eval_source = record.evaluator
        print(
            f"CELL[{record.cell_index}] Evaluation: {record.eval_type}[{eval_source}] "
            f"{'SUCCESS' if record.is_success else 'FAILURE'} "
            f"duration: {record.execution_duration:.2f}s "
            f"correct: {record.correct_score:.2f}"
        )
        if self.evaluate_path:
            with open(self.evaluate_path, "a") as eval_file:
                eval_file.write(record.model_dump_json() + "\n")

    def handle_cell_payloads(self, cell_index, cell_payloads):
        for payload in cell_payloads:
            if payload.get("source") == "set_next_input" and payload.get("replace") is True:
                print(f"CELL[{cell_index}] Replacing cell with set_next_input payload")
                self.notebook.cells[cell_index].source = payload.get("text", "")

    def handle_jupyter_agent_data(self, cell_index, cell_meta, cell_output_metas):
        cell_agent_data_timestamp = cell_meta.get("jupyter-agent-data-timestamp", 0)
        output_agent_data_timestamp = cell_agent_data_timestamp
        for output_meta in cell_output_metas:
            if (
                output_meta.get("jupyter-agent-data-store")
                and output_meta.get("jupyter-agent-data-timestamp", 0) > cell_agent_data_timestamp
                and output_meta.get("jupyter-agent-data", {})
            ):
                print(f"CELL[{cell_index}] Found jupyter-agent-data-store outputs, save it to cell metadata")
                output_agent_data_timestamp = max(
                    output_agent_data_timestamp,
                    output_meta.get("jupyter-agent-data-timestamp", 0),
                )
                self.notebook.cells[cell_index].metadata["jupyter-agent-data-store"] = True
                self.notebook.cells[cell_index].metadata["jupyter-agent-data-timestamp"] = output_agent_data_timestamp
                if "jupyter-agent-data" not in self.notebook.cells[cell_index].metadata:
                    self.notebook.cells[cell_index].metadata["jupyter-agent-data"] = {}
                self.notebook.cells[cell_index].metadata["jupyter-agent-data"].update(
                    output_meta["jupyter-agent-data"]
                )

    def handle_evaluation_record(self, cell_index, cell_output_metas):
        is_bot_cell = False
        is_flow_completed = False
        for output_meta in cell_output_metas:
            for record in output_meta.get("jupyter-agent-evaluation-records", []):
                is_bot_cell = True
                if record["eval_type"] == "NOTEBOOK":
                    record = NotebookEvaluationRecord(**record)
                    record.timestamp = record.timestamp or time.time()
                    record.notebook_name = str(self.output_path)
                    record.execution_duration = time.time() - self.start_time
                    self.is_global_finished = True
                    is_flow_completed = True
                    del self.notebook.cells[cell_index + 1 :]  # Remove all cells after the notebook cell
                elif record["eval_type"] == "FLOW":
                    record = FlowEvaluationRecord(**record)
                    record.timestamp = record.timestamp or time.time()
                    record.notebook_name = str(self.output_path)
                    is_flow_completed = True
                elif record["eval_type"] == "STAGE":
                    record = StageEvaluationRecord(**record)
                    record.timestamp = record.timestamp or time.time()
                    record.notebook_name = str(self.output_path)
                else:
                    record = BaseEvaluationRecord(**record)
                    record.timestamp = record.timestamp or time.time()
                    record.notebook_name = str(self.output_path)
                self.save_evaluation_record(record)
        if is_bot_cell and not is_flow_completed:
            self.save_evaluation_record(
                FlowEvaluationRecord(
                    timestamp=time.time(),
                    notebook_name=str(self.output_path),
                    evaluator="bot",
                    eval_type="FLOW",
                    cell_index=cell_index,
                    is_success=False,
                )
            )

    def handle_set_next_cell(self, cell_index, action):
        if action.params.index == 0:
            self.notebook.cells[cell_index].source = action.params.source
            self.notebook.cells[cell_index].metadata.update(action.params.metadata)
            self.notebook.cells[cell_index].metadata["tags"] = action.params.tags
            print(f"CELL[{cell_index}] Replacing cell with set_next_cell action")
            return cell_index
        else:
            metadata = dict(action.params.metadata)
            metadata["tags"] = action.params.tags
            if action.params.type == "code":
                new_cell = nbformat.v4.new_code_cell(source=action.params.source, metadata=metadata)
            elif action.params.type == "markdown":
                new_cell = nbformat.v4.new_markdown_cell(source=action.params.source, metadata=metadata)
            else:
                raise ValueError(f"Unsupported cell type: {action.params.type}")
            insert_idx = cell_index if action.params.index == -1 else cell_index + action.params.index
            ret_idx = cell_index + 1 if action.params.index == -1 else cell_index
            self.notebook.cells.insert(insert_idx, new_cell)
            print(f"CELL[{cell_index}] Inserting  cell at [{insert_idx}] with set_next_cell action")
            return ret_idx

    def handle_jupyter_agent_actions(self, cell_index, cell_meta, cell_output_metas):
        cell_action_timestamp = cell_meta.get("jupyter-agent-action-timestamp", 0)
        output_action_timestamp = cell_action_timestamp
        for output_meta in cell_output_metas:
            for action in output_meta.get("jupyter-agent-action-records", []):
                action = get_action_class(action["action"])(**action)
                if action.timestamp > cell_action_timestamp:
                    output_action_timestamp = max(action.timestamp, output_action_timestamp)
                    if isinstance(action, ActionSetNextCell):
                        print(f"CELL[{cell_index}] Action: {action.action} - {action.source} - {action.timestamp}")
                        cell_index = self.handle_set_next_cell(cell_index, action)
        self.notebook.cells[cell_index].metadata["jupyter-agent-action-timestamp"] = output_action_timestamp

    def on_cell_executed(self, cell_index, cell, execute_reply):
        cell_id = cell.get("id")
        cell_type = cell.get("cell_type")
        cell_meta = cell.get("metadata", {})
        cell_outputs = cell.get("outputs", [])
        cell_payloads = execute_reply.get("content", {}).get("payload", [])
        cell_output_metas = [
            output["metadata"]
            for output in cell_outputs
            if output.get("output_type") == "display_data" and output.get("metadata")
        ]
        self.handle_cell_payloads(cell_index, cell_payloads)
        self.handle_jupyter_agent_data(cell_index, cell_meta, cell_output_metas)
        self.handle_evaluation_record(cell_index, cell_output_metas)
        self.handle_jupyter_agent_actions(cell_index, cell_meta, cell_output_metas)
        print(f"CELL[{cell_index}] Saving executed {cell_type} cell - {cell_id}")
        nbformat.write(self.notebook, self.output_path)

    def on_notebook_start(self, notebook):
        print("Notebook execution started.")
        self.start_time = time.time()
        if not self.notebook.cells[0].source.startswith("# -*- Jupyter Agent Evaluation Notebook -*-"):
            self.notebook.cells.insert(
                0,
                nbformat.v4.new_code_cell(
                    source=(
                        f"# -*- Jupyter Agent Evaluation Notebook -*-\n"
                        f"# Executed notebook: {self.input_path}\n"
                        f"# Output saved to: {self.output_path}\n\n"
                        f"__evaluation_ipynb_file__ = '{self.output_path}'\n"
                    ),
                    metadata={"tags": ["CTX_EXCLUDE"]},
                ),
            )

    def on_notebook_complete(self, notebook):
        print("Notebook execution completed.")
        # If the notebook did not finish globally, append an evaluation record
        if not self.is_global_finished:
            print("Notebook execution did not finish globally, appending evaluation records.")
            self.save_evaluation_record(
                NotebookEvaluationRecord(
                    notebook_name=str(self.output_path),
                    timestamp=time.time(),
                    evaluator="bot",
                    eval_type="NOTEBOOK",
                    execution_duration=time.time() - self.start_time,
                    is_success=False,
                )
            )
        print(f"Saving executed notebook to: {self.output_path}")
        nbformat.write(self.notebook, self.output_path)

    def run(self):

        self.client.execute()


def main():
    """Main function to run the notebook execution."""
    parser = argparse.ArgumentParser(description="Run a Jupyter notebook.")
    parser.add_argument(
        "-o", "--output_path", type=str, default="", help="Path to save the executed notebook (default: same as input)"
    )
    parser.add_argument(
        "-e", "--evaluate_path", type=str, default="", help="Path to save evaluate records (default: same as input)"
    )
    parser.add_argument(
        "--timeout", type=int, default=-1, help="Execution timeout in seconds (default: -1, no timeout)"
    )
    parser.add_argument(
        "--startup_timeout", type=int, default=60, help="Kernel startup timeout in seconds (default: 60)"
    )
    parser.add_argument(
        "--allow_errors", action="store_true", help="Allow errors in the notebook execution (default: False)"
    )
    parser.add_argument(
        "--kernel_name", type=str, default="", help="Kernel name to use for execution (default: use notebook's kernel)"
    )
    parser.add_argument(
        "--skip_cells_with_tag",
        type=str,
        default="skip-execution",
        help="Tag to skip cells with (default: 'skip-execution')",
    )
    parser.add_argument("input_path", type=str, help="Path to the input notebook file")
    args = parser.parse_args()

    NotebookRunner(
        input_path=args.input_path,
        output_path=args.output_path,
        evaluate_path=args.evaluate_path,
        timeout=args.timeout,
        startup_timeout=args.startup_timeout,
        allow_errors=args.allow_errors,
        kernel_name=args.kernel_name,
        skip_cells_with_tag=args.skip_cells_with_tag,
    ).run()


if __name__ == "__main__":
    main()
