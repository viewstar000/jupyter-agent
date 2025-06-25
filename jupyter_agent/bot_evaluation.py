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
from nbclient.client import NotebookClient
from .bot_outputs import NotebookEvalutionRecord, FlowEvalutionRecord, StageEvalutionRecord, BaseEvalutionRecord


def run_notebook(
    input_path: str | Path,
    output_path: str | Path = "",
    inplace: bool = False,
    timeout: int = -1,
    startup_timeout: int = 60,
    allow_errors: bool = False,
    kernel_name: str = "",
    skip_cells_with_tag: str = "skip-execution",
    evaluation_path: str | Path = "",
) -> None:
    """Run a notebook by path."""
    input_path = Path(input_path).with_suffix(".ipynb")

    assert not (inplace and output_path), "Cannot specify both 'inplace' and 'output_path'"
    if inplace:
        output_path = input_path
    elif not output_path:
        output_path = input_path.parent.joinpath(f"{input_path.with_suffix('').name}_executed.ipynb")
    output_path = os.path.abspath(output_path)

    # Open up the notebook we're going to run
    with input_path.open() as f:
        print("Running notebook:", input_path)
        nb = nbformat.read(f, as_version=4)

    start_time = 0
    is_global_finished = False

    def save_evaluation_record(record):
        nonlocal evaluation_path

        print(
            f"CELL[{record.cell_index}] Evaluation: {record.eval_type}[{record.evaluator}] "
            f"duration: {record.execution_duration:.2f}s "
            f"success: {record.is_success} "
            f"correct: {record.correct_score:.2f}"
        )
        if evaluation_path:
            with open(evaluation_path, "a") as eval_file:
                eval_file.write(record.model_dump_json() + "\n")

    def save_notebook(**kwargs):
        """Save the executed notebook to the specified output path."""
        nonlocal is_global_finished

        if kwargs:
            cell_idx = kwargs.get("cell_index", 0)
            cell_type = kwargs.get("cell", {}).get("cell_type")
            cell_id = kwargs.get("cell", {}).get("id")
            cell_exec_count = kwargs.get("cell", {}).get("execution_count")
            cell_meta = kwargs.get("cell", {}).get("metadata", {})
            cell_payloads = kwargs.get("execute_reply", {}).get("content", {}).get("payload", [])
            cell_outputs = kwargs.get("cell", {}).get("outputs", [])
            for payload in cell_payloads:
                if payload.get("source") == "set_next_input" and payload.get("replace") is True:
                    print(f"CELL[{cell_idx}] Replacing cell with set_next_input payload")
                    nb.cells[cell_idx].source = payload.get("text", "")
            cell_agent_data_timestamp = cell_meta.get("jupyter-agent-data-timestamp", 0)
            output_agent_data_timestamp = cell_agent_data_timestamp
            is_bot_cell = False
            is_flow_completed = False
            for output in cell_outputs:
                if output["output_type"] == "display_data":
                    output_meta = output.get("metadata", {})
                    if (
                        output_meta.get("jupyter-agent-data-store")
                        and output_meta.get("jupyter-agent-data-timestamp", 0) > output_agent_data_timestamp
                        and output_meta.get("jupyter-agent-data", {})
                    ):
                        print(f"CELL[{cell_idx}] Found jupyter-agent-data-store outputs, save it to cell metadata")
                        output_agent_data_timestamp = output_meta.get("jupyter-agent-data-timestamp", 0)
                        nb.cells[cell_idx].metadata["jupyter-agent-data-store"] = True
                        nb.cells[cell_idx].metadata["jupyter-agent-data-timestamp"] = output_agent_data_timestamp
                        if "jupyter-agent-data" not in nb.cells[cell_idx].metadata:
                            nb.cells[cell_idx].metadata["jupyter-agent-data"] = {}
                        nb.cells[cell_idx].metadata["jupyter-agent-data"].update(output_meta["jupyter-agent-data"])

                    for record in output_meta.get("jupyter-agent-evaluation-records", []):
                        is_bot_cell = True
                        if record["eval_type"] == "NOTEBOOK":
                            record = NotebookEvalutionRecord(**record)
                            record.timestamp = record.timestamp or time.time()
                            record.notebook_name = output_path
                            record.execution_duration = time.time() - start_time
                            is_global_finished = True
                            is_flow_completed = True
                            del nb.cells[cell_idx + 1 :]  # Remove all cells after the notebook cell
                        elif record["eval_type"] == "FLOW":
                            record = FlowEvalutionRecord(**record)
                            record.timestamp = record.timestamp or time.time()
                            record.notebook_name = output_path
                            is_flow_completed = True
                        elif record["eval_type"] == "STAGE":
                            record = StageEvalutionRecord(**record)
                            record.timestamp = record.timestamp or time.time()
                            record.notebook_name = output_path
                        else:
                            record = BaseEvalutionRecord(**record)
                            record.timestamp = record.timestamp or time.time()
                            record.notebook_name = output_path
                        save_evaluation_record(record)
            if is_bot_cell and not is_flow_completed:
                record = FlowEvalutionRecord(
                    timestamp=time.time(),
                    notebook_name=output_path,
                    evaluator="bot",
                    eval_type="FLOW",
                    cell_index=cell_idx,
                    is_success=False,
                )
                save_evaluation_record(record)
            print(f"CELL[{cell_idx}] Saving executed {cell_type} cell - {cell_id}: {cell_exec_count}")
        else:
            print(f"Saving executed notebook to: {output_path}")
        nbformat.write(nb, output_path)

    # Add metadata to the notebook
    nb.cells.insert(
        0,
        nbformat.v4.new_code_cell(
            source=(
                f"# Executed notebook: {input_path.name}\n"
                f"# Output saved to: {output_path}\n\n"
                f"__evaluation_ipynb_file__ = '{output_path}'\n"
            ),
            metadata={"tags": ["CTX_EXCLUDE"]},
        ),
    )
    save_notebook()

    # Configure nbclient to run the notebook
    client = NotebookClient(
        nb,
        timeout=timeout,
        startup_timeout=startup_timeout,
        skip_cells_with_tag=skip_cells_with_tag,
        allow_errors=allow_errors,
        kernel_name=kernel_name,
        resources={"metadata": {"path": input_path.parent.absolute()}},
        on_cell_executed=save_notebook,
    )

    # Run it
    print("Executing notebook...")
    start_time = time.time()
    client.execute()
    save_notebook()
    print("Notebook execution completed.")

    # If the notebook did not finish globally, append an evaluation record
    if not is_global_finished:
        print("Notebook execution did not finish globally, appending evaluation records.")
        record = NotebookEvalutionRecord(
            notebook_name=output_path,
            timestamp=time.time(),
            evaluator="bot",
            eval_type="NOTEBOOK",
            execution_duration=time.time() - start_time,
            is_success=False,
            correct_score=0.0,
        )
        save_evaluation_record(record)


def main():
    """Main function to run the notebook execution."""
    parser = argparse.ArgumentParser(description="Run a Jupyter notebook.")
    parser.add_argument(
        "-o", "--output_path", type=str, default="", help="Path to save the executed notebook (default: same as input)"
    )
    parser.add_argument(
        "-i", "--inplace", action="store_true", help="Run the notebook in place (overwrite input file)"
    )
    parser.add_argument(
        "-e",
        "--evaluation_path",
        type=str,
        default="",
        help="Path to save evaluation records (default: no evaluation records saved)",
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

    run_notebook(
        input_path=args.input_path,
        output_path=args.output_path,
        inplace=args.inplace,
        timeout=args.timeout,
        startup_timeout=args.startup_timeout,
        allow_errors=args.allow_errors,
        kernel_name=args.kernel_name,
        skip_cells_with_tag=args.skip_cells_with_tag,
        evaluation_path=args.evaluation_path,
    )


if __name__ == "__main__":
    main()
