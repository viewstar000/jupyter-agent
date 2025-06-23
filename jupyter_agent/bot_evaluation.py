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

    def save_notebook(**kwargs):
        """Save the executed notebook to the specified output path."""
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
                        print(
                            f"CELL[{cell_idx}] Evaluating record: {record['eval_type']} "
                            f"duration: {record['execution_duration']:.2f}s "
                            f"success: {record['is_success']} "
                            f"correct: {record['correct_score']:.2f}"
                        )
                        if evaluation_path:
                            with open(evaluation_path, "a") as eval_file:
                                record["notebook_name"] = output_path
                                eval_file.write(json.dumps(record) + "\n")
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
    client.execute()

    # Save it
    save_notebook()


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
