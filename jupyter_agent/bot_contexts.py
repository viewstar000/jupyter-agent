"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import os
import re
import json
import yaml
import time
import shlex
import argparse
import traceback
import nbformat
import ipynbname

from IPython.core.getipython import get_ipython
from .utils import DebugMixin
from .utils import REPLY_TASK_RESULT, REPLY_CELL_ERROR


class TaskCellContext(DebugMixin):
    """任务单元格上下文类"""

    def __init__(
        self,
        cur_line,
        cur_content,
        notebook_path=None,
        max_output_size=24 * 1024,
        max_result_size=24 * 1024,
        max_error_size=4 * 1024,
        debug_level=0,
    ):
        """初始化任务单元格上下文"""
        DebugMixin.__init__(self, debug_level)
        self.cur_line = cur_line and cur_line.strip()
        self.cur_content = cur_line and cur_content.strip()
        self.notebook_path = notebook_path
        self.max_output_size = max_output_size
        self.max_result_size = max_result_size
        self.max_error_size = max_error_size
        self.task_stage = None
        self.task_subject = ""
        self.task_coding_prompt = ""
        self.task_verify_prompt = ""
        self.task_summary_prompt = ""
        self.cell_code = self.cur_content
        self._cell_output = ""
        self._cell_result = ""
        self._cell_error = ""
        self.task_issue = ""
        self.task_result = ""
        self.remain_args = []
        if self.cur_line is False and self.cur_content.startswith("%%bot"):
            self.cur_line, self.cur_content = self.cur_content.split("\n", 1)
            self.cur_line = self.cur_line[5:].strip()
            self.cur_content = self.cur_content.strip()
            self.cell_code = self.cur_content
        self.parse_bot_cell()
        if self.notebook_path is not False:
            self.load_cell_context()

    def get_cell_output(self):
        """获取任务单元格的输出"""
        if len(self._cell_output) > self.max_output_size:
            half_size = self.max_output_size // 2
            self._cell_output = self._cell_output[:half_size] + "..." + self._cell_output[-half_size:]
        return self._cell_output

    def set_cell_output(self, output):
        """设置任务单元格的输出"""
        self._cell_output = output
        if len(self._cell_output) > self.max_output_size:
            half_size = self.max_output_size // 2
            self._cell_output = self._cell_output[:half_size] + "..." + self._cell_output[-half_size:]

    cell_output = property(get_cell_output, set_cell_output)

    def get_cell_result(self):
        """获取任务单元格的结果"""
        if len(self._cell_result) > self.max_result_size:
            half_size = self.max_result_size // 2
            self._cell_result = self._cell_result[:half_size] + "..." + self._cell_result[-half_size:]
        return self._cell_result

    def set_cell_result(self, result):
        """设置任务单元格的结果"""
        self._cell_result = result
        if len(self._cell_result) > self.max_result_size:
            half_size = self.max_result_size // 2
            self._cell_result = self._cell_result[:half_size] + "..." + self._cell_result[-half_size:]

    cell_result = property(get_cell_result, set_cell_result)

    def get_cell_error(self):
        """获取任务单元格的错误信息"""
        if len(self._cell_error) > self.max_error_size:
            half_size = self.max_error_size // 2
            self._cell_error = self._cell_error[:half_size] + "..." + self._cell_error[-half_size:]
        return self._cell_error

    def set_cell_error(self, error):
        """设置任务单元格的错误信息"""
        self._cell_error = error
        if len(self._cell_error) > self.max_error_size:
            half_size = self.max_error_size // 2
            self._cell_error = self._cell_error[:half_size] + "..." + self._cell_error[-half_size:]

    cell_error = property(get_cell_error, set_cell_error)

    def parse_bot_cell(self):
        """解析任务单元格的魔法行"""
        parser = argparse.ArgumentParser()
        parser.add_argument("-s", "--stage", type=str, default=None, help="Task stage")
        args = shlex.split(self.cur_line)
        options, self.remain_args = parser.parse_known_args(args)
        self.task_stage = options.stage
        cell_options = ""
        cell_code = ""
        is_option = False
        for line in self.cur_content.split("\n"):
            if line.strip() == "## Task Options:":
                is_option = True
                continue
            if line.strip() == "## ---":
                is_option = False
                continue
            if is_option:
                if line.startswith("# "):
                    cell_options += line[2:] + "\n"
                else:
                    is_option = False
                    cell_code += line + "\n"
            else:
                cell_code += line + "\n"
        self.cell_code = cell_code.strip()
        if cell_options:
            cell_options = yaml.safe_load(cell_options)
            self.task_stage = self.task_stage or cell_options.get("stage", self.task_stage)
            self.task_subject = cell_options.get("subject", "")
            self.task_coding_prompt = cell_options.get("coding_prompt", "")
            self.task_verify_prompt = cell_options.get("verify_prompt", "")
            self.task_summary_prompt = cell_options.get("summary_prompt", "")
            self.task_result = cell_options.get("result", "")
            self.task_issue = cell_options.get("issues", "")

    def update_bot_cell(self):
        """生成Cell内容"""
        magic_args = ["%%bot", "-s", self.task_stage] + self.remain_args
        magic_line = shlex.join(magic_args) + "\n"
        cell_source = ""
        cell_options = {}
        if self.task_subject:
            cell_options["subject"] = self.task_subject
        if self.task_coding_prompt:
            cell_options["coding_prompt"] = self.task_coding_prompt
        if self.task_verify_prompt:
            cell_options["verify_prompt"] = self.task_verify_prompt
        if self.task_summary_prompt:
            cell_options["summary_prompt"] = self.task_summary_prompt
        if self.task_result:
            cell_options["result"] = self.task_result
        if self.task_issue:
            cell_options["issues"] = self.task_issue
        if cell_options:
            cell_options["update_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            cell_source += self.format_options(cell_options)
        cell_source += "\n" + self.cell_code
        cell_source = "\n".join([magic_line, cell_source])
        ipython = get_ipython()
        if ipython is not None:
            ipython.set_next_input(cell_source, replace=True)

    def format_options(self, cell_options):
        """格式化任务选项"""
        result = "\n## Task Options:\n"
        for key, value in cell_options.items():
            if isinstance(value, str) and "\n" in value:
                result += f"# {key}: |\n"
                for line in value.split("\n"):
                    result += "#     " + line + "\n"
            elif isinstance(value, str) and (":" in value or '"' in value):
                result += f"# {key}: {repr(value)}\n"
            else:
                result += f"# {key}: {value}\n"
        result += "## ---\n"
        return result

    def load_cell_context(self):
        """加载当前任务单元格的上下文"""
        try:
            self.notebook_path = self.notebook_path or ipynbname.path()
            self.debug(f"Loading Task Cell Context: {self.notebook_path}")
            with open(self.notebook_path, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
            for idx, cell in enumerate(nb.cells):
                self.debug(f"CELL[{idx}] Source:", cell["cell_type"], repr(cell["source"][:50]), level=3)
                if cell["cell_type"] == "code":
                    cell_source = cell["source"].strip()
                    if (
                        cell_source.startswith("%%bot")
                        and self.cur_line in cell_source
                        and cell_source.endswith(self.cur_content)
                    ):
                        self.debug(f"CELL[{idx}] is current cell", level=2)
                        self.cell_output = ""
                        self.cell_result = ""
                        self.cell_error = ""
                        for output in cell.get("outputs", []):
                            # Available output types: stream, error, execute_result, display_data
                            if output["output_type"] == "stream":
                                self.debug(
                                    f"CELL[{idx}] Stream output: {output["name"]}:{repr(output['text'])[:50]}", level=3
                                )
                                self.cell_output += output["name"] + ":\n" + output["text"] + "\n"
                            if output["output_type"] == "error":
                                self.debug(
                                    f"CELL[{idx}] Error output: {output.get("ename", "")} {output.get('evalue', "")}",
                                    level=3,
                                )
                                self.cell_error += output.get("ename", "") + ": " + output.get("evalue", "") + "\n"
                                if "trackback" in output:
                                    self.cell_error += "Traceback:\n" + "\n".join(output.get("traceback", [])) + "\n"
                            if output["output_type"] == "execute_result":
                                output_data = output.get("data", {})
                                output_text = output_data.get("text/markdown") or output_data.get("text/plain")
                                self.debug(f"CELL[{idx}] Execute result: {repr(output_text)[:50]}", level=3)
                                self.cell_result += output_text + "\n"
                            if output["output_type"] == "display_data":
                                output_meta = output.get("metadata", {})
                                if output_meta.get("exclude_from_context", False):
                                    continue
                                output_data = output.get("data", {})
                                output_text = output_data.get("text/markdown") or output_data.get("text/plain")
                                reply_type = output_meta.get("reply_type")
                                if reply_type == REPLY_CELL_ERROR:
                                    self.debug(f"CELL[{idx}] Cell error: {repr(output_text)[:50]}", level=3)
                                    self.cell_error += output_text + "\n"
                                else:
                                    self.debug(f"CELL[{idx}] Display data: {repr(output_text)[:50]}", level=3)
                                    self.cell_output += output_text + "\n"
                        break
        except Exception as e:
            self.debug("Failed to get notebook cells {}: {}".format(type(e), str(e)))
            self.debug(traceback.format_exc(limit=2), level=2)


class NotebookContext(DebugMixin):
    """Notebook上下文类"""

    def __init__(self, cur_line, cur_content, notebook_path=None, debug_level=0):
        """初始化Notebook上下文"""
        DebugMixin.__init__(self, debug_level)
        self.cur_line = cur_line.strip()
        self.cur_content = cur_content.strip()
        self.notebook_path = notebook_path
        self.notebook_state = None
        self._cells = []

    @property
    def cells(self):
        """获取当前cell之前的所有cell内容"""
        try:
            self.notebook_path = self.notebook_path or ipynbname.path()
            if (
                not self._cells
                or self.notebook_state is None
                or self.notebook_state != os.stat(self.notebook_path).st_mtime
            ):
                self.debug(f"Loading Notebook Context: {self.notebook_path}")
                with open(self.notebook_path, "r", encoding="utf-8") as f:
                    nb = nbformat.read(f, as_version=4)
                self._cells = []
                for idx, cell in enumerate(nb.cells):
                    self.debug(f"CELL[{idx}] type:", cell["cell_type"])
                    if cell["cell_type"] in ("code", "markdown"):
                        cell_type = cell["cell_type"]
                        cell_source = cell["source"].strip()
                        cell_outputs = self.get_cell_outputs(cell)
                        self.debug(f"CELL[{idx}] Source:", repr(cell_source)[:50], level=3)
                        self.debug(f"CELL[{idx}] Outputs:", repr(cell_outputs)[:50], level=3)
                        if cell_source.startswith("%%bot"):
                            self.debug(f"CELL[{idx}] is bot magic cell")
                            if self.cur_line in cell_source and cell_source.endswith(self.cur_content):
                                self.debug(f"CELL[{idx}] is current cell, BREAK", level=1)
                                break
                            line, source = cell_source.split("\n", 1)
                            if "-P" in line or "--planning" in line:
                                self.debug(f"CELL[{idx}] is planning cell")
                                cell_type = "global_plan"
                                cell_source = source.strip()
                                self._cells.append(
                                    {
                                        "type": cell_type,
                                        "context": ["TASK"],
                                        "source": cell_source,
                                        "outputs": cell_outputs,
                                    }
                                )
                            else:
                                self.debug(f"CELL[{idx}] is task cell")
                                cell_context = TaskCellContext(line, source, notebook_path=False)
                                cell_type = "task"
                                cell_subject = cell_context.task_subject
                                cell_source = cell_context.cell_code
                                if cell_context.task_result:
                                    cell_outputs = [cell_context.task_result]
                                self._cells.append(
                                    {
                                        "type": cell_type,
                                        "context": ["TASK"],
                                        "subject": cell_subject,
                                        "source": cell_source,
                                        "outputs": cell_outputs,
                                    }
                                )
                        else:
                            if mo := re.match(r"^#\s*BOT_CONTEXT:(.+)$", cell_source.split("\n", 1)[0]):
                                context_types = [t.strip().upper() for t in mo.group(1).strip().split(",")]
                                cell_source = cell_source.split("\n", 1)[1].strip()
                                if "EXCLUDE" in context_types:
                                    continue
                            elif cell_type in ("markdown", "text"):
                                context_types = ["TASK"]
                            elif cell_type == "code":
                                context_types = ["CODE"]
                            else:
                                continue
                            self.debug(f"CELL[{idx}] is {', '.join(context_types)} cell")
                            self._cells.append(
                                {
                                    "type": cell_type,
                                    "context": context_types,
                                    "source": cell_source,
                                    "outputs": cell_outputs,
                                }
                            )
                self.notebook_state = os.stat(self.notebook_path).st_mtime
                self.debug(f"Got {len(self._cells)} notebook cells")
        except Exception as e:
            self.debug("Failed to get notebook cells {}: {}".format(type(e), str(e)))
            self.debug(traceback.format_exc(limit=2), level=2)
            self._cells = []
        return self._cells

    def get_cell_outputs(self, cell):
        cell_outputs = []
        for output in cell.get("outputs", []):
            # Available output types: stream, error, execute_result, display_data
            if output["output_type"] == "display_data":
                output_data = output.get("data", {})
                output_meta = output.get("metadata", {})
                if output_meta.get("reply_type") == REPLY_TASK_RESULT:
                    output_text = output_data.get("text/markdown") or output_data.get("text/plain")
                    self.debug(f"CELL Task result: {repr(output_text)[:50]}", level=3)
                    cell_outputs.append(output_text)
        return cell_outputs

    def get_cell_contents(self, include_global=True):
        """准备上下文内容"""
        contents = []
        sizes = []
        if self.cells:
            contents.append({"type": "text", "text": "## 以下是当前notebook的内容：\n"})
            tid = 1
            for idx, context_cell in enumerate(self.cells):
                cell_type = context_cell["type"]
                if cell_type == "markdown" or cell_type == "text":
                    contents.append(
                        {
                            "type": "text",
                            "text": f"\n>>> Cell[{idx + 1}] Text:\n\n{context_cell['source']}\n\n---\n",
                        }
                    )
                    sizes.append(len(contents[-1]["text"]))
                elif cell_type == "code":
                    contents.append(
                        {
                            "type": "text",
                            "text": f"\n>>> Cell[{idx + 1}] Code:\n\n```python\n{context_cell['source']}\n```\n\n---\n",
                        }
                    )
                    sizes.append(len(contents[-1]["text"]))
                elif cell_type == "task":
                    cell_subject = context_cell["subject"]
                    cell_code = context_cell["source"]
                    cell_outputs = "\n".join(context_cell["outputs"])
                    contents.append(
                        {
                            "type": "text",
                            "text": (
                                f"\n>>> Cell[{idx + 1}] Task:\n"
                                f"\n## 子任务 {tid} ({'已完成' if cell_outputs else '未完成'})\n"
                                f"\n### 任务目标\n"
                                f"\n{cell_subject}\n"
                                f"\n### 任务代码\n"
                                f"\n```python\n{cell_code}\n```\n"
                                f"\n### 任务结果\n"
                                f"\n{cell_outputs}\n"
                                f"\n---\n"
                            ),
                        }
                    )
                    tid += 1
                    sizes.append(len(contents[-1]["text"]))
                elif cell_type == "global_plan" and include_global:
                    subject = context_cell["source"]
                    plan = "\n".join(context_cell["outputs"])
                    contents.append(
                        {
                            "type": "text",
                            "text": f"\n>>> Cell[{idx + 1}] Global Subject:\n\n{subject}\n\n{plan}\n\n---\n",
                        }
                    )
                    sizes.append(len(contents[-1]["text"]))
            total_size = sum(sizes)
            self.debug("Total Notebook Contents size: {} chars, {}".format(total_size, sizes))
        return contents

    def get_task_contents(self):
        """准备上下文内容"""
        contents = []
        sizes = []
        if self.cells:
            contents.append({"type": "text", "text": "## 以下当前的任务规划及完成情况：\n"})
            tid = 1
            for idx, context_cell in enumerate(self.cells):
                cell_type = context_cell["type"]
                context_types = context_cell["context"]
                if cell_type == "task":
                    cell_subject = context_cell["subject"]
                    cell_outputs = "\n".join(context_cell["outputs"])
                    contents.append(
                        {
                            "type": "text",
                            "text": (
                                f"\n## 子任务 {tid} ({'已完成' if cell_outputs else '未完成'})\n"
                                f"\n### 任务目标\n"
                                f"\n{cell_subject}\n"
                                f"\n### 任务结果\n"
                                f"\n{cell_outputs}\n"
                            ),
                        }
                    )
                    tid += 1
                    sizes.append(len(contents[-1]["text"]))
                elif cell_type == "global_plan" and context_cell["source"].strip():
                    subject = context_cell["source"]
                    plan = "\n".join(context_cell["outputs"])
                    contents.append({"type": "text", "text": f"\n{subject}\n\n{plan}\n"})
                    sizes.append(len(contents[-1]["text"]))
                elif "TASK" in context_types and context_cell["source"].strip():
                    contents.append({"type": "text", "text": f"\n{context_cell['source']}\n"})
                    sizes.append(len(contents[-1]["text"]))
            total_size = sum(sizes)
            self.debug("Total Task Contents size: {} chars, {}".format(total_size, sizes))
        return contents

    def get_code_contents(self):
        """准备上下文内容"""
        contents = []
        sizes = []
        if self.cells:
            contents.append({"type": "text", "text": "## 以下是当前已执行的代码：\n"})
            tid = 1
            for idx, context_cell in enumerate(self.cells):
                cell_type = context_cell["type"]
                context_types = context_cell["context"]
                if "CODE" in context_types and context_cell["source"].strip():
                    contents.append({"type": "text", "text": f"\n## Cell[{idx + 1}]\n\n{context_cell['source']}\n"})
                    sizes.append(len(contents[-1]["text"]))
                elif cell_type == "task":
                    cell_code = context_cell["source"]
                    contents.append(
                        {"type": "text", "text": f"\n## Cell[{idx + 1}] for Task[{tid}]:\n\n{cell_code}\n"}
                    )
                    tid += 1
                    sizes.append(len(contents[-1]["text"]))
            total_size = sum(sizes)
            self.debug("Total Code Contents size: {} chars, {}".format(total_size, sizes))
        return contents
