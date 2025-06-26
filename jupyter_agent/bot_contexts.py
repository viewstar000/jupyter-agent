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


from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field
from IPython.core.getipython import get_ipython
from .bot_outputs import _D, _I, _W, _E, _F, _A, ReplyType
from .utils import get_env_capbilities


class CellType(str, Enum):
    CODE = "code"
    MARKDOWN = "markdown"
    PLANNING = "planning"
    TASK = "task"


class CellContext:

    def __init__(self, idx: int, cell: dict):
        self.cell_idx = idx
        self.cell_id = cell.get("id")
        self.cell_type: Optional[CellType] = cell["cell_type"]
        self.cell_source = cell["source"].strip()
        self.cell_tags = set(cell.get("metadata", {}).get("tags", []))
        if mo := re.match(r"^#\s*BOT_CONTEXT:(.+)$", self.cell_source.split("\n", 1)[0]):
            context_tags = ["CTX_" + t.strip().upper() for t in mo.group(1).strip().split(",")]
            self.cell_tags |= set(context_tags)
            self.cell_source = self.cell_source.split("\n", 1)[1].strip()

    @property
    def type(self):
        return self.cell_type

    @property
    def source(self):
        return self.cell_source

    @property
    def is_code_context(self):
        return (
            self.cell_type == CellType.CODE or "CTX_CODE" in self.cell_tags
        ) and "CTX_EXCLUDE" not in self.cell_tags

    @property
    def is_task_context(self):
        return (
            self.cell_type in (CellType.TASK, CellType.PLANNING, CellType.MARKDOWN) or "CTX_TASK" in self.cell_tags
        ) and "CTX_EXCLUDE" not in self.cell_tags


class CodeCellContext(CellContext):
    """任务单元格上下文类"""

    max_output_size = 24 * 1024
    max_result_size = 24 * 1024
    max_error_size = 4 * 1024

    def __init__(self, idx: int, cell: dict):
        """初始化任务单元格上下文"""
        super().__init__(idx, cell)
        assert self.cell_type == CellType.CODE
        self._cell_output = ""
        self._cell_result = ""
        self._cell_error = ""
        self.load_cell_outputs(cell)

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

    def load_cell_outputs(self, cell):
        """加载当前任务单元格的上下文"""
        try:
            self.cell_output = ""
            self.cell_result = ""
            self.cell_error = ""
            for output in cell.get("outputs", []):
                # Available output types: stream, error, execute_result, display_data
                if output["output_type"] == "stream":
                    _D(f"CELL[{self.cell_idx}] Stream output: {output["name"]}:{repr(output['text'])[:50]}")
                    self.cell_output += output["name"] + ":\n" + output["text"] + "\n"
                if output["output_type"] == "error":
                    _D(f"CELL[{self.cell_idx}] Error output: {output.get("ename", "")} {output.get('evalue', "")}")
                    self.cell_error += output.get("ename", "") + ": " + output.get("evalue", "") + "\n"
                    if "traceback" in output:
                        self.cell_error += "Traceback:\n" + "\n".join(output.get("traceback", [])) + "\n"
                if output["output_type"] == "execute_result":
                    output_data = output.get("data", {})
                    output_text = output_data.get("text/markdown") or output_data.get("text/plain")
                    _D(f"CELL[{self.cell_idx}] Execute result: {repr(output_text)[:50]}")
                    self.cell_result += output_text + "\n"
                if output["output_type"] == "display_data":
                    output_meta = output.get("metadata", {})
                    if not output_meta.get("exclude_from_context", False):
                        output_data = output.get("data", {})
                        output_text = output_data.get("text/markdown") or output_data.get("text/plain")
                        reply_type = output_meta.get("reply_type")
                        if reply_type == ReplyType.CELL_ERROR:
                            _D(f"CELL[{self.cell_idx}] Display error data: {repr(output_text)[:50]}")
                            self.cell_error += output_text + "\n"
                        else:
                            _D(f"CELL[{self.cell_idx}] Display output data: {repr(output_text)[:50]}")
                            self.cell_output += output_text + "\n"
        except Exception as e:
            _W("Failed to load notebook cells {}: {}".format(type(e), str(e)))
            _W(traceback.format_exc(limit=2))


class AgentData(BaseModel):
    task_id: str = Field("", description="任务ID")
    subject: str = Field("", description="任务目标")
    coding_prompt: str = Field("", description="Agent编程提示")
    verify_prompt: str = Field("", description="Agent验证提示")
    summary_prompt: str = Field("", description="Agent总结提示")
    issue: str = Field("", description="Agent验证不通过的问题")
    result: str = Field("", description="Agent执行结果")
    important_infos: Optional[dict] = Field(None, description="重要信息[JSON]")
    request_above_supply_infos: Optional[list] = Field(None, description="前置用户需求补充[JSON]")
    request_below_supply_infos: Optional[list] = Field(None, description="后置用户需求补充[JSON]")

    @classmethod
    def default(cls) -> "AgentData":
        model_fields = getattr(cls, "model_fields", None)
        if model_fields and hasattr(model_fields, "items"):
            default = {
                name: field.examples[0] if getattr(field, "examples", None) else getattr(field, "default", None)
                for name, field in model_fields.items()
            }
        else:
            default = {}
        return cls(**default)  # type: ignore


class AgentCellContext(CodeCellContext):
    """任务单元格上下文类"""

    def __init__(self, idx: int, cell: dict):
        """初始化任务单元格上下文"""
        super().__init__(idx, cell)
        self.agent_flow = None
        self.agent_stage = None
        self.magic_line, self.magic_code = self.cell_source.split("\n", 1)
        self.magic_argv = shlex.split(self.magic_line)
        self.magic_name = self.magic_argv[0]
        self.magic_argv = self.magic_argv[1:]
        self.agent_data = AgentData.default()
        self._remain_args = []
        self._cell_code = ""
        self.parse_magic_argv()
        self.load_result_from_outputs(cell)
        self.load_data_from_metadata(cell)
        self.load_data_from_source()

    def get_source(self):
        return self._cell_code

    def set_source(self, value):
        self._cell_code = value

    source = property(get_source, set_source)

    @property
    def output(self):
        return self.cell_output + "\n" + self.cell_result

    @property
    def result(self):
        return self.agent_data.result

    def __getattr__(self, name):
        return getattr(self.agent_data, name)

    def has_data(self, name):
        return hasattr(self.agent_data, name)

    def get_data(self, name):
        return getattr(self.agent_data, name)

    def set_data(self, name, value):
        setattr(self.agent_data, name, value)

    def parse_magic_argv(self):
        """解析任务单元格的magic命令参数"""
        parser = argparse.ArgumentParser()
        parser.add_argument("-P", "--planning", action="store_true", default=False, help="Run in planning mode")
        parser.add_argument("-f", "--flow", type=str, default=None, help="Task stage")
        parser.add_argument("-s", "--stage", type=str, default=None, help="Task stage")
        options, self._remain_args = parser.parse_known_args(self.magic_argv)
        _D(
            "CELL[{}] Magic Name: {}, Magic Args: {}, Remain Args: {}".format(
                self.cell_idx, self.magic_name, options, self._remain_args
            )
        )
        self.agent_flow = options.flow
        self.agent_stage = options.stage
        if options.planning and not self.agent_flow:
            self.agent_flow = "planning"
        if self.agent_flow and self.agent_flow.startswith("planning"):
            self.cell_type = CellType.PLANNING
        else:
            self.cell_type = CellType.TASK

    def load_data_from_source(self):
        """解析任务单元格的选项"""
        cell_options = ""
        cell_code = ""
        is_option = False
        for line in self.magic_code.split("\n"):
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
        self._cell_code = cell_code.strip()
        _D("CELL[{}] Cell Options: {} ...".format(self.cell_idx, repr(cell_options)[:80]))
        _D("CELL[{}] Cell Code: {} ...".format(self.cell_idx, repr(self._cell_code)[:80]))
        if cell_options:
            try:
                cell_options = yaml.safe_load(cell_options)
                for key, value in cell_options.items():
                    if self.has_data(key):
                        if isinstance(self.get_data(key), (dict, list)) and isinstance(value, str):
                            value = json.loads(value)
                        _D("CELL[{}] Load task option {}: {}".format(self.cell_idx, key, value))
                        self.set_data(key, value)
            except Exception as e:
                _W("Failed to load task options {}: {}".format(type(e), str(e)))
                _W(traceback.format_exc(limit=2))

    def load_result_from_outputs(self, cell):
        task_result = ""
        for output in cell.get("outputs", []):
            # Available output types: stream, error, execute_result, display_data
            if output["output_type"] == "display_data":
                output_data = output.get("data", {})
                output_meta = output.get("metadata", {})
                if output_meta.get("reply_type") == ReplyType.TASK_RESULT:
                    output_text = output_data.get("text/markdown") or output_data.get("text/plain")
                    _D(f"CELL[{self.cell_idx}] Task result: {repr(output_text)[:80]}")
                    task_result += "\n" + output_text
        if task_result.strip():
            self.agent_data.result = task_result

    def load_data_from_metadata(self, cell):
        agent_meta_infos = cell.get("metadata", {}).get("jupyter-agent-data", {})
        _D("CELL[{}] Agent Meta Data: {}".format(self.cell_idx, repr(agent_meta_infos)[:80]))
        for k, v in agent_meta_infos.items():
            if self.has_data(k):
                _D(f"CELL[{self.cell_idx}] Load agent meta data: {k}: {repr(v)[:80]}")
                self.set_data(k, v)

    def format_magic_line(self):
        magic_args = ["%%bot"]
        if self.agent_stage:
            magic_args += ["-s", self.agent_stage]
        if self.agent_flow:
            if self.agent_flow == "planning":
                magic_args += ["-P"]
            else:
                magic_args += ["-f", self.agent_flow]
        magic_args += self._remain_args
        magic_line = shlex.join(magic_args) + "\n"
        return magic_line

    def _format_yaml_element(self, e, level=0, indent=4):

        space = " " * indent * level
        result = ""
        if isinstance(e, dict):
            result += "\n"
            for k, v in e.items():
                if not v:
                    continue
                result += f"{space}{k}: "
                result += self._format_yaml_element(v, level + 1, indent)
        elif isinstance(e, list):
            result += "\n"
            for v in e:
                result += f"{space}- "
                result += self._format_yaml_element(v, level + 1, indent)
        elif isinstance(e, BaseModel):
            result += self._format_yaml_element(e.model_dump(), level, indent)
        elif isinstance(e, str):
            if "\n" in e:
                if e.endswith("\n"):
                    result += f"|\n"
                else:
                    result += f"|-\n"
                for line in e.split("\n"):
                    result += f"{space}{line}\n"
            elif ":" in e or '"' in e or "'" in e:
                result += f"'{e.replace("'", "''")}'\n"
            else:
                result += f"{e}\n"
        elif e is None:
            result += "null\n"
        else:
            result += f"{e}\n"
        return result

    def format_cell_options(self):
        cell_options = {}
        if get_env_capbilities().save_metadata:
            if self.agent_data.task_id:
                cell_options["task_id"] = self.agent_data.task_id
            if self.agent_data.subject:
                cell_options["subject"] = self.agent_data.subject
        else:
            for key, value in self.agent_data.model_dump().items():
                if key == "result" and self.type == CellType.PLANNING:
                    continue
                if value:
                    if (
                        isinstance(value, (dict, list))
                        and AgentData.model_fields[key]
                        and AgentData.model_fields[key].description is not None
                        and "[JSON]" in AgentData.model_fields[key].description  # type: ignore
                    ):
                        value = json.dumps(value, ensure_ascii=False, indent=4)
                    cell_options[key] = value
            if cell_options:
                cell_options["update_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if cell_options:
            result = "\n".join(f"# {line}" for line in self._format_yaml_element(cell_options).strip().split("\n"))
            result = f"\n## Task Options:\n{result}\n## ---\n"
            return result
        return ""

    def update_cell(self):
        """生成Cell内容"""
        _I("Updating Cell ...")
        _A(**self.agent_data.model_dump())
        cell_source = ""
        cell_source += self.format_magic_line()
        cell_source += "\n" + self.format_cell_options()
        cell_source += "\n" + self.source
        ipython = get_ipython()
        if ipython is not None:
            _D("Updating Cell Source: {} ...".format(repr(cell_source)[:80]))
            ipython.set_next_input(cell_source, replace=True)


class NotebookContext:
    """Notebook上下文类"""

    def __init__(self, cur_line, cur_content, notebook_path):
        """初始化Notebook上下文"""
        self.cur_line = cur_line.strip()
        self.cur_content = cur_content.strip()
        self.notebook_path = notebook_path
        self.notebook_state = None
        self._cells = []
        self._current_cell = None

    @property
    def cells(self):
        """获取当前cell之前的所有cell内容"""
        try:
            if (
                not self._cells
                or self.notebook_state is None
                or self.notebook_state != os.stat(self.notebook_path).st_mtime
            ):
                _I(f"Loading Notebook Context: {self.notebook_path}")
                with open(self.notebook_path, "r", encoding="utf-8") as f:
                    nb = nbformat.read(f, as_version=4)
                self._cells = []
                for idx, cell in enumerate(nb.cells):
                    _D(f"CELL[{idx}] {cell['cell_type']} {repr(cell['source'])[:80]}")
                    if cell["cell_type"] == "code":
                        if cell["source"].strip().startswith("%%bot"):
                            cell_ctx = AgentCellContext(idx, cell)
                            if (
                                self.cur_line.strip() == cell_ctx.magic_line[len(cell_ctx.magic_name) :].strip()
                                and self.cur_content.strip() == cell_ctx.magic_code.strip()
                            ):
                                if self._current_cell is None:
                                    _I(f"CELL[{idx}] Reach current cell, RETURN!")
                                    self._current_cell = cell_ctx
                                else:
                                    _I(f"CELL[{idx}] Reach current cell, SKIP!")
                                break
                        else:
                            cell_ctx = CodeCellContext(idx, cell)
                    else:
                        cell_ctx = CellContext(idx, cell)
                    self._cells.append(cell_ctx)
                self.notebook_state = os.stat(self.notebook_path).st_mtime
                _I(f"Got {len(self._cells)} notebook cells")
        except Exception as e:
            _E("Failed to get notebook cells {}: {}".format(type(e), str(e)))
            _E(traceback.format_exc(limit=2))
            self._cells = []
        return self._cells

    @property
    def cur_task(self):
        """获取当前任务单元格的上下文"""
        if self._current_cell is None:
            len(self.cells)
        return self._current_cell
