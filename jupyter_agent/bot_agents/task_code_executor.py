"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import re

from IPython.core.getipython import get_ipython
from IPython.display import Markdown, clear_output
from .base import BaseAgent
from ..utils import TeeOutputCapture
from ..bot_outputs import _D, _I, _W, _E, _F, _M, _B, _C, ReplyType


class CodeExecutor(BaseAgent):

    def __call__(self):
        """执行代码逻辑"""
        _D(f"执行代码: {repr(self.task.source)[:80]}")
        ipython = get_ipython()
        exec_failed = False
        with TeeOutputCapture() as captured:
            if ipython is None:
                exec_failed = True
                self.task.cell_error = "IPython environment not found."
                _E("执行失败: IPython environment not found.")
                result = None
            else:
                result = ipython.run_cell(self.task.source)
                if result.success:
                    self.task.cell_result = "{}".format(result.result)
                    _D(f"执行结果: {repr(self.task.cell_result)[:80]}")
                else:
                    exec_failed = True
                    exc_info = ipython._format_exception_for_storage(result.error_before_exec or result.error_in_exec)
                    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
                    clean_traceback = "\n".join(ansi_escape.sub("", line) for line in exc_info["traceback"])
                    self.task.cell_error = clean_traceback
                    _E(f"执行失败: {clean_traceback}")
        self.task.cell_output = ""
        if captured.stdout:
            self.task.cell_output += "Stdout:\n" + captured.stdout + "\n"
        if captured.stderr:
            self.task.cell_output += "Stderr:\n" + captured.stderr + "\n"
        if captured.outputs:
            self.task.cell_output += "Outputs:\n"
            for output in captured.outputs:
                output_content = output.data.get("text/markdown", "") or output.data.get("text/plain", "")
                self.task.cell_output += output_content
                self.task.cell_output += "\n"
        return exec_failed, not exec_failed
