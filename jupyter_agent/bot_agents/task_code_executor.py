"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import re

from IPython.core.getipython import get_ipython
from IPython.display import Markdown, clear_output
from ..utils import DebugMixin, TeeOutputCapture, REPLY_TASK_RESULT


class CodeExecutor(DebugMixin):

    def __init__(self, notebook_context, task_context, debug_level=0):
        """初始化代码执行器"""
        DebugMixin.__init__(self, debug_level=debug_level)
        self.notebook_context = notebook_context
        self.task_context = task_context

    def __call__(self):
        """执行代码逻辑"""
        self.debug("执行代码", self.task_context.cell_code[:50])
        ipython = get_ipython()
        exec_failed = False
        with TeeOutputCapture() as captured:
            if ipython is None:
                exec_failed = True
                self.task_context.cell_error = "IPython environment not found."
                self.debug("执行失败", "IPython environment not found.")
                result = None
            else:
                result = ipython.run_cell(self.task_context.cell_code)
                if result.success:
                    self.task_context.cell_result = "{}".format(result.result)
                    self.debug("执行结果", self.task_context.cell_result)
                    if self.task_context.task_result:
                        self._C(
                            Markdown("### 任务结论\n" + self.task_context.task_result), reply_type=REPLY_TASK_RESULT
                        )
                else:
                    exec_failed = True
                    exc_info = ipython._format_exception_for_storage(result.error_before_exec or result.error_in_exec)
                    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
                    clean_traceback = "\n".join(ansi_escape.sub("", line) for line in exc_info["traceback"])
                    self.task_context.cell_error = clean_traceback
                    self.debug(f"执行失败", clean_traceback)
        self.task_context.cell_output = ""
        if captured.stdout:
            self.task_context.cell_output += "Stdout:\n" + captured.stdout + "\n"
        if captured.stderr:
            self.task_context.cell_output += "Stderr:\n" + captured.stderr + "\n"
        if captured.outputs:
            self.task_context.cell_output += "Outputs:\n"
            for output in captured.outputs:
                output_content = output.data.get("text/markdown", "") or output.data.get("text/plain", "")
                self.task_context.cell_output += output_content
                self.task_context.cell_output += "\n"
        return exec_failed, not exec_failed
