"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import json

from IPython.display import Markdown
from .base import BaseAgent
from ..bot_outputs import _D, _I, _W, _E, _F, _M, _B, _C, _O, ReplyType, markdown_block


class OutputTaskResult(BaseAgent):

    def __call__(self):
        """执行代码逻辑"""
        if self.task.result:
            _O(Markdown("### 任务结果\n\n"))
            _C(Markdown(self.task.result), reply_type=ReplyType.TASK_RESULT)
        if self.task.important_infos:
            _O(
                markdown_block(
                    f"```json\n{json.dumps(self.task.important_infos, indent=4, ensure_ascii=False)}\n```",
                    title="重要信息",
                )
            )
        return False, None
