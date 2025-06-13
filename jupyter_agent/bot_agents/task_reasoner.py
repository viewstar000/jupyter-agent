"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from IPython.display import Markdown
from ..utils import REPLY_TASK_RESULT
from .base import BaseTaskAgent, AGENT_OUTPUT_FORMAT_TEXT

TASK_REASONER_PROMPT = """\
**角色定义**：

你是一个推理分析与信息提炼专家，能够从已有的数据、结果中推理分析并提取出关键结论。

**任务要求**：

- 在已有的数据、结果中进行推理分析，按需提取关键结论，并将结论输出为**人类可读的总结**
- 包含以下内容：
  2. 核心发现（如"Electronics类别月均增长12%"）
  3. 数据支撑（引用关键数值或图表）
  4. 其它建议（如新子任务Prompt等）

{% include "TASK_OUTPUT_FORMAT" %}

---

{% include "TASK_CONTEXTS" %}

---

{% include "CODE_CONTEXTS" %}

---

**当前子任务信息**:

### 当前子任务目标：
{{ task.task_subject }}

### 当前任务分析总结要求：
{{ task.task_summary_prompt }}

---

请按要求输出任务结论：
"""


class TaskReasoningAgent(BaseTaskAgent):

    PROMPT = TASK_REASONER_PROMPT
    OUTPUT_FORMAT = AGENT_OUTPUT_FORMAT_TEXT
    DISPLAY_REPLY = False

    def on_reply(self, reply: str):
        self._C(Markdown("### 任务总结\n" + reply), reply_type=REPLY_TASK_RESULT)
        assert reply, "Reply is empty"
        self.task_context.task_result = reply
