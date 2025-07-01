"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from IPython.display import Markdown
from .base import BaseChatAgent, AgentOutputFormat
from ..bot_outputs import _D, _I, _W, _E, _F, _M, _B, _C, _O
from ..bot_outputs import ReplyType

TASK_REASONER_PROMPT = """\
**角色定义**：

你是一个推理分析与信息提炼专家，能够从已有的数据、结果中推理分析并提取出关键结论。

**任务要求**：

- 在已有的数据、结果中进行推理分析，按需提取关键结论，并将结论输出为**人类可读的总结**
- 包含以下内容：
  1. 核心发现（如"Electronics类别月均增长12%"）
  2. 数据支撑（引用关键数值或图表）
  3. 其它建议（如新子任务Prompt等）

{% include "TASK_OUTPUT_FORMAT" %}

---

{% include "TASK_CONTEXTS" %}

---

{% include "CODE_CONTEXTS" %}

---

**当前子任务信息**:

### 当前子任务目标：
{{ task.subject }}

### 当前任务分析总结要求：
{{ task.summary_prompt }}

---

请按要求输出任务结论：
"""


class TaskReasoningAgent(BaseChatAgent):

    PROMPT = TASK_REASONER_PROMPT
    OUTPUT_FORMAT = AgentOutputFormat.TEXT
    DISPLAY_REPLY = False

    def on_reply(self, reply: str):
        assert reply, "Reply is empty"
        _M("### 任务总结\n" + reply)
        self.task.agent_data.issue = ""
        self.task.agent_data.result = reply
