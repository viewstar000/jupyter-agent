"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from IPython.display import Markdown
from ..utils import REPLY_TASK_RESULT
from .base import BaseTaskAgent, AGENT_OUTPUT_FORMAT_TEXT

TASK_SUMMARY_PROMPT = """\
**角色定义**：

你是一个信息提炼专家，能够从分析结果中提取关键结论。

**任务要求**：

- 将代码执行的输出与结果转化为**人类可读的总结**
- 包含以下内容：
  1. 代码执行结果总结
  2. 核心发现（如"Electronics类别月均增长12%"）
  3. 数据支撑（引用关键数值或图表）
  4. 其它建议（如新子任务Prompt等）

注：任务代码执行的结果不会记录在全局上下文中，只有任务总结的结果会记录在全局上下文中，
因此任务总结中应包含对代码执行结果的简要说明，以便后续子任务使用。

{% include "TASK_OUTPUT_FORMAT" %}

---

{% include "TASK_CONTEXTS" %}

---

{% include "CODE_CONTEXTS" %}

---

**当前子任务信息**:

### 当前子任务目标：
{{ task.task_subject }}

### 当前子任务代码需求：
{{ task.task_coding_prompt }}

### 当前代码：
```python
{{ task.cell_code }}
```

### 当前代码执行的输出与结果：
{{ task.cell_output }}
{{ task.cell_result }}

### 当前任务总结要求：
{{ task.task_summary_prompt }}

---

请按要求输出任务总结：
"""


class TaskSummaryAgent(BaseTaskAgent):

    PROMPT = TASK_SUMMARY_PROMPT
    OUTPUT_FORMAT = AGENT_OUTPUT_FORMAT_TEXT
    DISPLAY_REPLY = False

    def on_reply(self, reply: str):
        self._C(Markdown("### 任务总结\n" + reply), reply_type=REPLY_TASK_RESULT)
        assert reply, "Reply is empty"
        self.task_context.task_result = reply
