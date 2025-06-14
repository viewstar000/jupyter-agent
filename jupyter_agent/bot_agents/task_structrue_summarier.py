"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import json

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from IPython.display import Markdown
from ..utils import REPLY_TASK_RESULT, RequestUserPrompt, UserPromptResponse, request_user_response
from .base import BaseTaskAgent, AGENT_OUTPUT_FORMAT_JSON

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
- 在引用其它已完成的子任务的结果时，特别是其important_infos中的信息，要保证准确、清晰、完整，不要出现任何误导信息

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


class RequestInfo(BaseModel):
    prompt: str = Field(description="需要用户补充更详细的信息的 Prompt", examples=["请补充与...相关的详细的信息"])
    example: Optional[str] = Field(None, description="示例", examples=["..."])


class TaskStructureSumaryOutput(BaseModel):

    summary: str = Field(description=f"任务总结的详细描述", examples=["..."])
    important_infos: Optional[Dict[str, Any]] = Field(
        None,
        description="任务总结中的重要信息，特别是需要后续子任务重点关注的信息。"
        "注意：该字段仅支持结构化信息，不能使用代码、长文本等非结构化信息",
        examples=[
            {
                "..._constraint": "...",
                "..._expression": "...",
                "..._patterns": ["...", "..."],
                "..._execution_strategies": ["...", "..."],
                "..._features": {"...": "...", "...": "..."},
                "..._mapping_rules": {"...": "...", "...": "..."},
                "...": "...",
            }
        ],
    )
    request_confirm_infos: Optional[List[RequestUserPrompt]] = Field(
        None, description="需要用户补充确认的信息，问题应尽量简单，只需要用户回答是/否或在备选项中选择即可"
    )


class TaskStructureSummaryAgent(BaseTaskAgent):

    PROMPT = TASK_SUMMARY_PROMPT
    OUTPUT_FORMAT = AGENT_OUTPUT_FORMAT_JSON
    OUTPUT_JSON_SCHEMA = TaskStructureSumaryOutput
    DISPLAY_REPLY = True

    def on_reply(self, reply: TaskStructureSumaryOutput):
        assert reply.summary, "Reply is empty"
        self._C(Markdown("### 任务总结\n\n" + reply.summary), reply_type=REPLY_TASK_RESULT)
        self.task_context.task_result = reply.summary
        if reply.important_infos:
            self.task_context.task_important_infos = reply.important_infos
        if reply.request_confirm_infos:
            self.task_context.task_confirm_infos = request_user_response(reply.request_confirm_infos)
