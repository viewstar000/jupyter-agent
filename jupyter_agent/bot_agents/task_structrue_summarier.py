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
from .base import BaseChatAgent, AgentOutputFormat
from ..bot_outputs import ReplyType, _D, _I, _W, _E, _F, _M, _B, _C, _O, markdown_block
from ..bot_actions import RequestUserSupplyInfo

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
{{ task.subject }}

### 当前子任务代码需求：
{{ task.coding_prompt }}

### 当前代码：
```python
{{ task.source }}
```

### 当前代码执行的输出与结果：
{{ task.output }}

### 当前任务总结要求：
{{ task.summary_prompt }}

---

请按要求输出任务总结：
"""


class TaskStructureSummaryState(str, Enum):
    DONE = "done"
    REQUEST_INFO = "request_info"


class TaskStructureSummaryOutput(BaseModel):

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
    request_confirm_infos: Optional[List[RequestUserSupplyInfo]] = Field(
        None, description="需要用户补充确认的信息，问题应尽量简单，只需要用户回答是/否或在备选项中选择即可"
    )


class TaskStructureSummaryAgent(BaseChatAgent):

    PROMPT = TASK_SUMMARY_PROMPT
    OUTPUT_FORMAT = AgentOutputFormat.JSON
    OUTPUT_JSON_SCHEMA = TaskStructureSummaryOutput
    DISPLAY_REPLY = True

    def on_reply(self, reply: TaskStructureSummaryOutput):
        assert reply.summary, "Reply is empty"
        _M("### 任务总结\n\n" + reply.summary)
        self.task.agent_data.issue = ""
        self.task.agent_data.result = reply.summary
        self.task.agent_data.important_infos = None
        self.task.agent_data.request_below_supply_infos = None
        if reply.important_infos:
            self.task.agent_data.important_infos = reply.important_infos
            _B(
                json.dumps(reply.important_infos, indent=4, ensure_ascii=False),
                format="code",
                code_language="json",
                title="重要信息",
            )
        if reply.request_confirm_infos:
            self.task.agent_data.request_below_supply_infos = reply.request_confirm_infos
            return TaskStructureSummaryState.REQUEST_INFO
        return TaskStructureSummaryState.DONE
