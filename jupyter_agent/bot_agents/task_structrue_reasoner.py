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


TASK_REASONER_PROMPT = """\
**角色定义**：

你是一个推理分析与信息提炼专家，能够从已有的数据、结果中推理分析并提取出关键结论。

**任务要求**：

- 在已有的数据、结果中进行推理分析，按需提取关键结论，并将结论输出为**人类可读的总结**
- 包含以下内容：
  1. 核心发现（如"Electronics类别月均增长12%"）
  2. 数据支撑（引用关键数值或图表）
  3. 其它建议（如新子任务Prompt等）
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

### 当前任务总结要求：
{{ task.summary_prompt }}

---

请按要求输出任务结论：
"""


class TaskStructureReasonState(str, Enum):
    DONE = "done"
    REQUEST_INFO = "request_info"


class TaskStructureReasonOutput(BaseModel):

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


class TaskStructureReasoningAgent(BaseChatAgent):

    PROMPT = TASK_REASONER_PROMPT
    OUTPUT_FORMAT = AgentOutputFormat.JSON
    OUTPUT_JSON_SCHEMA = TaskStructureReasonOutput
    DISPLAY_REPLY = True

    def on_reply(self, reply: TaskStructureReasonOutput):
        assert reply.summary, "Reply is empty"
        _M("### 任务总结\n\n" + reply.summary)
        self.task.agent_data.result = reply.summary
        if reply.important_infos:
            self.task.agent_data.important_infos = reply.important_infos
            _B(
                json.dumps(reply.important_infos, indent=4, ensure_ascii=False),
                title="重要信息",
                format="code",
                code_language="json",
            )
        if reply.request_confirm_infos:
            self.task.agent_data.request_below_supply_infos = reply.request_confirm_infos
            return TaskStructureReasonState.REQUEST_INFO
        return TaskStructureReasonState.DONE
