"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import json

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .base import BaseChatAgent, AgentOutputFormat
from ..bot_outputs import _D, _I, _W, _E, _F, _M, _B, _C, _O
from ..bot_actions import RequestUserSupplyInfo


PROMPT_ROLE = """
你是一个信息提炼专家，能够从分析结果中提取关键结论。
"""
PROMPT_RULES = """
- 将代码执行的输出与结果转化为**人类可读的总结**
- 包含以下内容：
  1. 代码执行结果总结
  2. 核心发现（如"Electronics类别月均增长12%"）
  3. 数据支撑（引用关键数值或图表）
  4. 其它建议（如新子任务Prompt等）
- 在引用其它已完成的子任务的结果时，特别是其important_infos中的信息，要保证准确、清晰、完整，不要出现任何误导信息
- 对于用户已提供的补充信息，特别是user_supply_infos中的信息，要充分利用，不要出现任何遗漏、冲突、误导、反复确认的情况

注：任务代码执行的结果不会记录在全局上下文中，只有任务总结的结果会记录在全局上下文中，
因此任务总结中应包含对代码执行结果的简要说明，以便后续子任务使用。
"""
PROMPT_TRIGGER = """
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

    PROMPT_ROLE = PROMPT_ROLE
    PROMPT_RULES = PROMPT_RULES
    PROMPT_TRIGGER = PROMPT_TRIGGER
    OUTPUT_FORMAT = AgentOutputFormat.JSON
    OUTPUT_JSON_SCHEMA = TaskStructureSummaryOutput
    DISPLAY_REPLY = True

    def get_task_data(self):
        return {
            "cell_idx": self.task.cell_idx,
            "task_id": self.task.task_id,
            "subject": self.task.subject,
            "coding_prompt": self.task.coding_prompt,
            "source": self.task.source,
            "output": self.task.output,
            "summary_prompt": self.task.summary_prompt,
        }

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
