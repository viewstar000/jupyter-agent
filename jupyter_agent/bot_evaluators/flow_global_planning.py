"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import time

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field
from IPython.display import Markdown
from .base import BaseEvaluator
from ..bot_outputs import _D, _I, _W, _E, _F, _A, _O, _C, _M, _B
from ..bot_evaluation import FlowEvaluationRecord


FLOW_GLOBAL_PLANNING_EVAL_PROMPT = """\
**角色定义**：

你是一个任务规划评估专家，负责对任务规划的结果进行评估。

**任务要求**：

请你根据任务规划的结果，评估任务规划的质量和准确性，并给出相应的评分和反馈。

{% include "TASK_OUTPUT_FORMAT" %}

---

**当前用户提交的任务目标**

{{ task.source }}

---

**当前生成的全局任务规划**

{{ task.result }}

---

请按要求给出当前任务规划的评估结果：
"""


class FlowGlobalPlanningEvalResult(BaseModel):
    """
    任务规划评估结果
    """

    is_correct: bool = Field(description="任务规划是否与用户目标一致", examples=[True, False])
    quality_score: float = Field(
        description="任务规划质量评分，任务规划是否符合用户目标要求，是否是完整、详细、准确的步骤说明，"
        "是否存在逻辑错误、冗余、抽象不合理等情况，范围0-1，>=0.5表示符合要求，<0.5表示不符合要求",
        examples=[0.8, 0.3],
    )
    feedback: Optional[str] = Field(default=None, description="评估反馈")


class EvaluationResult(BaseModel):
    """
    任务规划评估结果
    """

    description: str = Field(description="评估任务描述", examples=["任务规划评估结果"])
    properties: FlowGlobalPlanningEvalResult = Field(
        description="评估任务具体结果",
        examples=[FlowGlobalPlanningEvalResult(is_correct=True, quality_score=0.8, feedback="任务规划符合要求")],
    )


class FlowGlobalPlanningEvaluator(BaseEvaluator):
    """
    任务规划评估器
    """

    PROMPT_TPL = FLOW_GLOBAL_PLANNING_EVAL_PROMPT
    OUTPUT_JSON_SCHEMA = EvaluationResult

    def on_reply(self, reply):
        reply = super().on_reply(reply)
        return FlowEvaluationRecord(
            timestamp=time.time(),
            evaluator="flow_global_planning",
            correct_score=reply.properties.quality_score,
        )
