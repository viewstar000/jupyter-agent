"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import time
import random

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field
from IPython.display import Markdown
from .base import BaseEvaluator
from ..bot_outputs import _D, _I, _W, _E, _F, _A, _O, _C, _M, _B
from ..bot_evaluation import FlowEvaluationRecord


PROMPT_TPL = """
{% include "TASK_AGENT" %}

---

**当前用户提交的任务目标**

{{ task.source }}

---

**当前生成的全局任务规划**

{{ task.result }}

---

{% include "TASK_TRIGGER" %}
"""
PROMPT_ROLE = """
你是一位**任务规划质量评估专家**，专精于对Jupyter Notebook任务规划结果进行量化评估。需：
1. **多维评估**：从目标一致性、逻辑完整性、可执行性等维度分析任务规划质量
2. **量化评分**：依据预设标准给出0-1分的精确评分
3. **结构化反馈**：提供可操作的具体改进建议
"""
PROMPT_RULES = """
请根据提供的任务规划结果，完成如下评估任务：

1. 判断该规划是否与用户目标高度一致（结构、顺序、内容是否贴合目标意图）
2. 对任务规划质量进行量化评分（范围为 0 到 1，小数保留一位）
3. 输出结构化的评估反馈意见，包括但不限于：
  - 是否存在逻辑错误、步骤冗余、顺序混乱、任务抽象不合理
  - 是否缺少关键步骤或前置依赖
  - 语言是否明确、可操作性是否足够强
  - 对任务执行是否具有良好引导性

评分标准建议（quality_score）

| 评分区间          | 含义说明                       |
| ------------- | -------------------------- |
| **0.90–1.00** | 规划高度合理，目标一致性强，步骤清晰完整，无明显问题 |
| **0.70–0.89** | 规划较为合理，存在轻微瑕疵但不影响整体可执行性    |
| **0.50–0.69** | 规划基本合理，存在明显问题或步骤遗漏，需部分优化   |
| **< 0.50**    | 规划质量不合格，与目标不符或结构混乱，需重构     |
"""
PROMPT_TRIGGER = "请按要求给出当前任务规划的评估结果："


class FlowGlobalPlanningEvalResult(BaseModel):
    """
    任务规划评估结果
    """

    is_correct: bool = Field(description="任务规划是否与用户目标一致", examples=[True, False])
    quality_score: float = Field(
        description="任务规划质量评分，任务规划是否符合用户目标要求，是否是完整、详细、准确的步骤说明，"
        "是否存在逻辑错误、冗余、抽象不合理等情况，范围0-1，>=0.5表示符合要求，<0.5表示不符合要求",
        examples=[round(random.uniform(0.5, 1.0), 2), round(random.uniform(0.0, 0.5), 2)],
    )
    feedback: str = Field(description="评估反馈", examples=["任务规划符合要求，但..."])


class EvaluationResult(BaseModel):
    """
    任务规划评估结果
    """

    description: str = Field(description="评估任务描述", examples=["任务规划评估结果"])
    properties: FlowGlobalPlanningEvalResult = Field(
        description="评估任务具体结果",
        examples=[
            FlowGlobalPlanningEvalResult(
                is_correct=True, quality_score=round(random.uniform(0.5, 1.0), 2), feedback="任务规划符合要求, 但..."
            )
        ],
    )


class FlowGlobalPlanningEvaluator(BaseEvaluator):
    """
    任务规划评估器
    """

    PROMPT_TPL = PROMPT_TPL
    PROMPT_ROLE = PROMPT_ROLE
    PROMPT_RULES = PROMPT_RULES
    PROMPT_TRIGGER = PROMPT_TRIGGER
    OUTPUT_JSON_SCHEMA = EvaluationResult

    def on_reply(self, reply: EvaluationResult):
        reply = super().on_reply(reply)
        return FlowEvaluationRecord(
            timestamp=time.time(),
            evaluator="flow_global_planning",
            correct_score=reply.properties.quality_score,
        )
