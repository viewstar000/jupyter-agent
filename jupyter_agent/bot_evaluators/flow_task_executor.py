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


PROMPT_ROLE = """
你是一个任务规划评估专家，负责对任务规划的结果进行评估。
"""
PROMPT_RULES = """
请你根据任务规划的结果，评估任务规划的质量和准确性，并给出相应的评分和反馈。
"""
PROMPT_TRIGGER = "请按要求给出当前子任务规划的评估结果："


class FlowTaskExecEvalResult(BaseModel):
    """
    任务规划评估结果
    """

    is_correct: bool = Field(description="最终结果是否符合当前子任务的目标", examples=[True, False])
    correct_score: float = Field(
        description="最终结果符合当前子任务目标的分数，范围0-1，>=0.5表示符合目标，<0.5表示不符合目标",
        examples=[0.95, 0.3],
    )
    planning_score: float = Field(
        description="当前子任务的目标规划、代码生成、总结是否符合全局目标规划要求，范围0-1，>=0.5表示符合要求，<0.5表示不符合要求",
        examples=[0.85, 0.25],
    )
    reasoning_score: float = Field(
        description="当前子任务的推理过程是否合理，是否存在逻辑错误，是否存在与前置子任务相冲突的情况，"
        "范围0-1，>=0.5表示合理、正确、无冲突，<0.5表示不合理",
        examples=[0.9, 0.4],
    )
    coding_score: float = Field(
        description="代码生成的质量评分，代码逻辑是否符合规划要求，是否存在逻辑错误，是否存在冗余、抽象不合理等情况，"
        "范围0-1，>=0.5表示代码质量较高，<0.5表示代码质量较低",
        examples=[0.75, 0.2],
    )
    important_info_score: float = Field(
        description="重要信息分数，当前子任务的规划、代码生成、总结是否充分考虑了前置任务生成的重要信息，"
        "以及当前子任务的重要信息是否完整、准确、无误导、无冲突，"
        "范围0-1，>=0.5表示重要信息完整、准确，<0.5表示重要信息不完整或不准确",
        examples=[0.9, 0.4],
    )
    user_supply_info_score: float = Field(
        description="用户补充信息分数，当前子任务的规划、代码生成、总结是否充分考虑了用户补充的信息，"
        "范围0-1，>=0.5表示充分考虑，<0.5表示未充分考虑",
        examples=[0.8, 0.3],
    )
    feedback: Optional[str] = Field(default=None, description="评估反馈")


class FlowTaskExecEvaluator(BaseEvaluator):
    """
    任务规划评估器
    """

    PROMPT_ROLE = PROMPT_ROLE
    PROMPT_RULES = PROMPT_RULES
    PROMPT_TRIGGER = PROMPT_TRIGGER
    OUTPUT_JSON_SCHEMA = FlowTaskExecEvalResult

    def on_reply(self, reply: FlowTaskExecEvalResult):
        reply = super().on_reply(reply)
        return FlowEvaluationRecord(
            timestamp=time.time(),
            evaluator="flow_task_executor",
            correct_score=reply.correct_score,
            planning_score=reply.planning_score,
            reasoning_score=reply.reasoning_score,
            coding_score=reply.coding_score,
            important_score=reply.important_info_score,
            user_supply_score=reply.user_supply_info_score,
        )
