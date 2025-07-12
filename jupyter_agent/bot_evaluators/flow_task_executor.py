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


PROMPT_ROLE = """
你是一位严谨的任务执行评估专家，擅长从多维度对子任务的执行结果进行结构化评分。
你将从全局规划一致性、推理合理性、代码实现质量和信息利用完整性等方面对每一个子任务进行细致评估，确保整个任务链的质量闭环。
"""
PROMPT_RULES = """
请你根据当前子任务的执行结果，从以下七个维度出发，对其规划质量、代码实现、逻辑推理与信息引用的完整性进行准确评估，输出结构化评分结果和评估反馈。

评估维度说明

| 维度                    | 说明                                                 | 合格评分范围       |
| ---------------------- | -------------------------------------------------- | ------------ |
| is_correct             | 当前子任务最终结果是否达成其目标                                   | true / false |
| correct_score          | 子任务输出与其目标的匹配程度；是否得到预期结果                            | 0.0–1.0      |
| planning_score         | 子任务目标是否清晰，规划是否合理，是否与全局目标一致                         | 0.0–1.0      |
| reasoning_score        | 推理过程是否合理，是否有逻辑跳跃或与前序任务冲突                           | 0.0–1.0      |
| coding_score           | 代码实现是否符合规划目标，是否存在冗余或设计不当                           | 0.0–1.0      |
| important_info_score   | 是否充分引用和利用了前置任务产出的 `important_infos` 信息，是否完整、准确、无歧义 | 0.0–1.0      |
| user_supply_info_score | 是否充分考虑了用户通过 `user_supply_infos` 补充的信息，是否存在遗漏或逻辑冲突  | 0.0–1.0      |
"""
PROMPT_TRIGGER = "请按要求给出当前子任务规划的评估结果："


class FlowTaskExecEvalResult(BaseModel):
    """
    任务规划评估结果
    """

    is_correct: bool = Field(description="最终结果是否符合当前子任务的目标", examples=[True, False])
    correct_score: float = Field(
        description="最终结果符合当前子任务目标的分数，范围0-1，>=0.5表示符合目标，<0.5表示不符合目标",
        examples=[round(random.uniform(0.5, 1.0), 2), round(random.uniform(0.0, 0.5), 2)],
    )
    correct_score_feedback: str = Field(
        description="针对correct_score评估反馈", examples=["最终结果符合目标要求， 但..."]
    )
    planning_score: float = Field(
        description="当前子任务的目标规划、代码生成、总结是否符合全局目标规划要求，范围0-1，>=0.5表示符合要求，<0.5表示不符合要求",
        examples=[round(random.uniform(0.5, 1.0), 2), round(random.uniform(0.0, 0.5), 2)],
    )
    planning_score_feedback: str = Field(
        description="针对planning_score评估反馈", examples=["任务规划符合要求， 但..."]
    )
    reasoning_score: float = Field(
        description="当前子任务的推理过程是否合理，是否存在逻辑错误，是否存在与前置子任务相冲突的情况，"
        "范围0-1，>=0.5表示合理、正确、无冲突，<0.5表示不合理",
        examples=[round(random.uniform(0.5, 1.0), 2), round(random.uniform(0.0, 0.5), 2)],
    )
    reasoning_score_feedback: str = Field(description="针对reasoning_score评估反馈", examples=["推理过程合理， 但..."])
    coding_score: float = Field(
        description="代码生成的质量评分，代码逻辑是否符合规划要求，是否存在逻辑错误，是否存在冗余、抽象不合理等情况，"
        "范围0-1，>=0.5表示代码质量较高，<0.5表示代码质量较低",
        examples=[round(random.uniform(0.5, 1.0), 2), round(random.uniform(0.0, 0.5), 2)],
    )
    coding_score_feedback: str = Field(description="针对coding_score评估反馈", examples=["代码质量较高， 但..."])
    important_info_score: float = Field(
        description="重要信息分数，当前子任务的规划、代码生成、总结是否充分考虑了前置任务生成的重要信息，"
        "以及当前子任务的重要信息是否完整、准确、无误导、无冲突，"
        "范围0-1，>=0.5表示重要信息完整、准确，<0.5表示重要信息不完整或不准确",
        examples=[round(random.uniform(0.5, 1.0), 2), round(random.uniform(0.0, 0.5), 2)],
    )
    important_info_score_feedback: str = Field(
        description="针对important_info_score评估反馈", examples=["重要信息完整， 但..."]
    )
    user_supply_info_score: float = Field(
        description="用户补充信息分数，当前子任务的规划、代码生成、总结是否充分考虑了用户补充的信息，"
        "范围0-1，>=0.5表示充分考虑，<0.5表示未充分考虑",
        examples=[round(random.uniform(0.5, 1.0), 2), round(random.uniform(0.0, 0.5), 2)],
    )
    user_supply_info_score_feedback: str = Field(
        description="针对user_supply_info_score评估反馈", examples=["充分考虑， 但..."]
    )


class EvaluationResult(BaseModel):
    """
    任务规划评估结果
    """

    description: str = Field(description="评估任务描述", examples=["任务规划评估结果"])
    properties: FlowTaskExecEvalResult = Field(
        description="评估任务具体结果",
        examples=[
            FlowTaskExecEvalResult(
                is_correct=True,
                correct_score=round(random.uniform(0.5, 1.0), 2),
                correct_score_feedback="结果存在...",
                planning_score=round(random.uniform(0.5, 1.0), 2),
                planning_score_feedback="任务规划基本符合要求， 但...",
                reasoning_score=round(random.uniform(0.5, 1.0), 2),
                reasoning_score_feedback="推理过程...， 存在...",
                coding_score=round(random.uniform(0.5, 1.0), 2),
                coding_score_feedback="代码质量...， 结果...",
                important_info_score=round(random.uniform(0.5, 1.0), 2),
                important_info_score_feedback="重要信息引用...， 但...",
                user_supply_info_score=round(random.uniform(0.5, 1.0), 2),
                user_supply_info_score_feedback="充分考虑...， 但...",
            )
        ],
    )


class FlowTaskExecEvaluator(BaseEvaluator):
    """
    任务规划评估器
    """

    PROMPT_ROLE = PROMPT_ROLE
    PROMPT_RULES = PROMPT_RULES
    PROMPT_TRIGGER = PROMPT_TRIGGER
    OUTPUT_JSON_SCHEMA = EvaluationResult

    def on_reply(self, reply: EvaluationResult):
        reply = super().on_reply(reply)
        return FlowEvaluationRecord(
            timestamp=time.time(),
            evaluator="flow_task_executor",
            correct_score=reply.properties.correct_score,
            planning_score=reply.properties.planning_score,
            reasoning_score=reply.properties.reasoning_score,
            coding_score=reply.properties.coding_score,
            important_score=reply.properties.important_info_score,
            user_supply_score=reply.properties.user_supply_info_score,
        )
