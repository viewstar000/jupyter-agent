"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field
from IPython.display import Markdown
from .base import BaseChatAgent, AgentOutputFormat, AgentModelType
from ..bot_outputs import ReplyType, _D, _I, _W, _E, _F, _A, _O, _C, _M, _B
from ..bot_actions import RequestUserSupplyInfo


PROMPT_ROLE = """
你是一个任务规划专家，负责根据全局目标规划，需要将一个复杂的Jupyter Notebook编程任务分解成若干步骤(Cell子任务)，
并逐步推进子任务的执行。
"""
PROMPT_RULES = """
- 审查全局目标与已完成的cell子任务的结果，判断是否已实现整体目标，若全局目标已达成，终止流程并输出最终结果。
- 若全局目标未达成，请根据全局目标规划和已完成的cell子任务，制定下一个**Cell子任务**的执行计划，包括：
  - 首先拆解出Cell子任务的工作描述，包括子任务目标、输入与输出约束等
  - 然后跟据子任务目标的特点，选择合适的子任务执行方式
    - LLM直接推理模式：
      - 若子任务的目标可以直接通过推理实现，则直接能过推理分析完成子任务，输出任务结果后结束当前子任务的执行
      - 该模式通常适用于简单数据的比对、数据模型设计、数据模型比对、策略生成、报告生成等任务场景
    - 代码执行+LLM分析总结模式：
      - 若子任务的目标需要通过执行代码并对执行结果进行推理分析总结的方式实现，则协调代码生成Agent、LLM分析总结Agent共同完成当前子任务
      - 该模式通常适用于数据加截、预览、清洗、统计、可视化、复杂算法等任务场景
      - 此时应结合代码执行Agent与LLM分析总结agent的能力的优劣，进行合理的分工并生成相应的Prompt
        - 代码生成Prompt，提供给代码生成Agent的Prompt包括：
          - 需生成的代码类型（如数据处理、建模、可视化等）
          - 具体输入（数据、变量、参数等）
          - 预期输出形式（变量名、图表、文本等）
          - 代码执行的结果仅在当前子任务中可见，不会记录在全局上下文中
        - 分析总结Prompt，提供给LLM分析总结Agent的Prompt包括：
          - 说明本子任务结果分析总结的要点和输出要素，以便后续子任务使用
          - 验证总结的结果会记录在全局上下文中
    - 若需要用户提供更多的信息才能完成子任务，给出详细的提示信息
  - 子任务的划分应严格遵守全局目标规划说明的要求
  - 在引用已完成的子任务的结果时，特别是important_infos中的信息，要保证准确、清晰、完整，不要出现任何误导信息
  - 对于用户已补充确认的信息，特别是user_supply_infos中的信息，要充分利用，不要出现任何遗漏、冲突、误导、反复确认的情况
  - 子任务代码执行的结果不会记录在全局上下文中，只有LLM直接推理或LLM分析总结的结果会记录在全局上下文中以支持后续子任务的执行
"""


class TaskPlannerState(Enum):
    CODING_PLANNED = "coding_planned"  # 任务规划完成
    REASONING_PLANNED = "reasoning_planned"  # 任务规划完成
    REQUEST_INFO = "request_info"  # 需要用户补充更详细的信息
    GLOBAL_FINISHED = "global_finished"  # 全局目标已达成


class TaskPlannerOutput(BaseModel):
    state: TaskPlannerState = Field(
        description=(
            "任务规划状态: "
            f"{TaskPlannerState.CODING_PLANNED}, 完成规划, 使用代码执行+LLM分析总结模式; "
            f"{TaskPlannerState.REASONING_PLANNED}, 完成规划, 使用LLM直接推理模式; "
            f"{TaskPlannerState.REQUEST_INFO}, 需要用户补充更详细的信息; "
            f"{TaskPlannerState.GLOBAL_FINISHED}, 全局目标是否已达成, 无需继续规划; "
        ),
        examples=[TaskPlannerState.CODING_PLANNED.value],
    )
    subtask_id: str = Field(
        "",
        description=f'子任务全局唯一ID，在 state="{TaskPlannerState.CODING_PLANNED}" '
        f'或state="{TaskPlannerState.REASONING_PLANNED}" 时必填',
        examples=["task-xxx"],
    )
    subtask_subject: str = Field(
        "",
        description=(
            f'子任务工作描述, 在 state="{TaskPlannerState.CODING_PLANNED}"'
            f'或state="{TaskPlannerState.REASONING_PLANNED}" 时必填'
        ),
        examples=["对...进行...处理，输出..."],
    )
    subtask_coding_prompt: str = Field(
        "",
        description=f'代码生成Prompt, 在 state="{TaskPlannerState.CODING_PLANNED}" 时必填',
        examples=["请基于...，计算...，并保存结果为..."],
    )
    subtask_summary_prompt: str = Field(
        "",
        description=(
            f'结果总结Prompt, 在 state="{TaskPlannerState.CODING_PLANNED}"'
            f'或state="{TaskPlannerState.REASONING_PLANNED}" 时必填'
        ),
        examples=["请对当前任务的结果进行总结，输出以下要素：..."],
    )
    request_supply_infos: Optional[List[RequestUserSupplyInfo]] = Field(
        None, description=f'需要用户补充更详细的信息的 Prompt, 在 state="{TaskPlannerState.REQUEST_INFO}" 时必填'
    )


class TaskPlannerAgentV3(BaseChatAgent):
    """任务规划器代理类"""

    PROMPT_ROLE = PROMPT_ROLE
    PROMPT_RULES = PROMPT_RULES
    OUTPUT_FORMAT = AgentOutputFormat.JSON
    OUTPUT_JSON_SCHEMA = TaskPlannerOutput
    MODEL_TYPE = AgentModelType.PLANNER

    def get_task_data(self):
        if self.task.subject and self.task.issue:
            return {
                "cell_idx": self.task.cell_idx,
                "task_id": self.task.task_id,
                "subject": self.task.subject,
                "issue": self.task.issue,
                "coding_prompt": self.task.coding_prompt,
                "source": self.task.source,
                "output": self.task.output,
            }
        else:
            return {}

    def get_trigger_prompt(self):
        if self.task.subject and self.task.issue:
            return "请参考上述信息重新规划当前子任务："
        else:
            return "请按要求规划下一个子任务："

    def on_reply(self, reply: TaskPlannerOutput):
        """执行规划逻辑"""
        self.task.agent_data.result = ""
        self.task.agent_data.coding_prompt = ""
        self.task.agent_data.summary_prompt = ""
        self.task.agent_data.important_infos = None
        self.task.agent_data.request_above_supply_infos = None
        self.task.agent_data.request_below_supply_infos = None
        if reply.state == TaskPlannerState.GLOBAL_FINISHED:
            _C(Markdown("全局目标已达成，任务完成！"), reply_type=ReplyType.TASK_RESULT)
            return reply.state
        elif reply.state == TaskPlannerState.REQUEST_INFO:
            assert reply.request_supply_infos, "Request info prompt is empty"
            self.task.agent_data.request_above_supply_infos = reply.request_supply_infos
            return reply.state
        elif reply.state == TaskPlannerState.CODING_PLANNED:
            assert reply.subtask_id, "Subtask id is empty"
            assert reply.subtask_subject, "Subtask subject is empty"
            assert reply.subtask_coding_prompt, "Subtask coding prompt is empty"
            assert reply.subtask_summary_prompt, "Subtask summary prompt is empty"
            _M(
                f"### 子任务: {reply.subtask_subject}\n"
                f"- ID: {reply.subtask_id}\n"
                f"- Coding: {reply.subtask_coding_prompt}\n"
                f"- Summary: {reply.subtask_summary_prompt}\n"
            )
            self.task.agent_data.task_id = reply.subtask_id
            self.task.agent_data.subject = reply.subtask_subject
            self.task.agent_data.coding_prompt = reply.subtask_coding_prompt
            self.task.agent_data.summary_prompt = reply.subtask_summary_prompt
            return reply.state
        elif reply.state == TaskPlannerState.REASONING_PLANNED:
            assert reply.subtask_id, "Subtask id is empty"
            assert reply.subtask_subject, "Subtask subject is empty"
            assert reply.subtask_summary_prompt, "Subtask summary prompt is empty"
            _M(
                f"### 子任务: {reply.subtask_subject}\n"
                f"- ID: {reply.subtask_id}\n"
                f"- Reasoning: {reply.subtask_summary_prompt}\n"
            )
            self.task.agent_data.task_id = reply.subtask_id
            self.task.agent_data.subject = reply.subtask_subject
            self.task.agent_data.summary_prompt = reply.subtask_summary_prompt
            return reply.state
        else:
            raise ValueError(f"Unknown task planner state: {reply.state}")
