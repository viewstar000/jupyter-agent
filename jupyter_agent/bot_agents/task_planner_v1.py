"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from enum import Enum
from pydantic import BaseModel, Field
from IPython.display import Markdown
from .base import BaseChatAgent, AgentOutputFormat, AgentModelType
from ..bot_outputs import _D, _I, _W, _E, _F, _M, _B, _C, _O
from ..bot_outputs import ReplyType


TASK_PLANNER_PROMPT = """\
**角色定义**：

你是一名任务调度专家，负责根据全局分析规划，逐步推进并协调各子任务的执行。

**任务要求**：

- 审查全局目标与已完成的子任务结果，判断是否已实现整体目标：
  - 若目标已达成，终止流程并输出最终结果。
  - 若目标未达成，请根据目标规划说明和已完成子任务，制定下一个**子任务**的执行计划
    - 子任务的划分应严格遵守全局目标规划说明的要求
    - 协调代码生成Agent、结果验证Agent、结果总结Agent共同完成当前子任务
    - 子任务的执行计划具体包括：
      1. 子任务工作描述：
          - 简明阐述子任务目标、输入与输出约束
      2. 子任务代码生成Prompt：
          - 需生成的代码类型（如数据处理、建模、可视化等）
          - 具体输入（数据、变量、参数等）
          - 预期输出形式（变量名、图表、文本等）
      3. 子任务结果验证Prompt：
          - 检测子任务代码执行结果是否符合当前子任务的要求
          - 检测条件只考虑当前子任务的要求，不要考虑全局目标
      4. 子任务结果总结Prompt：
          - 说明本子任务结果总结的要点和输出要素，以便后续子任务使用
  - 若需要用户提供更多的信息，给出详细的提示信息

{% include "TASK_OUTPUT_FORMAT" %}

---

{% include "TASK_CONTEXTS" %}

---

{% include "CODE_CONTEXTS" %}

---

{% if task.subject and task.issue %}
**当前子任务信息**:

### 当前子任务目标：
{{ task.subject }}

### 当前子任务代码：
```python
{{ task.source }}
```

### 当前子任务输出：
{{ task.output }}

### 当前子任务存在的问题：
{{ task.issue }}

---

请参考上述信息重新规划当前子任务：
{% else %}
请按要求规划下一个子任务：
{% endif %}

"""


class TaskPlannerState(Enum):
    PLANNED = "planned"  # 任务规划完成
    REQUEST_INFO = "request_info"  # 需要用户补充更详细的信息
    GLOBAL_FINISHED = "global_finished"  # 全局目标已达成


class TaskPlannerOutput(BaseModel):
    state: TaskPlannerState = Field(
        description=(
            "任务规划状态: "
            f"{TaskPlannerState.PLANNED}, 完成规划, 可以开始执行下一步; "
            f"{TaskPlannerState.REQUEST_INFO}, 需要用户补充更详细的信息; "
            f"{TaskPlannerState.GLOBAL_FINISHED}, 全局目标是否已达成, 无需继续规划; "
        ),
        examples=[TaskPlannerState.PLANNED.value],
    )
    subtask_subject: str = Field(
        "",
        description='子任务简要描述, 在 state="planned" 时必填',
        examples=["对...进行...处理，输出..."],
    )
    subtask_coding_prompt: str = Field(
        "",
        description='子任务代码生成Prompt, 在 state="planned" 时必填',
        examples=["请基于...，计算...，并保存结果为..."],
    )
    subtask_verify_prompt: str = Field(
        "",
        description='子任务结果验证Prompt, 在 state="planned" 时必填',
        examples=["请验证当前任务结果是否符合以下条件：..."],
    )
    subtask_summary_prompt: str = Field(
        "",
        description='子任务结果总结Prompt, 在 state="planned" 时必填',
        examples=["请对当前任务的结果进行总结，输出以下要素：..."],
    )
    request_info_prompt: str = Field(
        "",
        description='需要用户补充更详细的信息的 Prompt, 在 state="request_info" 时必填',
        examples=["请补充与...相关的详细的信息"],
    )


class TaskPlannerAgentV1(BaseChatAgent):
    """任务规划器代理类"""

    PROMPT = TASK_PLANNER_PROMPT
    OUTPUT_FORMAT = AgentOutputFormat.JSON
    OUTPUT_JSON_SCHEMA = TaskPlannerOutput
    MODEL_TYPE = AgentModelType.PLANNER

    def on_reply(self, reply: TaskPlannerOutput):
        """执行规划逻辑"""
        if reply.state == TaskPlannerState.GLOBAL_FINISHED:
            _C(Markdown("全局目标已达成，任务完成！"), reply_type=ReplyType.TASK_RESULT)
            return False, reply.state
        elif reply.state == TaskPlannerState.REQUEST_INFO:
            assert reply.request_info_prompt, "Request info prompt is empty"
            _O(Markdown(f"### 需要补充更详细的信息\n\n{reply.request_info_prompt}"), reply_type=ReplyType.TASK_ISSUE)
            return True, reply.state
        elif reply.state == TaskPlannerState.PLANNED:
            assert reply.subtask_subject, "Subtask subject is empty"
            assert reply.subtask_coding_prompt, "Subtask coding prompt is empty"
            assert reply.subtask_verify_prompt, "Subtask verify prompt is empty"
            assert reply.subtask_summary_prompt, "Subtask summary prompt is empty"
            _M(
                f"### 子任务: {reply.subtask_subject}\n"
                f"- Coding: {reply.subtask_coding_prompt}\n"
                f"- Verify: {reply.subtask_verify_prompt}\n"
                f"- Summary: {reply.subtask_summary_prompt}\n"
            )
            self.task.set_data("subject", reply.subtask_subject)
            self.task.set_data("coding_prompt", reply.subtask_coding_prompt)
            self.task.set_data("verify_prompt", reply.subtask_verify_prompt)
            self.task.set_data("summary_prompt", reply.subtask_summary_prompt)
            return False, reply.state
        else:
            raise ValueError(f"Unknown task planner state: {reply.state}")
