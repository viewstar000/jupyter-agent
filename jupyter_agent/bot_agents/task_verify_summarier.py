"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from IPython.display import Markdown
from .base import BaseChatAgent, AgentOutputFormat
from ..bot_outputs import _D, _I, _W, _E, _F, _M, _B, _C, _O
from ..bot_outputs import ReplyType


TASK_SUMMARY_PROMPT = """\
**角色定义**：

你是一个任务总结规划专家，能够从分析结果中提取关键结论，并对任务结果进行总结分析，给出优化建议。

**任务要求**：

- 对任务代码的执行结果进行进一步的推理分析总结，并输出**人类可读的总结**，包含以下内容：
  1. 代码执行结果总结
  1. 核心发现（如"Electronics类别月均增长12%"）
  2. 数据支撑（引用关键数值或图表）
  3. 其它建议（如新子任务Prompt等）
- 若代码的执行结果不满足当前子任务的要求，则输出**人类可读的修改建议**，包含以下内容：
  1. 当前结果不满足子任务目标的具体原因
  2. 修改后的代码生成Prompt，包括：
    - 需生成的代码类型（如数据处理、建模、可视化等）
    - 具体输入（数据、变量、参数等）
    - 预期输出形式（变量名、图表、文本等）
    - 代码执行的结果仅在当前子任务中可见，不会记录在全局上下文中
  3. 修改后的分析总结Prompt，包括：
    - 说明本子任务结果总结的要点和输出要素，以便后续子任务使用
    - 验证总结的结果会记录在全局上下文中

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

### 当前输出：
{{ task.output }}

### 当前任务总结要求：
{{ task.summary_prompt }}

---

请按要求输出验证结果：
"""


class TaskSummaryState(str, Enum):
    SUCCESS = "success"
    NOT_SATISFY = "not_satisfy"


class TaskEnhancement(BaseModel):
    issues: List[str] = Field([], description="当前子任务不满足要求的问题清单", examples=[["...", "..."]])
    code_prompt: str = Field("", description="修改后的代码生成Prompt", examples=["..."])
    summary_prompt: str = Field("", description="修改后的分析总结Prompt", examples=["..."])


class TaskSummaryOutput(BaseModel):
    state: TaskSummaryState = Field(description="是否完成总结", examples=[TaskSummaryState.SUCCESS.value])
    summary: str = Field(
        "", description=f'任务总结的详细描述，在 state="{TaskSummaryState.SUCCESS}" 时必填', examples=["..."]
    )
    enhancement: Optional[TaskEnhancement] = Field(
        None,
        description=f"任务不满足要求时的修改建议，在 state='{TaskSummaryState.NOT_SATISFY}' 时必填",
        examples=[{"issues": ["...", "..."], "code_prompt": "...", "verify_prompt": "...", "summary_prompt": "..."}],
    )


class TaskVerifySummaryAgent(BaseChatAgent):

    PROMPT = TASK_SUMMARY_PROMPT
    OUTPUT_FORMAT = AgentOutputFormat.JSON
    OUTPUT_JSON_SCHEMA = TaskSummaryOutput

    def on_reply(self, reply: TaskSummaryOutput):

        if reply.state == TaskSummaryState.SUCCESS:
            assert reply.summary, "Summary is empty"
            _M("### 任务总结\n\n" + reply.summary)
            self.task.agent_data.issue = ""
            self.task.agent_data.result = reply.summary
            return False, reply.state
        else:
            _M("### 任务验证不通过！\n")
            assert reply.enhancement, "Enhancement is empty"
            assert reply.enhancement.issues, "Issues is empty"
            assert reply.enhancement.code_prompt, "Code prompt is empty"
            assert reply.enhancement.summary_prompt, "Summary prompt is empty"
            task_issue = ""
            if reply.enhancement.issues:
                for issue in reply.enhancement.issues:
                    task_issue += "- {}\n".format(issue)
            self.task.agent_data.issue = task_issue
            self.task.agent_data.coding_prompt = reply.enhancement.code_prompt
            self.task.agent_data.summary_prompt = reply.enhancement.summary_prompt
            _M(task_issue)
            _M("### 修改后的子任务信息\n")
            _M(f"### 当前子任务代码需求：\n\n{reply.enhancement.code_prompt}")
            _M(f"### 当前子任务总结要求：\n\n{reply.enhancement.summary_prompt}")
            return True, reply.state
