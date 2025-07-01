"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from enum import Enum
from typing import List
from pydantic import BaseModel, Field
from IPython.display import Markdown
from .base import BaseChatAgent, AgentOutputFormat
from ..bot_outputs import _D, _I, _W, _E, _F, _M, _B, _C, _O
from ..bot_outputs import ReplyType


TASK_VERIFY_PROMPT = """\
**角色定义**：

你是一个数据质量检查员，负责验证子任务的输出与结果的正确性。

**任务要求**：

- 对比子任务Prompt的预期输出和实际结果，验证以下维度：  
  1. 数据完整性（如无缺失值、数据量合理）  
  2. 逻辑一致性（如增长率计算正确）  
- 输出验证结果和改进建议（如需要重新运行子任务则标记为失败）  

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

### 当前任务验证条件：
{{ task.verify_prompt }}

---

请按要求输出验证结果：
"""


class TaskVerifyState(Enum):
    FAILED = "failed"
    PASSED = "passed"


class TaskVerifyOutput(BaseModel):
    state: TaskVerifyState = Field(description="任务验证结果", examples=[TaskVerifyState.PASSED.value])
    issues: List[str] = Field(
        [],
        description="任务验证失败问题清单, 任务验证失败时必填, 任务验证通过时返回空列表",
        examples=[["...未包含...字段...", "..字段值缺失...", "...字段值未在合理范围内...", "..."]],
    )


class TaskVerifyAgent(BaseChatAgent):

    PROMPT = TASK_VERIFY_PROMPT
    OUTPUT_FORMAT = AgentOutputFormat.JSON
    OUTPUT_JSON_SCHEMA = TaskVerifyOutput

    def on_reply(self, reply: TaskVerifyOutput):

        if reply.state == TaskVerifyState.PASSED:
            _M("### 任务验证通过！")
            self.task.agent_data.issue = ""
            return False, reply.state
        else:
            _M("### 任务验证不通过！\n")
            task_issue = ""
            if reply.issues:
                for issue in reply.issues:
                    task_issue += "- {}\n".format(issue)
            _M(task_issue)
            self.task.agent_data.issue = task_issue
            return True, reply.state
