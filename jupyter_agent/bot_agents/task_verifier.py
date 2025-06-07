"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from enum import Enum
from typing import List
from pydantic import BaseModel, Field
from IPython.display import Markdown
from ..utils import REPLY_TASK_ISSUE
from .base import BaseTaskAgent, AGENT_OUTPUT_FORMAT_JSON


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
{{ task.task_subject }}

### 当前子任务代码需求：
{{ task.task_coding_prompt }}

### 当前代码：
```python
{{ task.cell_code }}
```

### 当前输出：
{{ task.cell_output }}
{{ task.cell_result }}

### 当前任务验证条件：
{{ task.task_verify_prompt }}

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


class TaskVerifyAgent(BaseTaskAgent):

    PROMPT = TASK_VERIFY_PROMPT
    OUTPUT_FORMAT = AGENT_OUTPUT_FORMAT_JSON
    OUTPUT_JSON_SCHEMA = TaskVerifyOutput

    def on_reply(self, reply: TaskVerifyOutput):

        if reply.state == TaskVerifyState.PASSED:
            self._D(Markdown("### 任务验证通过！"))
            return False, reply.state
        else:
            self._D(Markdown("### 任务验证不通过！"))
            self.task_context.task_issue = ""
            if reply.issues:
                for issue in reply.issues:
                    self.task_context.task_issue += "- {}\n".format(issue)
            self._D(Markdown(self.task_context.task_issue), reply_type=REPLY_TASK_ISSUE)
            return True, reply.state
