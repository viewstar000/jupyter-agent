"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import time

from .base import BaseChatAgent, AgentOutputFormat, AgentModelType


TASK_CODING_PROMPT = """\
**角色定义**：

你是一个代码生成器，能够将自然语言描述转化为可执行的Jupyter Notebook代码。

**任务要求**：

- 根据子任务Prompt生成**Python代码**（需包含必要的依赖导入）
- 代码需严格遵守以下原则：
  - 使用已存在的变量（如`cleaned_sales_df`）
  - 新生成的变量需命名清晰（如`growth_rates_df`）
  - 发生错误时，直接抛出异常，不需要进行任何错误处理，以便于执行器发现问题
  - 代码在Jupyter环境中执行，但也应考虑通用性，尽量封装为函数或类
  - 执行结果应保存在变量中，同时打印输出（或做为cell的返回值输出）
  - 代码中应保含必要的文档注释
  - 避免生成重复的代码

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

{% if task.issue %}
### 尽量避免的问题：  
{{ task.issue }}
{% endif %}

---

请按要求生成任务代码：
"""


class TaskCodingAgent(BaseChatAgent):

    PROMPT = TASK_CODING_PROMPT
    OUTPUT_FORMAT = AgentOutputFormat.CODE
    OUTPUT_CODE_LANG = "python"
    MODEL_TYPE = AgentModelType.CODING

    def on_reply(self, reply: str):
        generated_code = "# Generated by Jupyter Agent (Coder) {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"))
        generated_code += reply
        self.task.source = generated_code
