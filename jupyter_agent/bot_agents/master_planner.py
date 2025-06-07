"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from IPython.display import Markdown
from .base import BaseTaskAgent, AGENT_MODEL_TYPE_PLANNER
from ..utils import REPLY_TASK_RESULT

MASTER_PLANNER_PROMPT = """\
**角色定义**：

你是一个高级分析规划专家，擅长将自然语言任务拆解为可执行的Jupyter Notebook分析流程。

**任务要求**：

- 解析用户输入的自然语言指令（目标Prompt），提取核心需求（如数据来源、分析目标、输出格式等）
- 将核心需求转化为可执行的Jupyter Notebook分析流程
- 流程的每个环节应尽量简单明确
- 生成**目标规划说明**，包含：
  1. 需要执行的子任务列表（按逻辑顺序排列）
  3. 每个子任务的输入/输出依赖关系
  4. 预期的最终输出形式（文字/图表/表格等）

**用户需求**：

{{ task.cell_code }}

---

请根据上述信息为用户规划全局执行计划：
"""


class MasterPlannerAgent(BaseTaskAgent):
    """全局规划器代理类"""

    PROMPT = MASTER_PLANNER_PROMPT
    DISPLAY_REPLY = False
    MODEL_TYPE = AGENT_MODEL_TYPE_PLANNER

    def on_reply(self, reply):
        self._C(Markdown(reply), reply_type=REPLY_TASK_RESULT)
