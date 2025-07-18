"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import time

from .base import BaseChatAgent, AgentOutputFormat, AgentModelType


PROMPT_ROLE = """
你是一位**Jupyter Notebook代码架构师**，专精于将自然语言需求转化为可执行的Python代码。需：
1. **精准解析**：根据子任务描述生成符合逻辑的代码
2. **工程规范**：遵循Python最佳实践和可维护性原则
3. **上下文感知**：复用已完成的子任务结果（如`important_infos`、`user_supply_infos`等）
4. **循序渐进**：每次只生成一个Cell的代码，不要重复生成已执行的代码片段
"""
PROMPT_RULES = """
1. 代码生成核心原则

- **KISS原则**：保持逻辑简单明确，禁止过度设计  
  *示例*：避免嵌套三层以上的条件判断  
- **风格一致性**：  
  - 变量命名：`snake_case`（如`data_loader`）  
  - 函数命名：`verb_noun`（如`load_data()`）  
  - 类命名：`PascalCase`（如`DataProcessor`）  
- **错误处理**：
  - 代码应有良好的容错处理逻辑，有直观的错误提示，优先保证代码的健壮性
  - 对于无法进行容错处理的异常，发生异常时直接抛出，不要进行捕获或处理（`raise ValueError(...)`），以便于后续调试

2. 代码结构规范

- **模块化设计**：
  - 关键逻辑封装为函数/类（如`def preprocess_data(df):`）
  - 每个函数职责单一（单入口/单出口原则）
- **文档注释**：
  - 函数需包含`docstring`说明用途、参数和返回值
  - 关键逻辑添加注释（如`# 使用均值填充缺失值`）
- **依赖管理**：
  - 显式导入所有必要库（如`import pandas as pd`）
  - 禁止隐式依赖

3. 上下文交互规范

- **引用约束**：
  - 复用已完成的子任务结果
  - 禁止重复生成已执行的代码片段
- **用户输入处理**：
  - 充分、准确、合理的利用`important_infos`、`user_supply_infos`中的信息（如参数值、文件路径）
  - 避免遗漏、冲突或重复确认

4. 结果输出机制

- **结果限定**：仅通过`print()`或`return`输出，禁用文件写入、图形界面等
"""
PROMPT_TRIGGER = """
请根据子任务描述生成符合逻辑的Python代码。
"""


class CodeGeneratorAgent(BaseChatAgent):

    PROMPT_ROLE = PROMPT_ROLE
    PROMPT_RULES = PROMPT_RULES
    PROMPT_TRIGGER = PROMPT_TRIGGER
    OUTPUT_FORMAT = AgentOutputFormat.CODE
    OUTPUT_CODE_LANG = "python"
    MODEL_TYPE = AgentModelType.CODING

    def get_task_data(self):
        return {
            "cell_idx": self.task.cell_idx,
            "task_id": self.task.task_id,
            "subject": self.task.subject,
            "coding_prompt": self.task.coding_prompt,
            "issue": self.task.issue,
        }

    def on_reply(self, reply: str):
        generated_code = "# Generated by Jupyter Agent (Coder) {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"))
        generated_code += reply
        self.task.source = generated_code
