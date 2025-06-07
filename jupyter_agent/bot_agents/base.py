"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import json

from IPython.display import Markdown
from ..utils import ChatMixin

_TASK_CONTEXTS = """
**全局任务规划及子任务完成情况**：

{% set ns = namespace(tid=0) %}
{% for cell in cells %}
{% if cell.type == "global_plan" and cell.source.strip() %}
{{ cell.source }}
{% for plan in cell.outputs %}
{{ plan }}
{% endfor %}
{% elif cell.type == "task" and cell.source.strip() %}
{% set ns.tid = ns.tid + 1 %}
## 子任务 {{ ns.tid }} ({{ '已完成' if cell.outputs else '未完成' }})

### 任务目标
{{ cell.subject }}

### 任务结果
{% for output in cell.outputs %}
{{ output }}
{% endfor %}
{% elif "TASK" in cell.context and cell.source.strip() %}
{{ cell.source }}
{% endif %}
{% endfor %}
"""

_CODE_CONTEXTS = """
**已执行的代码**：

```python
{% set ns = namespace(tid=0, cid=0) %}
{% for cell in cells %}
{% if cell.type == "task" and cell.source.strip() %}
{% set ns.cid = ns.cid + 1 %}
{% set ns.tid = ns.tid + 1 %}
## Cell[{{ ns.cid }}] for Task[{{ ns.tid }}]:

{{ cell.source }}
{% elif "CODE" in cell.context and cell.source.strip() %}
{% set ns.cid = ns.cid + 1 %}
## Cell[{{ ns.cid }}]:

{{ cell.source }}
{% endif %}
{% endfor %}
```
"""

_TASK_OUTPUT_FORMAT = """
{% if OUTPUT_FORMAT == "code" %}
**输出格式**：

输出{{ OUTPUT_CODE_LANG }}代码块，以Markdown格式显示，使用```{{ OUTPUT_CODE_LANG }}...```包裹。

示例代码：

```python
def xxx(xxx):
    ...
    return xxx

xxx(...)
```

{% elif OUTPUT_FORMAT == "json" %}
**输出格式**：

输出JSON格式数据，以Markdown格式显示，使用```json...```包裹。

数据符合JSON Schema：

```json
{{ OUTPUT_JSON_SCHEMA }}
```

数据示例:

```json
{{ OUTPUT_JSON_EXAMPLE }}
```

{% endif %}
"""

PREDEFINE_PROMPT_BLOCKS = {
    "TASK_CONTEXTS": _TASK_CONTEXTS,
    "CODE_CONTEXTS": _CODE_CONTEXTS,
    "TASK_OUTPUT_FORMAT": _TASK_OUTPUT_FORMAT,
}

AGENT_OUTPUT_FORMAT_RAW = "raw"
AGENT_OUTPUT_FORMAT_TEXT = "text"
AGENT_OUTPUT_FORMAT_CODE = "code"
AGENT_OUTPUT_FORMAT_JSON = "json"
AGENT_COMBINE_REPLY_FIRST = "first"
AGENT_COMBINE_REPLY_LAST = "last"
AGENT_COMBINE_REPLY_LIST = "list"
AGENT_COMBINE_REPLY_MERGE = "merge"
AGENT_MODEL_TYPE_PLANNER = "planner"
AGENT_MODEL_TYPE_CODING = "coding"
AGENT_MODEL_TYPE_REASONING = "reasoning"


class BaseTaskAgent(ChatMixin):
    """基础聊天代理类"""

    PROMPT = "You are a helpful assistant. {{ prompt }}\n\nAnswer:"
    OUTPUT_FORMAT = AGENT_OUTPUT_FORMAT_RAW
    OUTPUT_CODE_LANG = "python"
    OUTPUT_JSON_SCHEMA = None  # Pydantic Model
    DISPLAY_REPLY = True
    COMBINE_REPLY = AGENT_COMBINE_REPLY_MERGE
    ACCEPT_EMPYT_REPLY = False
    MODEL_TYPE = AGENT_MODEL_TYPE_REASONING

    def __init__(self, notebook_context, task_context, base_url, api_key, model_name, debug_level=0):
        """初始化基础任务代理"""
        ChatMixin.__init__(self, base_url, api_key, model_name, debug_level=debug_level)
        self.notebook_context = notebook_context
        self.task_context = task_context

    def prepare_contexts(self, **kwargs):
        contexts = {
            "cells": self.notebook_context.cells,
            "task": self.task_context,
            "OUTPUT_FORMAT": self.OUTPUT_FORMAT,
            "OUTPUT_CODE_LANG": self.OUTPUT_CODE_LANG,
        }
        if self.OUTPUT_JSON_SCHEMA:
            json_schema = self.OUTPUT_JSON_SCHEMA.model_json_schema()
            json_example = {
                name: field.examples[0] if field.examples else field.default
                for name, field in self.OUTPUT_JSON_SCHEMA.model_fields.items()
            }
            contexts["OUTPUT_JSON_SCHEMA"] = json.dumps(json_schema, indent=2, ensure_ascii=False)
            contexts["OUTPUT_JSON_EXAMPLE"] = json.dumps(json_example, indent=2, ensure_ascii=False)
        contexts.update(kwargs)
        return contexts

    def create_messages(self, contexts):
        messages = super().create_messages(contexts, templates=PREDEFINE_PROMPT_BLOCKS)
        messages.add(self.PROMPT)
        return messages

    def combine_raw_replies(self, replies):
        if self.COMBINE_REPLY == AGENT_COMBINE_REPLY_FIRST:
            return replies[0]["raw"]
        elif self.COMBINE_REPLY == AGENT_COMBINE_REPLY_LAST:
            return replies[-1]["raw"]
        elif self.COMBINE_REPLY == AGENT_COMBINE_REPLY_MERGE:
            return "".join([reply["raw"] for reply in replies])
        else:
            raise ValueError("Unsupported combine_reply: {} for raw output".format(self.COMBINE_REPLY))

    def combine_code_replies(self, replies):
        code_replies = [
            reply for reply in replies if reply["type"] == "code" and reply["lang"] == self.OUTPUT_CODE_LANG
        ]
        if self.COMBINE_REPLY == AGENT_COMBINE_REPLY_FIRST:
            return code_replies[0]["content"]
        elif self.COMBINE_REPLY == AGENT_COMBINE_REPLY_LAST:
            return code_replies[-1]["content"]
        elif self.COMBINE_REPLY == AGENT_COMBINE_REPLY_MERGE:
            return "\n".join([reply["content"] for reply in code_replies])
        else:
            raise ValueError("Unsupported combine_reply: {} for code output".format(self.COMBINE_REPLY))

    def combine_json_replies(self, replies):
        json_replies = [reply for reply in replies if reply["type"] == "code" and reply["lang"] == "json"]
        if self.COMBINE_REPLY == AGENT_COMBINE_REPLY_FIRST:
            json_obj = json.loads(json_replies[0]["content"])
            if self.OUTPUT_JSON_SCHEMA:
                json_obj = self.OUTPUT_JSON_SCHEMA(**json_obj)
            return json_obj
        elif self.COMBINE_REPLY == AGENT_COMBINE_REPLY_LAST:
            json_obj = json.loads(json_replies[-1]["content"])
            if self.OUTPUT_JSON_SCHEMA:
                json_obj = self.OUTPUT_JSON_SCHEMA(**json_obj)
            return json_obj
        elif self.COMBINE_REPLY == AGENT_COMBINE_REPLY_LIST:
            json_objs = [json.loads(reply["content"]) for reply in json_replies]
            if self.OUTPUT_JSON_SCHEMA:
                json_objs = [self.OUTPUT_JSON_SCHEMA(**json_obj) for json_obj in json_objs]
            return json_objs
        elif self.COMBINE_REPLY == AGENT_COMBINE_REPLY_MERGE:
            json_obj = {}
            for json_reply in json_replies:
                json_obj.update(json.loads(json_reply["content"]))
            if self.OUTPUT_JSON_SCHEMA:
                json_obj = self.OUTPUT_JSON_SCHEMA(**json_obj)
            return json_obj
        else:
            raise ValueError("Unsupported combine_reply: {} for json output".format(self.COMBINE_REPLY))

    def combine_text_replies(self, replies):
        text_replies = [reply for reply in replies if reply["type"] == "text"]
        if self.COMBINE_REPLY == AGENT_COMBINE_REPLY_FIRST:
            return text_replies[0]["content"]
        elif self.COMBINE_REPLY == AGENT_COMBINE_REPLY_LAST:
            return text_replies[-1]["content"]
        elif self.COMBINE_REPLY == AGENT_COMBINE_REPLY_MERGE:
            return "".join([reply["content"] for reply in text_replies])
        else:
            raise ValueError("Unsupported combine_reply: {} for text output".format(self.COMBINE_REPLY))

    def combine_replies(self, replies):
        if self.OUTPUT_FORMAT == AGENT_OUTPUT_FORMAT_RAW:
            return self.combine_raw_replies(replies).strip()
        elif self.OUTPUT_FORMAT == AGENT_OUTPUT_FORMAT_TEXT:
            return self.combine_text_replies(replies).strip()
        elif self.OUTPUT_FORMAT == AGENT_OUTPUT_FORMAT_CODE:
            return self.combine_code_replies(replies).strip()
        elif self.OUTPUT_FORMAT == AGENT_OUTPUT_FORMAT_JSON:
            return self.combine_json_replies(replies)
        else:
            raise ValueError("Unsupported output format: {}".format(self.OUTPUT_FORMAT))

    def on_reply(self, reply):
        self._C(Markdown(reply))

    def __call__(self, **kwargs):
        contexts = self.prepare_contexts(**kwargs)
        messages = self.create_messages(contexts)
        replies = self.chat(messages.get(), display_reply=self.DISPLAY_REPLY)
        reply = self.combine_replies(replies)
        if not self.ACCEPT_EMPYT_REPLY and not reply:
            raise ValueError("Reply is empty")
        result = self.on_reply(reply)
        if not isinstance(result, tuple):
            return False, result
        else:
            return result


class AgentFactory:

    def __init__(
        self,
        notebook_context,
        task_context,
        planner_api_url=None,
        planner_api_key="API_KEY",
        planner_model=None,
        coding_api_url=None,
        coding_api_key="API_KEY",
        coding_model=None,
        reasoning_api_url=None,
        reasoning_api_key="API_KEY",
        reasoning_model=None,
        debug_level=0,
    ):
        self.notebook_context = notebook_context
        self.task_context = task_context
        self.debug_level = debug_level
        self.models = {
            AGENT_MODEL_TYPE_PLANNER: {
                "api_url": planner_api_url,
                "api_key": planner_api_key,
                "model": planner_model,
            },
            AGENT_MODEL_TYPE_CODING: {
                "api_url": coding_api_url,
                "api_key": coding_api_key,
                "model": coding_model,
            },
            AGENT_MODEL_TYPE_REASONING: {
                "api_url": reasoning_api_url,
                "api_key": reasoning_api_key,
                "model": reasoning_model,
            },
        }

    def __call__(self, agent_class):

        from .. import bot_agents

        if isinstance(agent_class, str):
            agent_class = getattr(bot_agents, agent_class)
        else:
            if not hasattr(bot_agents, agent_class.__name__):
                raise ValueError("Unsupported agent class: {}".format(agent_class))

        if issubclass(agent_class, BaseTaskAgent):
            agent_model = agent_class.MODEL_TYPE if hasattr(agent_class, "MODEL_TYPE") else AGENT_MODEL_TYPE_REASONING
            return agent_class(
                notebook_context=self.notebook_context,
                task_context=self.task_context,
                base_url=self.models[agent_model]["api_url"],
                api_key=self.models[agent_model]["api_key"],
                model_name=self.models[agent_model]["model"],
                debug_level=self.debug_level,
            )
        else:
            return agent_class(
                notebook_context=self.notebook_context,
                task_context=self.task_context,
                debug_level=self.debug_level,
            )
