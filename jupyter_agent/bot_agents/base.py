"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import json
import importlib
import traceback

from collections import OrderedDict
from typing import Tuple, Any, Optional, Type
from enum import Enum, unique
from pydantic import BaseModel, Field
from IPython.display import Markdown
from ..bot_outputs import _C, _O, _W, _T, flush_output
from ..bot_chat import BotChat
from ..utils import no_indent

_TASK_CONTEXTS = no_indent(
    """
**全局任务规划及子任务完成情况**：

{%+ for cell in cells %}
    {% if cell.type == "planning" and cell.source.strip() +%}

        {{ cell.source }}
        {{ cell.result }}
    {% elif cell.type == "task" and cell.subject.strip() %}
        ## 子任务[{{ cell.task_id }}] - {{ '已完成' if cell.result else '未完成' }}

        ### 任务目标

        {{ cell.subject }}

        ### 任务结果

        {{ cell.result }}

        {% if cell.important_infos and not merged_important_infos %}
            ### 【重要】任务结论中的重要信息(Important Infos)

            ```json
            {{ cell.important_infos | json }}
            ```
        {%+ endif %}
    {% elif cell.type == "user_supply_info" %}
        {% if cell.get_user_supply_infos() and not merged_user_supply_infos %}
            ## 【重要】用户提供的补充信息(User Supply Infos)

            {% if user_supply_info_format == "markdown" %}
                {% for info in cell.get_user_supply_infos() %}
                    - Question: {{ info.question }}
                    - Answer: {{ info.answer }}
                {%+ endfor %}
            {% else %}
                ```json
                {{ cell.get_user_supply_infos() | json }}
                ```
            {% endif %}
        {% endif %}
    {% elif cell.is_task_context and cell.source.strip() %}
        {{ cell.source }}
    {% endif %}
{% endfor %}

{% if merged_user_supply_infos %}
## 用户提供的补充信息(User Supply Infos)

```json
{{ merged_user_supply_infos | json }}
```
{% endif %}

{% if merged_important_infos %}
## 已完成的任务生成的重要信息(Important Infos)

```json
{{ merged_important_infos | json }}
```
{% endif %}

"""
)


_CODE_CONTEXTS = no_indent(
    """
**已执行的代码**：

```python
{% for cell in cells +%}
    {% if cell.type == "task" and cell.source.strip() %}
        ######## Cell[{{ cell.cell_idx }}] for Task[{{ cell.task_id }}] ########

        {{ cell.source }}
    {% elif cell.is_code_context and cell.source.strip() %}
        ######## Cell[{{ cell.cell_idx }}] ########

        {{ cell.source }}
    {% endif %}
{%+ endfor %}
```

"""
)


_TASK_AGENT = """
**角色定义**：

{{ agent_role }}

**任务要求**：

{{ task_rules }}

{% if output_format == "code" %}
**输出格式**：

输出{{ output_code_lang }}代码块，以Markdown格式显示，使用```{{ output_code_lang }}...```包裹。

示例代码：

```python
def xxx(xxx):
    ...
    return xxx

xxx(...)
```

{% elif output_format == "json" %}
**输出格式**：

输出结果为JSON数据，以Markdown文档形式输出，使用```json...```包裹。

输出结果必须符合如下JSON Schema的约束：

```json
{{ output_json_schema }}
```

输出结果示例:

```json
{{ output_json_example }}
```

{% endif %}
"""

_TASK_DATA = no_indent(
    """
{% if task and task.subject %}
    **当前子任务信息**:

    ### 当前子任务规划目标：
    {{ task.subject }}

    {% if task.coding_prompt %}
        ### 当前子任务代码需求：
        {{ task.coding_prompt }}

        {% if task.source %}
            ### 当前子任务生成的代码：
            ```python
            {{ task.source }}
            ```

            {% if task.output %}
            ### 当前代码执行的输出与结果：
            {{ task.output }}
            {% endif %}
        {% endif %}
    {% endif %}

    {% if task.summary_prompt %}
        ### 当前子任务总结要求：
        {{ task.summary_prompt }}

        {% if task.result %}
            ### 当前子任务输出的分析总结后的最终结果：
            ```markdown
            {{ task.result }}
            ```

            {% if task.important_infos %}
            ### 当前子任务输出的重要信息：
            ```json
            {{ task.important_infos | json }}
            ```
            {% endif %}

            {% if task.request_below_supply_infos %}
            ### 当前子任务输出的请求用户补充确认的信息：
            ```json
            {{ task.request_below_supply_infos | json }}
            ```
            {% endif %}
        {% endif %}
    {% endif %}

    {% if task.issue %}
        ### 当前子任务存在的问题）
        {{ task.issue }}
    {% endif %}
{% endif %}
"""
)

_TASK_TRIGGER = """
{{ task_trigger }}
"""

PREDEFINE_PROMPT_BLOCKS = OrderedDict(
    TASK_CONTEXTS=_TASK_CONTEXTS,
    CODE_CONTEXTS=_CODE_CONTEXTS,
    TASK_DATA=_TASK_DATA,
    TASK_AGENT=_TASK_AGENT,
    TASK_TRIGGER=_TASK_TRIGGER,
)


_DEFAULT_AGENT_PROMPT_TPL = """\
{% for block in blocks %}
{% include block %}
{% if not loop.last %}

---

{% endif %}
{% endfor %}
"""


@unique
class AgentOutputFormat(str, Enum):
    RAW = "raw"
    TEXT = "text"
    CODE = "code"
    JSON = "json"


@unique
class AgentCombineReply(str, Enum):
    FIRST = "first"
    LAST = "last"
    LIST = "list"
    MERGE = "merge"


@unique
class AgentModelType(str, Enum):
    DEFAULT = "default"
    PLANNER = "planner"
    CODING = "coding"
    EVALUATING = "evaluating"
    REASONING = "reasoning"


class BaseAgent:
    """基础代理类"""

    def __init__(self, notebook_context):
        self.notebook_context = notebook_context

    @property
    def task(self):
        return self.notebook_context.cur_task

    @property
    def cells(self):
        return self.notebook_context.cells

    def __call__(self, **kwds: Any) -> Tuple[bool, Any]:
        raise NotImplementedError


class BaseChatAgent(BotChat, BaseAgent):
    """基础聊天代理类"""

    PROMPT_TPL = _DEFAULT_AGENT_PROMPT_TPL
    PROMPT_ROLE = "You are a helpful assistant. "
    PROMPT_RULES = "Help the user to complete the task."
    PROMPT_TRIGGER = "Now start the task:"
    OUTPUT_FORMAT: AgentOutputFormat = AgentOutputFormat.RAW
    OUTPUT_CODE_LANG = "python"
    OUTPUT_JSON_SCHEMA: Optional[Type[BaseModel]] = None
    DISPLAY_REPLY = True
    COMBINE_REPLY: AgentCombineReply = AgentCombineReply.MERGE
    ACCEPT_EMPYT_REPLY = False
    REPLY_ERROR_RETRIES = 1
    MODEL_TYPE: AgentModelType = AgentModelType.DEFAULT

    def __init__(self, notebook_context, **chat_kwargs):
        """初始化基础任务代理"""
        BaseAgent.__init__(self, notebook_context)
        BotChat.__init__(self, **chat_kwargs)

    def get_prompt_tpl(self):
        return self.PROMPT_TPL

    def get_prompt_blocks(self) -> OrderedDict:
        return PREDEFINE_PROMPT_BLOCKS.copy()

    def get_role_prompt(self):
        return self.PROMPT_ROLE

    def get_rules_prompt(self):
        return self.PROMPT_RULES

    def get_trigger_prompt(self):
        return self.PROMPT_TRIGGER

    def get_task_data(self):
        return self.task

    def prepare_contexts(self, **kwargs):
        contexts = {
            "blocks": self.get_prompt_blocks().keys(),
            "cells": self.cells,
            "task": self.get_task_data(),
            "merged_important_infos": None,  # self.notebook_context.merged_important_infos,
            "merged_user_supply_infos": None,  # self.notebook_context.merged_user_supply_infos,
            "agent_role": self.get_role_prompt(),
            "task_rules": self.get_rules_prompt(),
            "task_trigger": self.get_trigger_prompt(),
            "output_format": self.OUTPUT_FORMAT,
            "output_code_lang": self.OUTPUT_CODE_LANG,
        }
        if self.OUTPUT_JSON_SCHEMA:
            json_schema = self.OUTPUT_JSON_SCHEMA.model_json_schema()
            model_fields = getattr(self.OUTPUT_JSON_SCHEMA, "model_fields", None)
            if model_fields and hasattr(model_fields, "items"):
                json_example = {
                    name: field.examples[0] if getattr(field, "examples", None) else getattr(field, "default", None)
                    for name, field in model_fields.items()
                }
            else:
                json_example = {}

            def _default(o):
                if isinstance(o, BaseModel):
                    return o.model_dump()
                if isinstance(o, Enum):
                    return o.value
                return repr(o)

            contexts["output_json_schema"] = json.dumps(json_schema, indent=2, ensure_ascii=False, default=_default)
            contexts["output_json_example"] = json.dumps(json_example, indent=2, ensure_ascii=False, default=_default)
        contexts.update(kwargs)
        return contexts

    def create_messages(self, contexts):
        messages = super().create_messages(contexts, templates=self.get_prompt_blocks())
        messages.add(self.get_prompt_tpl())
        return messages

    def combine_raw_replies(self, replies):
        if self.COMBINE_REPLY == AgentCombineReply.FIRST:
            return replies[0]["raw"]
        elif self.COMBINE_REPLY == AgentCombineReply.LAST:
            return replies[-1]["raw"]
        elif self.COMBINE_REPLY == AgentCombineReply.MERGE:
            return "".join([reply["raw"] for reply in replies])
        else:
            raise ValueError("Unsupported combine_reply: {} for raw output".format(self.COMBINE_REPLY))

    def combine_code_replies(self, replies):
        code_replies = [
            reply for reply in replies if reply["type"] == "code" and reply["lang"] == self.OUTPUT_CODE_LANG
        ]
        if self.COMBINE_REPLY == AgentCombineReply.FIRST:
            return code_replies[0]["content"]
        elif self.COMBINE_REPLY == AgentCombineReply.LAST:
            return code_replies[-1]["content"]
        elif self.COMBINE_REPLY == AgentCombineReply.MERGE:
            return "\n".join([reply["content"] for reply in code_replies])
        else:
            raise ValueError("Unsupported combine_reply: {} for code output".format(self.COMBINE_REPLY))

    def combine_json_replies(self, replies):
        json_replies = [reply for reply in replies if reply["type"] == "code" and reply["lang"] == "json"]
        assert self.COMBINE_REPLY in [
            AgentCombineReply.FIRST,
            AgentCombineReply.LAST,
            AgentCombineReply.LIST,
            AgentCombineReply.MERGE,
        ]
        try:
            if self.COMBINE_REPLY == AgentCombineReply.FIRST:
                json_obj = json.loads(json_replies[0]["content"])
                if self.OUTPUT_JSON_SCHEMA:
                    json_obj = self.OUTPUT_JSON_SCHEMA(**json_obj)
                return json_obj
            elif self.COMBINE_REPLY == AgentCombineReply.LAST:
                json_obj = json.loads(json_replies[-1]["content"])
                if self.OUTPUT_JSON_SCHEMA:
                    json_obj = self.OUTPUT_JSON_SCHEMA(**json_obj)
                return json_obj
            elif self.COMBINE_REPLY == AgentCombineReply.LIST:
                json_objs = [json.loads(reply["content"]) for reply in json_replies]
                if self.OUTPUT_JSON_SCHEMA:
                    json_objs = [self.OUTPUT_JSON_SCHEMA(**json_obj) for json_obj in json_objs]
                return json_objs
            elif self.COMBINE_REPLY == AgentCombineReply.MERGE:
                json_obj = {}
                for json_reply in json_replies:
                    json_obj.update(json.loads(json_reply["content"]))
                if self.OUTPUT_JSON_SCHEMA:
                    json_obj = self.OUTPUT_JSON_SCHEMA(**json_obj)
                return json_obj
            else:
                return False
        except Exception as e:
            _T(f"提取JSON失败: {type(e).__name__}: {e}")
            _W(traceback.format_exc())
            return False

    def combine_text_replies(self, replies):
        text_replies = [reply for reply in replies if reply["type"] == "text"]
        if self.COMBINE_REPLY == AgentCombineReply.FIRST:
            return text_replies[0]["content"]
        elif self.COMBINE_REPLY == AgentCombineReply.LAST:
            return text_replies[-1]["content"]
        elif self.COMBINE_REPLY == AgentCombineReply.MERGE:
            return "".join([reply["content"] for reply in text_replies])
        else:
            raise ValueError("Unsupported combine_reply: {} for text output".format(self.COMBINE_REPLY))

    def combine_replies(self, replies):
        if self.OUTPUT_FORMAT == AgentOutputFormat.RAW:
            return self.combine_raw_replies(replies).strip()
        elif self.OUTPUT_FORMAT == AgentOutputFormat.TEXT:
            return self.combine_text_replies(replies).strip()
        elif self.OUTPUT_FORMAT == AgentOutputFormat.CODE:
            return self.combine_code_replies(replies).strip()
        elif self.OUTPUT_FORMAT == AgentOutputFormat.JSON:
            return self.combine_json_replies(replies)
        else:
            raise ValueError("Unsupported output format: {}".format(self.OUTPUT_FORMAT))

    def on_reply(self, reply) -> Tuple[bool, Any] | Any:
        _C(Markdown(reply))

    def __call__(self, **kwargs) -> Tuple[bool, Any]:
        contexts = self.prepare_contexts(**kwargs)
        messages = self.create_messages(contexts)
        reply_retries = 0
        while reply_retries <= self.REPLY_ERROR_RETRIES:
            replies = self.chat(messages.get(), display_reply=self.DISPLAY_REPLY)
            reply = self.combine_replies(replies)
            if reply is False:
                reply_retries += 1
                if reply_retries > self.REPLY_ERROR_RETRIES:
                    raise ValueError("Failed to get reply")
                _W("Failed to get reply, retrying...")
            elif not self.ACCEPT_EMPYT_REPLY and not reply:
                reply_retries += 1
                if reply_retries > self.REPLY_ERROR_RETRIES:
                    raise ValueError("Reply is empty")
                _W("Reply is empty, retrying...")
            else:
                break
        result = self.on_reply(reply)
        flush_output()
        if not isinstance(result, tuple):
            return False, result
        else:
            return result


class AgentFactory:

    def __init__(self, notebook_context, **chat_kwargs):
        self.notebook_context = notebook_context
        self.chat_kwargs = chat_kwargs
        self.models = {AgentModelType.DEFAULT: {"api_url": None, "api_key": None, "model": None}}

    def config_model(self, agent_model, api_url, api_key, model_name):
        self.models[agent_model] = {
            "api_url": api_url,
            "api_key": api_key,
            "model": model_name,
        }

    def get_agent_class(self, agent_class):
        if isinstance(agent_class, str):
            bot_agents = importlib.import_module("..bot_agents", __package__)
            agent_class = getattr(bot_agents, agent_class)
        assert issubclass(agent_class, BaseAgent), "Unsupported agent class: {}".format(agent_class)
        return agent_class

    def get_chat_kwargs(self, agent_class):
        if issubclass(agent_class, BaseChatAgent):
            agent_model = agent_class.MODEL_TYPE if hasattr(agent_class, "MODEL_TYPE") else AgentModelType.DEFAULT
            chat_kwargs = {
                "base_url": self.models.get(agent_model, {}).get("api_url")
                or self.models[AgentModelType.DEFAULT]["api_url"],
                "api_key": self.models.get(agent_model, {}).get("api_key")
                or self.models[AgentModelType.DEFAULT]["api_key"],
                "model_name": self.models.get(agent_model, {}).get("model")
                or self.models[AgentModelType.DEFAULT]["model"],
            }
            chat_kwargs.update(self.chat_kwargs)
            return chat_kwargs
        else:
            return {}

    def __call__(self, agent_class):

        agent_class = self.get_agent_class(agent_class)
        chat_kwargs = self.get_chat_kwargs(agent_class)
        return agent_class(self.notebook_context, **chat_kwargs)
