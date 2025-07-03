"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import time
import uuid
import json

from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from IPython.display import Markdown
from .base import BaseChatAgent, AgentOutputFormat
from ..bot_outputs import _D, _I, _W, _E, _F, _M, _B, _C, _O
from ..bot_actions import (
    get_action_dispatcher,
    RequestUserSupplyInfo,
    UserSupplyInfoReply,
    ActionRequestUserSupplyInfo,
    RequestUserSupplyInfoParams,
    ActionReceiveUserSupplyInfo,
    ReceiveUserSupplyInfoParams,
    ActionSetCellContent,
    SetCellContentParams,
)
from ..utils import get_env_capbilities

MOCK_USER_REPLY_PROMPT = """\
**角色定义**：

你是一个用户需求补充专家，负责代替用户补充回答的需求中的问题，以便于更好的完成任务。

**任务要求**：

- 根据提示补充回答待确认的问题，以便于更好的完成任务。
- 需要确保所有的问题都有明确的答案，以便于更好的完成任务。


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

### 当前代码执行的输出与结果：
{{ task.output }}

### 当前任务总结要求：
{{ task.summary_prompt }}

### 当前任务总结结果：
{{ task.result }}

---

需要你代替用户补充确认的问题：

{% for issue in request_supply_infos %}
- {{ issue.prompt }}, (例如: {{ issue.example }})
{% endfor %}

---

请按要求代替用户补充回答上述待确认的问题：
"""


def format_request_user_supply_info(
    issues: list[RequestUserSupplyInfo], title="用户补充确认信息", use_markdown_block=True
) -> str:
    result = f"### {title}\n\n"
    result += "\n".join(
        [f"- **Assistant**: {prompt.prompt} (例如: {prompt.example})\n- **User Reply**: " for prompt in issues]
    )
    if use_markdown_block:
        result = f"```markdown\n{result}\n```\n"
    return result


def format_received_user_supply_info(
    replies: list[UserSupplyInfoReply], title="用户补充确认信息", use_markdown_block=True
) -> str:
    result = f"### {title}\n\n"
    result += "\n".join([f"- **Assistant**: {reply.prompt}\n- **User Reply**: {reply.reply}" for reply in replies])
    if use_markdown_block:
        result = f"```markdown\n{result}\n```\n"
    return result


class RequestUserSupplyAgent(BaseChatAgent):

    PROMPT = MOCK_USER_REPLY_PROMPT
    OUTPUT_FORMAT = AgentOutputFormat.JSON
    OUTPUT_JSON_SCHEMA = ReceiveUserSupplyInfoParams
    DISPLAY_REPLY = True
    MOCK_USER_SUPPLY: bool = False
    WHERE_USER_SUPPLY = "below"  # "above" or "below"

    def on_reply(self, reply: ReceiveUserSupplyInfoParams):
        assert reply, "Reply is empty"
        if get_env_capbilities().set_cell_content:
            insert_cell_idx = -1 if self.WHERE_USER_SUPPLY == "above" else 1
            action = ActionSetCellContent(
                source=self.__class__.__name__,
                params=SetCellContentParams(
                    index=insert_cell_idx,
                    type="markdown",
                    source=format_received_user_supply_info(reply.replies, use_markdown_block=False),
                ),
            )
            get_action_dispatcher().send_action(action, need_reply=False)
        else:
            _M("### 用户补充确认的信息\n\n请将下面的内容保存到单独的单元格中，以便于更好的完成任务\n\n")
            _M(format_received_user_supply_info(reply.replies))

    def __call__(self, **kwargs) -> Tuple[bool, Any]:
        request_supply_infos = (
            self.task.request_above_supply_infos
            if self.WHERE_USER_SUPPLY == "above"
            else self.task.request_below_supply_infos
        )
        kwargs["request_supply_infos"] = request_supply_infos
        if self.MOCK_USER_SUPPLY:
            return super().__call__(**kwargs)
        else:
            if get_env_capbilities().user_supply_info:
                _I(f"Request User Supply Info: {request_supply_infos}")
                action = ActionRequestUserSupplyInfo(
                    source=self.__class__.__name__,
                    params=RequestUserSupplyInfoParams(title="用户需求补充确认", issues=request_supply_infos),
                )
                get_action_dispatcher().send_action(action, need_reply=True)
                res = get_action_dispatcher().get_action_reply(action, wait=True)
                return False, self.on_reply(res and res.params)  # type: ignore
            elif get_env_capbilities().set_cell_content:
                _M(
                    f"**需要用户补充确认信息**，"
                    f"请将{'下面' if self.WHERE_USER_SUPPLY == 'below' else '上面'}的单元格中的内容补充完整，"
                    f"以便于更好的完成任务"
                )
                insert_cell_idx = -1 if self.WHERE_USER_SUPPLY == "above" else 1
                action = ActionSetCellContent(
                    source=self.__class__.__name__,
                    params=SetCellContentParams(
                        index=insert_cell_idx,
                        type="markdown",
                        source=format_request_user_supply_info(request_supply_infos, use_markdown_block=False),
                    ),
                )
                get_action_dispatcher().send_action(action, need_reply=False)
            else:
                _M(
                    "### 需要补充确认的信息\n\n"
                    "请将下面的内容保存到单独的单元格中并将其补充完整，以便于更好的完成任务\n\n"
                )
                _M(format_request_user_supply_info(request_supply_infos))
            return False, None


class RequestAboveUserSupplyAgent(RequestUserSupplyAgent):
    WHERE_USER_SUPPLY = "above"


class RequestBelowUserSupplyAgent(RequestUserSupplyAgent):
    WHERE_USER_SUPPLY = "below"
