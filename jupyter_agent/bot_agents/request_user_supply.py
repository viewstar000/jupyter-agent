"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import re
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
    ReceiveUserSupplyInfoParams,
    ActionSetCellContent,
    SetCellContentParams,
    CellContentType,
)
from ..utils import get_env_capbilities


PROMPT_ROLE = """
你是一个用户需求补充专家，负责代替用户补充回答的需求中的问题，以便于更好的完成任务。
"""
PROMPT_RULES = """
- 根据提示补充回答待确认的问题，以便于更好的完成任务。
- 需要确保所有的问题都有明确的答案，以便于更好的完成任务。
- 若问题可能有多种可能的答案，需要选择最合适的答案，以便于更好的完成任务。
"""
TASK_TRIGGER = """
请按要求代替用户补充回答下述待确认的问题：

{% for issue in request_supply_infos %}
- {{ issue.question }}, (例如: {{ issue.example }})
{% endfor %}
"""


def filter_special_chars(text):
    if text is None:
        return ""
    return re.sub(r"[\s\"\']", "", text)


def format_request_info_yaml(
    issues: list[RequestUserSupplyInfo], title="用户补充确认信息", use_code_block=True
) -> str:
    result = f"### USER_SUPPLY_INFO: {title}[YAML]\n\n"
    result += "\n".join(
        [
            f'- assistant: "{filter_special_chars(prompt.question)} (例如: {filter_special_chars(prompt.example)})"\n'
            f'  user: ""'
            for prompt in issues
        ]
    )
    if use_code_block:
        result = f"```yaml\n{result}\n```\n"
    return result


def format_received_info_yaml(
    replies: list[UserSupplyInfoReply], title="用户补充确认信息", use_code_block=True
) -> str:
    result = f"### USER_SUPPLY_INFO: {title}[YAML]\n\n"
    result += "\n".join(
        [
            f'- assistant: "{filter_special_chars(reply.question)}"\n  user: "{filter_special_chars(reply.answer)}"'
            for reply in replies
        ]
    )
    if use_code_block:
        result = f"```yaml\n{result}\n```\n"
    return result


def format_request_info_json(infos: list[RequestUserSupplyInfo], title="用户补充确认信息", use_code_block=True) -> str:
    result = f"### USER_SUPPLY_INFO: {title}[JSON]\n\n"
    result += json.dumps(
        [{"question": f"{info.question} (例如: {info.example})", "answer": ""} for info in infos],
        indent=4,
        ensure_ascii=False,
    )
    if use_code_block:
        result = f"```json\n{result}\n```\n"
    return result


def format_received_info_json(
    replies: list[UserSupplyInfoReply], title="用户补充确认信息", use_code_block=True
) -> str:
    result = f"### USER_SUPPLY_INFO: {title}[JSON]\n\n"
    result += json.dumps(
        [{"question": f"{reply.question}", "answer": f"{reply.answer}"} for reply in replies],
        indent=4,
        ensure_ascii=False,
    )
    if use_code_block:
        result = f"```json\n{result}\n```\n"
    return result


def format_request_info_markdown(
    issues: list[RequestUserSupplyInfo], title="用户补充确认信息", use_markdown_block=True
) -> str:
    result = f"### {title}\n\n"
    result += "\n".join(
        [f"- **Assistant**: {prompt.question} (例如: {prompt.example})\n- **User Reply**: " for prompt in issues]
    )
    if use_markdown_block:
        result = f"```markdown\n{result}\n```\n"
    return result


def format_received_info_markdown(
    replies: list[UserSupplyInfoReply], title="用户补充确认信息", use_markdown_block=True
) -> str:
    result = f"### {title}\n\n"
    result += "\n".join([f"- **Assistant**: {reply.question}\n- **User Reply**: {reply.answer}" for reply in replies])
    if use_markdown_block:
        result = f"```markdown\n{result}\n```\n"
    return result


new_cell_content_type = CellContentType.RAW
format_received_user_supply_info = format_received_info_json
format_request_user_supply_info = format_request_info_json


class RequestUserSupplyAgent(BaseChatAgent):

    PROMPT_ROLE = PROMPT_ROLE
    PROMPT_RULES = PROMPT_RULES
    OUTPUT_FORMAT = AgentOutputFormat.JSON
    OUTPUT_JSON_SCHEMA = ReceiveUserSupplyInfoParams
    DISPLAY_REPLY = True
    MOCK_USER_SUPPLY: bool = False
    WHERE_USER_SUPPLY = "below"  # "above" or "below"

    def get_prompt_blocks(self):
        blocks = super().get_prompt_blocks()
        blocks["TASK_TRIGGER"] = TASK_TRIGGER
        return blocks

    def on_reply(self, reply: ReceiveUserSupplyInfoParams):
        assert reply, "Reply is empty"
        if get_env_capbilities().set_cell_content:
            insert_cell_idx = -1 if self.WHERE_USER_SUPPLY == "above" else 1
            action = ActionSetCellContent(
                source=self.__class__.__name__,
                params=SetCellContentParams(
                    index=insert_cell_idx,
                    type=new_cell_content_type,
                    source=format_received_user_supply_info(reply.replies, use_code_block=False),
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
                        type=new_cell_content_type,
                        source=format_request_user_supply_info(request_supply_infos, use_code_block=False),
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
