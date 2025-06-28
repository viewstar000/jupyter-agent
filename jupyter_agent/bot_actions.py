"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import json
import time
import uuid
import threading
import queue
import traceback
import importlib
import socket

from enum import Enum
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from wsgiref.simple_server import make_server
from bottle import default_app, get, post, request, response
from .utils import get_env_capbilities


class ActionBase(BaseModel):
    timestamp: float = 0
    uuid: str = ""
    source: str = ""
    action: str
    params: Dict[str, Any] = {}

    def __init__(self, **data):
        super().__init__(**data)
        self.timestamp = self.timestamp or time.time()
        self.uuid = self.uuid or str(uuid.uuid4())


class ReplyActionBase(ActionBase):
    reply_host: str = ""
    reply_port: int = 0


class SetCellContentParams(BaseModel):
    index: int = 1  # -1 previous, 0 current, 1 next
    type: str = "code"  # code/markdown
    source: str = ""
    tags: List[str] = []
    metadata: Dict[str, Any] = {}


class ActionSetCellContent(ActionBase):

    action: str = "set_cell_content"
    params: SetCellContentParams = SetCellContentParams()


class ConfirmChoiceItem(BaseModel):
    label: str = ""
    value: str


class RequestUserConfirmParams(BaseModel):
    prompt: str = ""
    choices: List[ConfirmChoiceItem] = []
    default: str = ""


class ActionRequestUserConfirm(ReplyActionBase):

    action: str = "request_user_confirm"
    params: RequestUserConfirmParams = RequestUserConfirmParams()


class ReceiveUserConfirmParams(BaseModel):
    result: str = ""


class ActionReceiveUserConfirm(ActionBase):

    action: str = "receive_user_confirm"
    params: ReceiveUserConfirmParams = ReceiveUserConfirmParams()


class RequestUserSupplyInfo(BaseModel):
    prompt: str = Field(
        description="需要用户补充详细信息的Prompt",
        examples=["请补充与...相关的详细的信息", "请确认...是否...", "请提供..."],
    )
    example: Optional[str] = Field(None, description="示例", examples=["..."])


class UserSupplyInfoReply(BaseModel):
    prompt: str = Field(description="需要用户补充详细信息的Prompt", examples=["..."])
    reply: str = Field(description="用户补充的详细信息", examples=["..."])


class RequestUserSupplyInfoParams(BaseModel):
    title: str = ""
    issues: List[RequestUserSupplyInfo] = []


class ActionRequestUserSupplyInfo(ReplyActionBase):

    action: str = "request_user_supply_info"
    params: RequestUserSupplyInfoParams = RequestUserSupplyInfoParams()


class ReceiveUserSupplyInfoParams(BaseModel):
    replies: List[UserSupplyInfoReply] = Field(
        description="完成补充确认的信息列表",
        examples=[
            UserSupplyInfoReply(prompt="请确认...是否...", reply="是"),
            UserSupplyInfoReply(prompt="请补充...", reply="..."),
        ],
    )


class ActionReceiveUserSupplyInfo(ActionBase):
    action: str = "receive_user_supply_info"
    params: ReceiveUserSupplyInfoParams = ReceiveUserSupplyInfoParams(replies=[])


def request_user_reply(prompts: list[RequestUserSupplyInfo]) -> list[UserSupplyInfoReply]:
    responses = []
    for prompt in prompts:
        response = input(f"{prompt.prompt} (例如: {prompt.example})")
        responses.append(UserSupplyInfoReply(prompt=prompt.prompt, reply=response))
    return responses


def get_action_class(action_name: str) -> type[ActionBase]:
    for obj in globals().values():
        if isinstance(obj, type) and issubclass(obj, ActionBase):
            if obj.__name__ == action_name or obj.model_fields["action"].default == action_name:
                return obj
    raise ValueError(f"Unknown action: {action_name}")


class ActionReply(BaseModel):
    reply_timestamp: float
    retrieved_timestamp: float = 0
    uuid: str
    source: str = ""
    action: str = ""
    retrieved: bool = False
    reply: ActionBase


class ActionDispatcher(threading.Thread):
    def __init__(self, host="127.0.0.1", port=0, app=None):
        super().__init__(daemon=True)
        self.action_queue = queue.Queue()
        self.action_replies: dict[str, ActionReply] = {}
        self.app = app or default_app()
        self.host = host
        self.port = port
        self.server = None
        if get_env_capbilities().user_confirm or get_env_capbilities().user_supply_info:
            self.port = self.port or self.select_port(self.host)
            self.server = make_server(self.host, self.port, self.app)
            self.start()

    def select_port(self, host):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def run(self):
        if self.server is not None:
            self.server.serve_forever()

    def close(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def send_action(self, action: ActionBase, need_reply: bool = False):

        if need_reply:
            assert isinstance(action, ReplyActionBase)
            action.reply_host = self.host
            action.reply_port = self.port
        action.timestamp = action.timestamp or time.time()
        action.uuid = action.uuid and str(uuid.uuid4())
        self.action_queue.put(action.model_dump())
        bot_outputs = importlib.import_module(".bot_outputs", __package__)
        bot_outputs.output_action(action)

    def get_action_reply(self, action: ReplyActionBase, wait: bool = True) -> Optional[ActionBase]:

        while wait and action.uuid not in self.action_replies:
            time.sleep(1)
        if action.uuid in self.action_replies:
            self.action_replies[action.uuid].retrieved = True
            self.action_replies[action.uuid].retrieved_timestamp = time.time()
        return self.action_replies.get(action.uuid) and self.action_replies[action.uuid].reply


_default_action_dispatcher = None


def get_action_dispatcher() -> ActionDispatcher:
    global _default_action_dispatcher

    if not _default_action_dispatcher:
        _default_action_dispatcher = ActionDispatcher()
    elif not _default_action_dispatcher.is_alive():
        _default_action_dispatcher.close()
        _default_action_dispatcher = ActionDispatcher()
    return _default_action_dispatcher


def close_action_dispatcher():
    global _default_action_dispatcher

    if _default_action_dispatcher:
        _default_action_dispatcher.close()
        _default_action_dispatcher = None


@get("/echo")
def echo():
    response.content_type = "application/json"
    return json.dumps({"status": "OK"})


@post("/action_reply")
def action_reply():
    try:
        uuid = request.GET["uuid"]  # type: ignore
        action = request.GET.get("a") or request.json.get("action")  # type: ignore
        source = request.GET.get("s") or request.json.get("source")  # type: ignore
        reply = get_action_class(action)(**request.json)  # type: ignore
        action_reply = ActionReply(reply_timestamp=time.time(), uuid=uuid, source=source, action=action, reply=reply)
        get_action_dispatcher().action_replies[action_reply.uuid] = action_reply
        response.content_type = "application/json"
        return json.dumps({"status": "OK"})
    except Exception as e:
        response.content_type = "application/json"
        return json.dumps(
            {"status": "ERROR", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        )


@get("/action_fetch")
def action_fetch():
    try:
        action = get_action_dispatcher().action_queue.get(block=False)
        response.content_type = "application/json"
        return json.dumps({"status": "OK", "action": action})
    except queue.Empty:
        response.content_type = "application/json"
        return json.dumps({"status": "EMPTY"})
    except Exception as e:
        response.content_type = "application/json"
        return json.dumps(
            {"status": "ERROR", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        )
