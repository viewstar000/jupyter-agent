"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import re
import json
import jinja2
import openai

from enum import Enum
from pydantic import BaseModel
from .bot_outputs import _D, _I, _W, _E, _F, _B, _M


class ChatMessages:
    def __init__(self, contexts=None, templates=None, display_message=True):
        self.messages = []
        self.contexts = contexts
        self.templates = templates
        self.display_message = display_message
        if self.templates is not None:
            self.jinja_env = jinja2.Environment(
                loader=jinja2.DictLoader(self.templates), trim_blocks=True, lstrip_blocks=True
            )
        else:
            self.jinja_env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        self.jinja_env.filters["json"] = self._json

    def _json(self, obj):

        def _default(o):
            if isinstance(o, BaseModel):
                return o.model_dump()
            if isinstance(o, Enum):
                return o.value
            return repr(o)

        return json.dumps(obj, indent=2, ensure_ascii=False, default=_default)

    def add(self, content, role="user", content_type="text", tpl_context=None):
        tpl_context = tpl_context or self.contexts
        if content_type == "text" and tpl_context is not None:
            content = self.jinja_env.from_string(content).render(**tpl_context)
        if content_type == "text":
            content_key = "text"
        else:
            raise NotImplementedError
        _D("Adding message: role={}, content_type={}".format(role, content_type))
        if self.display_message:
            _B(content, title="Chat Message {}: {}".format(role, content_type))
        if len(self.messages) == 0 or self.messages[-1]["role"] != role:
            self.messages.append({"role": role, "content": [{"type": content_type, content_key: content}]})
        else:
            self.messages[-1]["content"].append({"type": content_type, content_key: content})

    def get(self):
        return self.messages

    def clear(self):
        self.messages = []


class BotChat:
    """聊天混合类，提供聊天相关功能"""

    display_think = True
    display_message = True
    display_response = False

    def __init__(self, base_url, api_key, model_name, **chat_kwargs):
        """初始化聊天混合类"""
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.display_think = chat_kwargs.get("display_think", self.display_think)
        self.display_message = chat_kwargs.get("display_message", self.display_message)
        self.display_response = chat_kwargs.get("display_response", self.display_response)

    def parse_reply(self, reply, ret_think_block=False, ret_empty_block=False, display_reply=True):
        """解析聊天回复"""

        def _read_think_block(tokens):
            text = ""
            while True:
                try:
                    token = next(iter_tokens)
                except StopIteration:
                    break
                if token is None:
                    continue
                if token == "</think>":
                    break
                elif token == "<think>":
                    text += _read_think_block(tokens)
                    text += "</think>"
                # elif token.startswith("```") and len(token) > 3:
                #     text += _read_code_block(tokens)
                # elif token.startswith("```") and len(token) == 3:
                #     text += _read_fence_block(tokens)
                else:
                    text += token
            return text

        def _read_code_block(tokens):
            text = ""
            while True:
                try:
                    token = next(iter_tokens)
                except StopIteration:
                    break
                if token is None:
                    continue
                if token == "```":
                    break
                # elif token == "<think>":
                #     text += _read_think_block(tokens)
                elif token.startswith("```") and len(token) > 3:
                    text += _read_code_block(tokens)
                    text += "```"
                else:
                    text += token
            return text

        def _read_fence_block(tokens):
            text = ""
            while True:
                try:
                    token = next(iter_tokens)
                except StopIteration:
                    break
                if token is None:
                    continue
                if token == "```":
                    break
                # elif token == "<think>":
                #     text += _read_think_block(tokens)
                elif token.startswith("```") and len(token) > 3:
                    text += _read_code_block(tokens)
                    text += "```"
                else:
                    text += token
            return text

        tokens = re.split(r"(<think>)|(</think>)|(```[a-zA-Z_0-9]+)|(```)", reply)
        iter_tokens = iter(tokens)
        while True:
            try:
                token = next(iter_tokens)
            except StopIteration:
                break
            if token:
                if token == "<think>":
                    think_block = _read_think_block(iter_tokens)
                    raw_think_block = token + think_block + "</think>"
                    if (self.display_think or display_reply) and think_block and think_block.strip():
                        _B(think_block, title="Thought Block")
                    if ret_think_block and (ret_empty_block or think_block and think_block.strip()):
                        yield {"type": "think", "content": think_block, "raw": raw_think_block}
                elif token.startswith("```") and len(token) > 3:
                    content = _read_code_block(iter_tokens)
                    raw_content = token + content + "```"
                    lang = token[3:].lower()
                    if display_reply and content and content.strip():
                        _B(content, title="Code Block", format="code", code_language=lang)
                    if ret_empty_block or content and content.strip():
                        yield {"type": "code", "lang": lang, "content": content, "raw": raw_content}
                elif token.startswith("```") and len(token) == 3:
                    content = _read_fence_block(iter_tokens)
                    raw_content = token + content + "```"
                    if display_reply and content and content.strip():
                        _B(content, title="Fence Block", format="code", code_language="text")
                    if ret_empty_block or content and content.strip():
                        yield {"type": "fence", "content": content, "raw": raw_content}
                else:
                    is_json_block = False
                    if (
                        token.strip().startswith("{")
                        and token.strip().endswith("}")
                        or token.strip().startswith("[")
                        and token.strip().endswith("]")
                    ):
                        try:
                            json.loads(token)
                            _I(f"Got JSON Block from text: {repr(token)[:80]}")
                            is_json_block = True
                            if display_reply and token and token.strip():
                                _B(token, title="JSON Block", format="code", code_language="json")
                            if ret_empty_block or token and token.strip():
                                yield {"type": "code", "lang": "json", "content": token.strip(), "raw": token}
                        except json.JSONDecodeError:
                            _I(f"Got non-JSON Block from text: {repr(token)[:80]}")
                    if not is_json_block:
                        if display_reply and token and token.strip():
                            _M(token)
                        if ret_empty_block or token and token.strip():
                            yield {"type": "text", "content": token, "raw": token}

    def create_messages(self, contexts=None, templates=None):
        return ChatMessages(contexts=contexts, templates=templates, display_message=self.display_message)

    def chat(
        self,
        messages,
        ret_think_block=False,
        ret_empty_block=False,
        display_reply=True,
        max_tokens=32 * 1024,
        max_completion_tokens=4 * 1024,
        temperature=0.8,
        n=1,
        **kwargs,
    ):
        """发送聊天请求"""
        sizes = [len(content["text"]) for content in messages[0]["content"]]
        total_size = sum(sizes)
        _D("Total message size: {} chars, {}".format(total_size, sizes))
        _I("Connecting to OpenAI API: {}".format(self.base_url or "default"))
        openai_client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        _I("Sending request to OpenAI API, model: {}".format(self.model_name))
        response = openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            n=n,
            **kwargs,
        )
        if not response.choices or not response.choices[0].message:
            _E("No valid response from OpenAI API")
            return []
        else:
            _I("Received response from OpenAI API")
            _D("Response content: " + repr(response.choices[0].message.content)[:50])
            if self.display_response:
                _B(response.choices[0].message.content, title="Chat Response")
            reply = response.choices[0].message.content
            return list(
                self.parse_reply(
                    reply,
                    ret_think_block=ret_think_block,
                    ret_empty_block=ret_empty_block,
                    display_reply=display_reply,
                )
            )
