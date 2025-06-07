"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import io
import os
import re
import sys
import json
import jinja2
import openai

from IPython.display import display as ipython_display, Markdown
from IPython.core.getipython import get_ipython
from IPython.core.displaypub import CapturingDisplayPublisher
from IPython.core.displayhook import CapturingDisplayHook
from IPython.utils.capture import capture_output, CapturedIO
from IPython.utils.io import Tee


REPLY_CELL_CODE = "cell_code"
REPLY_CELL_OUTPUT = "cell_output"
REPLY_CELL_RESULT = "cell_result"
REPLY_CELL_ERROR = "cell_error"
REPLY_TASK_PROMPT = "task_prompt"
REPLY_TASK_RESULT = "task_result"
REPLY_TASK_ISSUE = "task_issue"
REPLY_DEBUG = "debug"
REPLY_THINK = "think"
REPLY_CODE = "code"
REPLY_FENCE = "fence"
REPLY_TEXT = "text"


class CloselessStringIO(io.StringIO):
    def close(self):
        pass

    def __del__(self):
        super().close()
        return super().__del__()


class TeeCapturingDisplayPublisher(CapturingDisplayPublisher):

    def __init__(self, *args, original_display_pub=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_display_pub = original_display_pub or get_ipython().display_pub

    def publish(self, *args, **kwargs):
        super().publish(*args, **kwargs)
        self.original_display_pub.publish(*args, **kwargs)


class TeeCapturingDisplayHook(CapturingDisplayHook):

    def __init__(self, *args, original_display_hook=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_display_hook = original_display_hook or sys.displayhook

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        self.original_display_hook(*args, **kwargs)


class TeeOutputCapture(capture_output):

    def __enter__(self):

        self.sys_stdout = sys.stdout
        self.sys_stderr = sys.stderr

        if self.display:
            self.shell = get_ipython()
            if self.shell is None:
                self.save_display_pub = None
                self.display = False

        stdout = stderr = outputs = None
        if self.stdout:
            stdout = CloselessStringIO()
            sys.stdout = Tee(stdout, channel="stdout")
        if self.stderr:
            stderr = CloselessStringIO()
            sys.stderr = Tee(stderr, channel="stderr")
        if self.display:
            self.save_display_pub = self.shell.display_pub
            self.shell.display_pub = TeeCapturingDisplayPublisher()
            outputs = self.shell.display_pub.outputs
            self.save_display_hook = sys.displayhook
            sys.displayhook = TeeCapturingDisplayHook(shell=self.shell, outputs=outputs)

        return CapturedIO(stdout, stderr, outputs)


def display(obj, reply_from=None, reply_type=None, exclude_from_context=False, **kwargs):
    """自定义的 display 函数，用于在 Jupyter 中显示对象"""
    assert "metadata" not in kwargs
    metadata = {"reply_from": reply_from, "reply_type": reply_type, "exclude_from_context": exclude_from_context}
    metadata.update(kwargs)
    return ipython_display(obj, metadata=metadata)


class DisplayMixin:
    """显示混合类，提供显示功能"""

    def _D(self, obj, reply_from=None, reply_type=None, exclude_from_context=True, **kwargs):
        """显示对象, 默认不添加到上下文中"""
        return display(
            obj,
            reply_from=reply_from or self.__class__.__name__,
            reply_type=reply_type,
            exclude_from_context=exclude_from_context,
            **kwargs,
        )

    def _C(self, obj, reply_from=None, reply_type=None, exclude_from_context=False, **kwargs):
        """显示对象， 添加到上下文中"""
        return display(
            obj,
            reply_from=reply_from or self.__class__.__name__,
            reply_type=reply_type,
            exclude_from_context=exclude_from_context,
            **kwargs,
        )


class DebugMixin(DisplayMixin):
    """调试混合类，提供调试输出功能"""

    DEBUG_LEVEL = 0

    def __init__(self, debug_level=None):
        """初始化调试混合类"""
        if debug_level is None:
            debug_level = os.getenv("BOT_MAGIC_DEBUG_LEVEL", self.DEBUG_LEVEL)
        self.set_debug_level(debug_level)

    def debug(self, *args, level=1):
        """打印调试信息"""
        debug_level = self.debug_level if hasattr(self, "debug_level") else self.DEBUG_LEVEL
        if debug_level >= level:
            self._D("DEUBG: " + " ".join("{}".format(arg) for arg in args), reply_type=REPLY_DEBUG)

    def set_debug_level(self, level):
        """设置调试级别"""
        self.debug_level = int(level)


_block_style = """
<style>
.block-panel {
    background-color: rgba(128,128,128,0.2);
    border-radius: 0.5rem;
    width: 90%;
}

.block-title {
    cursor: pointer;
    font-style: italic;
    color: #888888;
    padding: 0.5rem;
}

.block-content {
    padding: 0.5rem;
}

.block-title.collapsed + .block-content {
    display: none;
}
</style>
"""


def markdown_block(block, title="Block"):

    return Markdown(
        _block_style
        + '<div class="block-panel" >'
        + '<div class="block-title collapsed" onclick="this.classList.toggle(\'collapsed\')">'
        + f"{title} (click to expand)"
        + "</div>"
        + '<div class="block-content" >\n\n'
        + block
        + "\n\n</div>"
        + "</div>"
    )


class ChatMessages(DebugMixin):
    def __init__(self, contexts=None, templates=None, display_message=True, debug_level=0):
        DebugMixin.__init__(self, debug_level)
        self.messages = []
        self.contexts = contexts
        self.templates = templates
        self.display_message = display_message
        if self.templates is not None:
            self.jinja_env = jinja2.Environment(loader=jinja2.DictLoader(self.templates))
        else:
            self.jinja_env = jinja2.Environment()
        self.jinja_env.filters["json"] = json.dumps

    def add(self, content, role="user", content_type="text", tpl_context=None):
        tpl_context = tpl_context or self.contexts
        if content_type == "text" and tpl_context is not None:
            content = self.jinja_env.from_string(content).render(**tpl_context)
        if content_type == "text":
            content_key = "text"
        else:
            raise NotImplementedError
        self.debug("Adding message: role={}, content_type={}".format(role, content_type), level=3)
        if self.display_message or self.debug_level >= 3:
            self._D(markdown_block(content, title="Chat {}: {}".format(role, content_type)), reply_type=REPLY_DEBUG)
        if len(self.messages) == 0 or self.messages[-1]["role"] != role:
            self.messages.append({"role": role, "content": [{"type": content_type, content_key: content}]})
        else:
            self.messages[-1]["content"].append({"type": content_type, content_key: content})

    def get(self):
        return self.messages

    def clear(self):
        self.messages = []


class ChatMixin(DebugMixin):
    """聊天混合类，提供聊天相关功能"""

    def __init__(self, base_url, api_key, model_name, dispaly_think=True, display_message=True, debug_level=0):
        """初始化聊天混合类"""
        DebugMixin.__init__(self, debug_level)
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.dispaly_think = dispaly_think
        self.display_message = display_message

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
                    if (self.dispaly_think or display_reply) and think_block and think_block.strip():
                        self._D(markdown_block(think_block, title="Thought Block"), reply_type=REPLY_THINK)
                    if ret_think_block and (ret_empty_block or think_block and think_block.strip()):
                        yield {"type": "think", "content": think_block, "raw": raw_think_block}
                elif token.startswith("```") and len(token) > 3:
                    content = _read_code_block(iter_tokens)
                    raw_content = token + content + "```"
                    lang = token[3:].lower()
                    if display_reply and content and content.strip():
                        self._D(markdown_block(raw_content, title="Code Block"), reply_type=REPLY_CODE)
                    if ret_empty_block or content and content.strip():
                        yield {"type": "code", "lang": lang, "content": content, "raw": raw_content}
                elif token.startswith("```") and len(token) == 3:
                    content = _read_fence_block(iter_tokens)
                    raw_content = token + content + "```"
                    if display_reply and content and content.strip():
                        self._D(markdown_block(raw_content, title="Fence Block"), reply_type=REPLY_FENCE)
                    if ret_empty_block or content and content.strip():
                        yield {"type": "fence", "content": content, "raw": raw_content}
                else:
                    if display_reply and token and token.strip():
                        self._D(Markdown(token), reply_type=REPLY_TEXT)
                    if ret_empty_block or token and token.strip():
                        yield {"type": "text", "content": token, "raw": token}

    def create_messages(self, contexts=None, templates=None):
        return ChatMessages(
            contexts=contexts,
            templates=templates,
            display_message=self.display_message,
            debug_level=self.debug_level,
        )

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
        self.debug("Total message size: {} chars, {}".format(total_size, sizes))
        self.debug("Connecting to OpenAI API: {}".format(self.base_url or "default"))
        openai_client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.debug("Sending request to OpenAI API, model: {}".format(self.model_name))
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
            self.debug("No valid response from OpenAI API")
            return []
        else:
            self.debug("Received response from OpenAI API")
            self.debug("Response content:", repr(response.choices[0].message.content)[:50], level=3)
            if self.debug_level >= 3:
                self._D(markdown_block(response.choices[0].message.content, title="Response"), reply_type=REPLY_DEBUG)
            reply = response.choices[0].message.content
            return list(
                self.parse_reply(
                    reply,
                    ret_think_block=ret_think_block,
                    ret_empty_block=ret_empty_block,
                    display_reply=display_reply,
                )
            )
