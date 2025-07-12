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

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from IPython.display import display as ipython_display, Markdown
from IPython.core.getipython import get_ipython
from IPython.core.displaypub import CapturingDisplayPublisher
from IPython.core.displayhook import CapturingDisplayHook
from IPython.utils.capture import capture_output, CapturedIO
from IPython.utils.io import Tee


class CloselessStringIO(io.StringIO):
    def close(self):
        pass

    def __del__(self):
        super().close()
        return super().__del__()


class TeeCapturingDisplayPublisher(CapturingDisplayPublisher):

    def __init__(self, *args, original_display_pub=None, **kwargs):
        super().__init__(*args, **kwargs)
        ipy = get_ipython()
        self.original_display_pub = original_display_pub or (ipy.display_pub if ipy is not None else None)

    def publish(self, *args, **kwargs):
        super().publish(*args, **kwargs)
        if self.original_display_pub is not None:
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
            if self.shell is not None:
                self.save_display_pub = self.shell.display_pub
                self.shell.display_pub = TeeCapturingDisplayPublisher()
                outputs = self.shell.display_pub.outputs
                self.save_display_hook = sys.displayhook
                sys.displayhook = TeeCapturingDisplayHook(shell=self.shell, outputs=outputs)
            else:
                self.save_display_pub = None
                outputs = None

        return CapturedIO(stdout, stderr, outputs)


def indent(text: str, indent: int = 4) -> str:
    return "\n".join(f"{' ' * indent}{line}" for line in text.split("\n"))


def no_indent(text: str) -> str:
    return re.sub(r"^\s+", "", text, flags=re.MULTILINE)


def no_wrap(text: str) -> str:
    return re.sub(r"\s+", " ", text, flags=re.MULTILINE)


def no_newline(text: str) -> str:
    return re.sub(r"\n+", " ", text, flags=re.MULTILINE)


def no_space(text: str) -> str:
    return re.sub(r"\s+", "", text, flags=re.MULTILINE)


class EnvironmentCapbilities(BaseModel):
    save_metadata: bool = False
    user_confirm: bool = False
    user_supply_info: bool = False
    set_cell_content: bool = False


__env_capbilities = EnvironmentCapbilities()


def get_env_capbilities() -> EnvironmentCapbilities:
    return __env_capbilities


def set_env_capbilities(env_capbilities: EnvironmentCapbilities):
    global __env_capbilities

    __env_capbilities = env_capbilities
