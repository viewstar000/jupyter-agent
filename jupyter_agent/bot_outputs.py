"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import json
import time
import datetime
import jinja2

from enum import Enum
from typing import Optional, Dict, List, Tuple, Any, Type
from pydantic import BaseModel, Field
from IPython.display import display, Markdown
from .bot_evaluation import BaseEvaluationRecord
from .bot_actions import ActionBase
from .utils import no_indent, no_wrap

STAGE_SWITCHER_SCRIPT = no_wrap(
    """
this.parentElement.parentElement.querySelectorAll('.agent-stage-title').forEach(function(item) {
    item.classList.remove('active');
});
this.parentElement.parentElement.querySelectorAll('.agent-stage-output-panel').forEach(function(item) {
    item.classList.remove('active');
});
this.parentElement.parentElement.querySelectorAll('.agent-stage-{{ stage }}').forEach(function(item) {
    item.classList.add('active');
});
"""
)


AGENT_OUTPUT_TEMPLEATE = no_indent(
    """
<style>
.agent-output-panel * {
    box-sizing: border-box;
}

.agent-output-panel {
    background-color: rgba(128,128,128,0.1);
    border-radius: 0.5rem; 
}

.agent-output-title {
    cursor: pointer;
    font-style: italic;
    color: #888888;
    padding: 0.5rem;
}

.agent-output-content {
    width: unset;
    padding: 0.5rem;
    padding-top: 0;
}

.agent-output-title.collapsed + .agent-output-content {
    display: none;
}

.agent-stage-switcher {
    background-color: rgba(128,128,128,0.1);
    border-bottom: 2px solid rgba(128,128,128,0.3);
    padding: 0.5rem 0;
}

.agent-stage-title {
    cursor: pointer;
    font-style: italic;
    color: #888888;
    padding: 0.5rem;
}

.agent-stage-title.active {
    background-color: rgba(128,128,128,0.3);
}

.agent-stage-output-panel {
    display: none;
}

.agent-stage-output-panel.active {
    display: block;
    margin-top: 0.5rem;
    overflow: auto;
}

.agent-output-block {
    background-color: rgba(128,128,128,0.2);
    border-radius: 0.5rem;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}

.agent-output-block-title {
    cursor: pointer;
    font-style: italic;
    color: #888888;
    padding: 0.5rem;
}

.agent-output-block-content {
    width: unset;
    padding: 0.5rem;
}

.agent-output-block-title.collapsed + .agent-output-block-content {
    display: none;
}
</style>

<div class="agent-output-panel">
    <div class="agent-output-title {{ 'collapsed' if collapsed else ''}}" onclick="this.classList.toggle('collapsed')">
        {{ title if title else 'Agent Output' }} - {{ active_stage }}
    </div>
    <div class="agent-output-content">
        <div class="agent-stage-switcher">
            {% for stage in contents.keys() %}
            <span class="agent-stage-title agent-stage-{{ stage }} {{ 'active' if stage == active_stage }}" onclick="{% include 'switcher_script' %}">
                {{ stage }}
            </span>
            {% endfor %}
        </div>
        {% for stage, contents in contents.items() %}
        <div class="agent-stage-output-panel agent-stage-{{ stage }} {{ 'active' if stage == active_stage }}">
            {% if stage == 'Logging' +%}
                ```log
                {% for content in contents %}
                    {{ content['content'] }}
                {% endfor %}
                ```
            {% else %}
                {% for content in contents %}
                    {% if content['type'] == 'block' %}
                    <div class="agent-output-block">
                        <div class="agent-output-block-title {{ 'collapsed' if content['collapsed'] else ''}}" onclick="this.classList.toggle('collapsed')">
                            {{ content['title'] }}
                        </div>
                        <div class="agent-output-block-content">
                            {% if content['format'] == 'markdown' +%}
                                {{ content['content'] }}
                            {%+ elif content['format'] == 'code' +%}
                                ```{{ content['code_language'] }}
                                {{ content['content'] }}
                                ```
                            {%+ endif %}
                        </div>
                    </div>
                    {% elif content['type'] == 'markdown' +%}
                        {{ content['content'] }}
                    {% elif content['type'] == 'text' +%}
                        ```{{ content['code_language'] }}
                        {{ content['content'] }}
                        ```
                    {%+ endif %}
                {% endfor %}
            {% endif %}
        </div>
        {% endfor %}
    </div>
</div>
"""
)

LOGGING_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARN": 30,
    "ERROR": 40,
    "FATAL": 50,
}


class AgentOutput:
    """
    AgentOutput 是一个用于在 Jupyter Notebook 中显示 Agent 输出的类。
    """

    def __init__(self, title=None, collapsed=False, logging_level="INFO"):
        self.title = title
        self.collapsed = collapsed
        self.logging_level = (
            logging_level if isinstance(logging_level, int) else LOGGING_LEVELS.get(logging_level.upper(), 20)
        )
        self.jinja_env = jinja2.Environment(
            trim_blocks=True, lstrip_blocks=True, loader=jinja2.DictLoader({"switcher_script": STAGE_SWITCHER_SCRIPT})
        )
        self.template = self.jinja_env.from_string(AGENT_OUTPUT_TEMPLEATE)
        self.handler = None
        self._is_dirty = True
        self._latest_display_tm = 0
        self._contents = {}
        self._active_stage = None
        self._agent_data_timestamp = None
        self._agent_data = {}
        self._logging_records = []
        self._evaluation_records = []
        self._action_records = []

    @property
    def content(self):
        contents = dict(self._contents)
        if self._agent_data:
            contents["Metadata"] = [
                {
                    "type": "text",
                    "content": json.dumps(self.metadata, indent=2, ensure_ascii=False),
                    "code_language": "json",
                }
            ]
        filtered_logs = [log for log in self._logging_records if log["level"] >= self.logging_level]
        if len(filtered_logs) > 0:
            contents["Logging"] = filtered_logs
        return self.template.render(
            title=self.title,
            collapsed=self.collapsed,
            active_stage=self._active_stage,
            contents=contents,
        )

    @property
    def metadata(self):
        metadata = {"reply_type": "AgentOutput", "exclude_from_context": True}
        if self._agent_data:
            metadata.update(
                {
                    "jupyter-agent-data-store": True,
                    "jupyter-agent-data-timestamp": self._agent_data_timestamp,
                    "jupyter-agent-data": self._agent_data,
                }
            )
        if self._evaluation_records:
            metadata["jupyter-agent-evaluation-records"] = [record.model_dump() for record in self._evaluation_records]
        if self._action_records:
            metadata["jupyter-agent-action-records"] = [record.model_dump() for record in self._action_records]
        return metadata

    def display(self, stage=None, force=False, wait=True):
        if stage is not None and stage != self._active_stage:
            self._active_stage = stage
            self._is_dirty = True
        if not self._is_dirty and not force:
            return
        if not force and time.time() - self._latest_display_tm < 1:
            if wait:
                time.sleep(1 - (time.time() - self._latest_display_tm))
            else:
                return
        if self.handler is None:
            self.handler = display(Markdown(self.content), metadata=self.metadata, display_id=True)
        else:
            self.handler.update(Markdown(self.content), metadata=self.metadata)
        self._latest_display_tm = time.time()
        self._is_dirty = False

    def clear(self, stage=None, clear_metadata=False):
        if stage is None:
            self._contents = {}
        else:
            self._contents[stage] = []
        if clear_metadata:
            self._agent_data = {}
        self._is_dirty = True
        self.display(force=False, wait=False)

    def output_block(
        self, content, title="Block", collapsed=True, stage=None, format="markdown", code_language="python"
    ):
        if stage is None:
            stage = self._active_stage
        if stage not in self._contents:
            self._contents[stage] = []
        self._contents[stage].append(
            {
                "type": "block",
                "title": title,
                "content": content,
                "collapsed": collapsed,
                "format": format,
                "code_language": code_language,
            }
        )
        self._is_dirty = True
        self.display(stage, force=False, wait=False)

    def output_text(self, content, stage=None, code_language="python"):
        if stage is None:
            stage = self._active_stage
        if stage not in self._contents:
            self._contents[stage] = []
        if (
            len(self._contents[stage]) > 0
            and self._contents[stage][-1]["type"] == "text"
            and self._contents[stage][-1]["code_language"] == code_language
        ):
            self._contents[stage][-1]["content"] += "\n" + content
        else:
            self._contents[stage].append({"type": "text", "content": content, "code_language": code_language})
        self._is_dirty = True
        self.display(stage, force=False, wait=False)

    def output_markdown(self, content, stage=None):
        if stage is None:
            stage = self._active_stage
        if stage not in self._contents:
            self._contents[stage] = []
        self._contents[stage].append({"type": "markdown", "content": content})
        self._is_dirty = True
        self.display(stage, force=False, wait=False)

    def output_agent_data(self, **kwargs):
        self.log(f"output agent data {kwargs}", level="DEBUG")
        self._agent_data.update(kwargs)
        self._agent_data_timestamp = time.time()
        self._is_dirty = True
        self.display(force=False, wait=False)

    def log(self, msg, level="INFO"):
        level = level.upper()
        assert level in LOGGING_LEVELS
        tm = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        level_n = LOGGING_LEVELS[level]
        content = f"[{tm}] {level}: {msg}"
        if self._logging_records and self._logging_records[-1]["level"] == level_n:
            self._logging_records[-1]["content"] += "\n" + content
        else:
            self._logging_records.append(
                {
                    "type": "logging",
                    "level": level_n,
                    "level_name": level,
                    "time": tm,
                    "msg": msg,
                    "content": content,
                }
            )
        self._is_dirty = True
        self.display(force=False, wait=False)

    def log_evaluation(self, record: BaseEvaluationRecord):
        assert isinstance(
            record, BaseEvaluationRecord
        ), "record must be an instance of BaseEvalutionRecord or its subclass"
        if record.timestamp == 0:
            record.timestamp = time.time()
        self._evaluation_records.append(record)
        self.log(
            f"Evaluation: {record.eval_type}[{record.cell_index}] duration: {record.execution_duration:.2f}s "
            f"success: {record.is_success} correct: {record.correct_score:.2f}",
            level="INFO",
        )
        self._is_dirty = True
        self.display(force=False, wait=False)

    def log_action(self, record: ActionBase):
        assert isinstance(record, ActionBase), "record must be an instance of BaseActionRecord or its subclass"
        if record.timestamp == 0:
            record.timestamp = time.time()
        self._action_records.append(record)
        self.log(f"Action: {record.action} from {record.source}", level="INFO")
        self._is_dirty = True
        self.display(force=False, wait=False)


__agent_output = None


def get_output():
    global __agent_output
    if __agent_output is None:
        __agent_output = AgentOutput()
    return __agent_output


def set_stage(stage):
    get_output().display(stage)


def reset_output(title=None, collapsed=False, stage=None, logging_level="INFO"):
    global __agent_output
    __agent_output = AgentOutput(title, collapsed, logging_level)
    if stage is not None:
        __agent_output.display(stage)
    return __agent_output


def log(msg, level="INFO"):
    get_output().log(msg, level)


def output_block(content, title="Block", collapsed=True, stage=None, format="markdown", code_language="python"):
    get_output().output_block(content, title, collapsed, stage, format, code_language)


def output_text(content, stage=None, code_language="python"):
    get_output().output_text(content, stage, code_language)


def output_markdown(content, stage=None):
    get_output().output_markdown(content, stage)


def output_agent_data(**kwargs):
    get_output().output_agent_data(**kwargs)


def output_evaluation(record: BaseEvaluationRecord):
    get_output().log_evaluation(record)


def output_action(record: ActionBase):
    get_output().log_action(record)


def clear_output(stage=None, clear_metadata=False):
    get_output().clear(stage, clear_metadata)


def flush_output(force=False):
    get_output().display(force=force, wait=True)


def set_title(title):
    get_output().title = title


def set_collapsed(collapsed):
    get_output().collapsed = collapsed


def set_logging_level(logging_level):
    if isinstance(logging_level, int):
        get_output().logging_level = logging_level
    else:
        logging_level = logging_level.upper()
        assert logging_level in LOGGING_LEVELS
        get_output().logging_level = LOGGING_LEVELS[logging_level]


class ReplyType(str, Enum):
    CELL_CODE = "cell_code"
    CELL_OUTPUT = "cell_output"
    CELL_RESULT = "cell_result"
    CELL_ERROR = "cell_error"
    TASK_PROMPT = "task_prompt"
    TASK_RESULT = "task_result"
    TASK_ISSUE = "task_issue"
    DEBUG = "debug"
    THINK = "think"
    CODE = "code"
    FENCE = "fence"
    TEXT = "text"


def agent_display(obj, reply_type=None, exclude_from_context=False, **kwargs):
    """自定义的 display 函数，用于在 Jupyter 中显示对象"""
    assert "metadata" not in kwargs
    metadata = {"reply_type": reply_type, "exclude_from_context": exclude_from_context}
    metadata.update(kwargs)
    return display(obj, metadata=metadata)


_block_style = """
<style>
.block-panel * {
    box-sizing: border-box;
}

.block-panel {
    background-color: rgba(128,128,128,0.2);
    border-radius: 0.5rem;
    box-sizing: border-box;
}

.block-title {
    cursor: pointer;
    font-style: italic;
    color: #888888;
    padding: 0.5rem;
}

.block-content {
    width: unset;
    padding: 0.5rem;
}

.block-title.collapsed + .block-content {
    display: none;
}
</style>
"""


def markdown_block(block, title="Block", collapsed=True):

    default_state = "collapsed" if collapsed else ""

    return Markdown(
        _block_style
        + '<div class="block-panel" >'
        + f'<div class="block-title {default_state}" onclick="this.classList.toggle(\'collapsed\')">'
        + f"{title} (click to expand)"
        + "</div>"
        + '<div class="block-content" >\n\n'
        + block
        + "\n\n</div>"
        + "</div>"
    )


_O = lambda obj, reply_type=None, **kwargs: agent_display(
    obj, reply_type=reply_type, exclude_from_context=True, **kwargs
)


_C = lambda obj, reply_from=None, reply_type=None, **kwargs: agent_display(
    obj, reply_type=reply_type, exclude_from_context=False, **kwargs
)

_M = output_markdown
_T = output_text
_B = output_block
_A = output_agent_data
_L = log
_D = lambda msg: log(msg, level="DEBUG")
_I = lambda msg: log(msg, level="INFO")
_W = lambda msg: log(msg, level="WARN")
_E = lambda msg: log(msg, level="ERROR")
_F = lambda msg: log(msg, level="FATAL")
