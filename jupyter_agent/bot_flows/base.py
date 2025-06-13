"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import traceback

from pydantic import BaseModel
from enum import Enum
from typing import List, Dict, Optional, Type
from IPython.display import Markdown
from ..utils import DebugMixin
from ..bot_agents.base import BaseTaskAgent

TASK_AGENT_STATE_ERROR = "_AGENT_STATE_ERROR_32534526_"
TASK_STAGE_START = "start"
TASK_STAGE_COMPLETED = "completed"


class TaskAction(str, Enum):
    DEFAULT = "*"
    CONTINUE = "continue"
    RETRY = "retry"
    SKIP = "skip"
    STOP = "stop"


class StageNext[ST](BaseModel):
    action: Optional[TaskAction] = None
    stage: ST | str
    message: str = ""


class StageTransition[ST, AS](BaseModel):
    stage: ST | str
    agent: Type[BaseTaskAgent | DebugMixin] | str
    states: Dict[AS | str, StageNext[ST] | List[StageNext[ST]] | Dict[TaskAction, StageNext[ST]] | ST | str] = {}
    next_stage: Optional[StageNext[ST] | List[StageNext[ST]] | Dict[TaskAction, StageNext[ST]] | ST | str] = None


class BaseTaskFlow(DebugMixin):
    """
    基础任务流程
    """

    STAGE_TRANSITIONS: List[StageTransition] = []
    START_STAGE = TASK_STAGE_START
    STOP_STAGES = [TASK_STAGE_COMPLETED]

    def __init__(self, notebook_context, task_context, agent_factory, debug_level=None):
        DebugMixin.__init__(self, debug_level)
        self.notebook_context = notebook_context
        self.task_context = task_context
        self.agent_factory = agent_factory
        self.stage_transitions = {}
        self.prepare_stage_transitions()

    def prepare_stage_transitions(self):
        for st in self.STAGE_TRANSITIONS:
            assert not (st.next_stage and st.states), "next_stage and states are mutually exclusive"
            self.stage_transitions[st.stage] = st
            if st.next_stage:
                st.states[TaskAction.DEFAULT] = st.next_stage
                st.next_stage = None
            for state, ns in st.states.items():
                st.states[state] = {}
                if isinstance(ns, str):
                    st.states[state] = {TaskAction.DEFAULT: StageNext(stage=ns)}
                elif isinstance(ns, StageNext):
                    action = ns.action or TaskAction.DEFAULT
                    st.states[state] = {action: ns}
                elif isinstance(ns, list):
                    state_dict = {}
                    for n in ns:
                        action = n.action or TaskAction.DEFAULT
                        state_dict[action] = n
                    st.states[state] = state_dict
                elif isinstance(ns, dict):
                    st.states[state] = ns
                else:
                    raise ValueError(f"Unknown next stage: {ns}")
                state_ns: Dict = st.states[state]  # type: ignore
                state_ns[TaskAction.CONTINUE] = state_ns.get(TaskAction.CONTINUE) or state_ns.get(TaskAction.DEFAULT)
                state_ns[TaskAction.RETRY] = state_ns.get(TaskAction.RETRY) or StageNext(stage=st.stage)
                state_ns[TaskAction.STOP] = state_ns.get(TaskAction.STOP) or StageNext(stage=st.stage)
                state_ns[TaskAction.SKIP] = state_ns.get(TaskAction.SKIP) or state_ns.get(TaskAction.CONTINUE)
            if TASK_AGENT_STATE_ERROR not in st.states:
                st.states[TASK_AGENT_STATE_ERROR] = {"*": StageNext(stage=st.stage)}

    def get_stage_agent(self, stage):
        for t in self.STAGE_TRANSITIONS:
            if t.stage == stage:
                return self.agent_factory(t.agent)
        raise ValueError(f"No agent for stage `{stage}`")

    def _get_next_stage_trans(self, stage, state, action=TaskAction.CONTINUE):

        st = self.stage_transitions.get(stage)
        if st:
            state_ns = st.states.get(state) or st.states.get("*")
            assert state_ns, f"No next stage for stage `{stage}` and state `{state}`"
            act_ns = state_ns.get(action) or state_ns.get("*")
            assert act_ns, f"No next stage for stage `{stage}`, state `{state}`, action `{action}`"
            return act_ns
        else:
            raise ValueError(f"No next stage for stage `{stage}`")

    def get_prompt_message(self, stage, state, failed):

        ns = self._get_next_stage_trans(stage, state)
        if failed:
            msg = ns.message or f"Staget `{stage}` FAILED!"
            return (
                f"{msg}\n Continue from stage `{ns.stage}`? \n"
                f"(C)ontinue, (R)etry, s(K)ip, (S)top, default `continue`"
            )
        else:
            return (
                f"{ns.message}\n Continue to stage `{ns.stage}`? \n"
                f"(C)ontinue, (R)etry, s(K)ip, (S)top, default `continue`"
            )

    def match_action(self, input):
        input = input.lower().strip()
        if input == "" or input == "c" or (len(input) > 1 and TaskAction.CONTINUE.value.startswith(input)):
            return TaskAction.CONTINUE
        elif input == "r" or (len(input) > 1 and TaskAction.RETRY.value.startswith(input)):
            return TaskAction.RETRY
        elif input == "k" or (len(input) > 1 and TaskAction.SKIP.value.startswith(input)):
            return TaskAction.SKIP
        elif input == "s" or (len(input) > 1 and TaskAction.STOP.value.startswith(input)):
            return TaskAction.STOP
        else:
            raise ValueError(f"Unknown action: {input}")

    def get_next_stage(self, stage, state, action):

        ns = self._get_next_stage_trans(stage, state, action)
        return ns.stage

    def __call__(self, stage, max_tries=3, stage_continue=True, stage_confirm=True):

        n_tries = 0
        stage = stage or self.START_STAGE
        while n_tries <= max_tries:

            try:
                agent = self.get_stage_agent(stage)
                self._D(Markdown(f"**Executing** stage `{stage}` with agent `{type(agent).__name__}` ..."))
                failed, state = agent()
            except Exception as e:
                self._D(Markdown(f"**Error** during task execution stage `{stage}`: `{type(e)}`: `{e}`"))
                self._D(Markdown(f"```python\n{traceback.format_exc()}\n```"))
                state = TASK_AGENT_STATE_ERROR
                failed = True

            if state != TASK_AGENT_STATE_ERROR:
                # Agent did not fail, check if we have reached the final stage
                next_stage = self.get_next_stage(stage, state, TaskAction.CONTINUE)
                self.task_context.task_stage = next_stage
                self.task_context.update_bot_cell()
                if next_stage in self.STOP_STAGES:
                    self._D(Markdown(f"Task execution **Stopped** at stage `{next_stage}`"))
                    break

            if failed:
                # Agent failed
                n_tries += 1

            if failed or stage_confirm:
                # Agent failed or we need to confirm
                message = self.get_prompt_message(stage, state, failed)
                self._D(Markdown("**Confirm**: " + message))
                action = self.match_action(input(message))
                next_stage = self.get_next_stage(stage, state, action)
                self.task_context.task_stage = next_stage
                self.task_context.update_bot_cell()
                if action == TaskAction.STOP:
                    self._D(Markdown(f"Task execution **Stopped**, and set next stage to `{next_stage}`"))
                    break
                elif n_tries > max_tries:
                    self._D(Markdown(f"**Max tries reached** during task execution stage `{stage}`, **Stop!**"))
                    break
                else:
                    self._D(Markdown(f"**Action**: `{action}` transits stage to `{next_stage}`"))
                    stage = next_stage
            else:
                # Agent succeeded, transit to the next stage without confirmation
                next_stage = self.get_next_stage(stage, state, TaskAction.CONTINUE)
                self.task_context.task_stage = next_stage
                self.task_context.update_bot_cell()
                self._D(Markdown(f"**Transits** stage to `{next_stage}`"))
                stage = next_stage

            if not stage_continue:
                break

        return stage
