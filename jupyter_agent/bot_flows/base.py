"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import time
import traceback

from pydantic import BaseModel
from enum import Enum
from typing import List, Dict, Optional, Type
from IPython.display import Markdown
from ..bot_agents.base import BaseAgent
from ..bot_outputs import _D, _I, _W, _E, _F, _M, _B
from ..bot_outputs import set_stage, flush_output, output_evaluation
from ..bot_outputs import FlowEvalutionRecord, StageEvalutionRecord, NotebookEvalutionRecord

TASK_AGENT_STATE_ERROR = "_AGENT_STATE_ERROR_32534526_"
TASK_STAGE_START = "start"
TASK_STAGE_COMPLETED = "completed"
TASK_STAGE_GLOBAL_FINISHED = "global_finished"


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
    agent: Type[BaseAgent] | str
    states: Dict[AS | str, StageNext[ST] | List[StageNext[ST]] | Dict[TaskAction, StageNext[ST]] | ST | str] = {}
    next_stage: Optional[StageNext[ST] | List[StageNext[ST]] | Dict[TaskAction, StageNext[ST]] | ST | str] = None


class BaseTaskFlow:
    """
    基础任务流程
    """

    STAGE_TRANSITIONS: List[StageTransition] = []
    START_STAGE = TASK_STAGE_START
    STOP_STAGES = [TASK_STAGE_COMPLETED, TASK_STAGE_GLOBAL_FINISHED]

    def __init__(self, notebook_context, agent_factory):
        self.notebook_context = notebook_context
        self.agent_factory = agent_factory
        self.stage_transitions = {}
        self.prepare_stage_transitions()

    @property
    def task(self):
        return self.notebook_context.cur_task

    @property
    def cells(self):
        return self.notebook_context.cells

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

    def __call__(self, stage, max_tries=5, stage_continue=True, stage_confirm=True):

        n_tries = 0
        flow_duration = 0.0
        stage_count = 0
        # Initialize the task stage
        stage = stage or self.START_STAGE
        while n_tries <= max_tries:
            stage_st = time.time()
            try:
                stage_name = stage.value if isinstance(stage, Enum) else stage
                stage_name = stage_name.replace(".", "-").capitalize()
                set_stage(stage_name)
                agent = self.get_stage_agent(stage)
                _M(f"**Executing** stage `{stage}` with agent `{type(agent).__name__}` ...")
                failed, state = agent()
            except Exception as e:
                _M(f"**Error** during task execution stage `{stage}`: `{type(e)}`: `{e}`")
                _M(f"```python\n{traceback.format_exc()}\n```")
                state = TASK_AGENT_STATE_ERROR
                failed = True
            stage_count += 1
            stage_duration = time.time() - stage_st
            flow_duration += stage_duration
            _M(f"Stage `{stage}` completed in {stage_duration:.2f} seconds with state `{state}` and failed `{failed}`")
            output_evaluation(
                StageEvalutionRecord(
                    cell_index=self.task.cell_idx,
                    flow=type(self).__name__,
                    stage=str(stage),
                    execution_duration=stage_duration,
                    is_success=not failed,
                )
            )

            if state != TASK_AGENT_STATE_ERROR:
                # Agent did not fail, check if we have reached the final stage
                next_stage = self.get_next_stage(stage, state, TaskAction.CONTINUE)
                self.task.agent_stage = next_stage
                self.task.update_cell()
                if next_stage in self.STOP_STAGES:
                    _M(f"Task execution **Stopped** at stage `{next_stage}`")
                    stage = next_stage
                    break

            if failed:
                # Agent failed
                n_tries += 1
                if n_tries > max_tries:
                    _M(f"**Max flow tries reached** during task execution stage `{stage}`, **Stop!**")
                    break

            if stage_confirm:
                # We need to confirm
                message = self.get_prompt_message(stage, state, failed)
                _M("**Confirm**: " + message)
                flush_output()
                action = self.match_action(input(message))
                next_stage = self.get_next_stage(stage, state, action)
                self.task.agent_stage = next_stage
                self.task.update_cell()
                if action == TaskAction.STOP:
                    _M(f"Task execution **Stopped**, and set next stage to `{next_stage}`")
                    stage = next_stage
                    break
                else:
                    _M(f"**Action**: `{action}` transits stage to `{next_stage}`")
                    stage = next_stage
            else:
                # transit to the next stage without confirmation
                next_stage = self.get_next_stage(stage, state, TaskAction.CONTINUE)
                self.task.agent_stage = next_stage
                self.task.update_cell()
                _M(f"**Transits** stage to `{next_stage}`")
                stage = next_stage

            if not stage_continue:
                break
        # Finalize the task execution
        stage_name = stage.value if isinstance(stage, Enum) else stage
        if stage_name == TASK_STAGE_GLOBAL_FINISHED:
            _M("Task execution **finished** globally.")
            output_evaluation(NotebookEvalutionRecord(cell_index=self.task.cell_idx, is_success=True))
        elif stage_name == TASK_STAGE_COMPLETED:
            _M(f"Task execution **completed** in {flow_duration:.2f} seconds with {stage_count} stages.")
            output_evaluation(
                FlowEvalutionRecord(
                    cell_index=self.task.cell_idx,
                    flow=type(self).__name__,
                    stage_count=stage_count,
                    execution_duration=flow_duration,
                    is_success=True,
                )
            )
        flush_output()
        return stage
