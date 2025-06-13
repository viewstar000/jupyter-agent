"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from enum import Enum
from .base import BaseTaskFlow, StageTransition, StageNext, TaskAction
from ..bot_agents import (
    TaskPlannerAgentHuge,
    TaskCodingAgent,
    CodeDebugerAgent,
    CodeExecutor,
    TaskSummaryAgent,
    TaskReasoningAgent,
)
from ..bot_agents.task_planner_v2 import TaskPlannerState
from ..bot_agents.task_verify_summarier import TaskSummaryState


class TaskStage(str, Enum):
    PLANNING = "planning"
    PLANNING_PAUSED = "planning_paused"
    CODING = "coding"
    EXECUTING = "executing"
    DEBUGGING = "debugging"
    REASONING = "reasoning"
    SUMMARY = "summary"
    COMPLETED = "completed"


class TaskExecutorFlowV2(BaseTaskFlow):

    START_STAGE = TaskStage.PLANNING
    STOP_STAGES = [TaskStage.COMPLETED, TaskStage.PLANNING_PAUSED]
    STAGE_TRANSITIONS = [
        StageTransition[TaskStage, TaskPlannerState](
            stage=TaskStage.PLANNING,
            agent=TaskPlannerAgentHuge,
            states={
                TaskPlannerState.CODING_PLANNED: TaskStage.CODING,
                TaskPlannerState.REASONING_PLANNED: TaskStage.REASONING,
                TaskPlannerState.REQUEST_INFO: TaskStage.PLANNING_PAUSED,
                TaskPlannerState.GLOBAL_FINISHED: TaskStage.COMPLETED,
            },
        ),
        StageTransition[TaskStage, TaskPlannerState](
            stage=TaskStage.PLANNING_PAUSED,
            agent=TaskPlannerAgentHuge,
            states={
                TaskPlannerState.CODING_PLANNED: TaskStage.CODING,
                TaskPlannerState.REASONING_PLANNED: TaskStage.REASONING,
                TaskPlannerState.REQUEST_INFO: TaskStage.PLANNING_PAUSED,
                TaskPlannerState.GLOBAL_FINISHED: TaskStage.COMPLETED,
            },
        ),
        StageTransition[TaskStage, None](
            stage=TaskStage.CODING, agent=TaskCodingAgent, next_stage=TaskStage.EXECUTING
        ),
        StageTransition[TaskStage, bool](
            stage=TaskStage.EXECUTING,
            agent=CodeExecutor,
            states={True: TaskStage.SUMMARY, False: TaskStage.DEBUGGING},
        ),
        StageTransition[TaskStage, None](
            stage=TaskStage.DEBUGGING, agent=CodeDebugerAgent, next_stage=TaskStage.EXECUTING
        ),
        StageTransition[TaskStage, None](
            stage=TaskStage.REASONING, agent=TaskReasoningAgent, next_stage=TaskStage.COMPLETED
        ),
        StageTransition[TaskStage, TaskSummaryState](
            stage=TaskStage.SUMMARY,
            agent=TaskSummaryAgent,
            next_stage={
                TaskAction.DEFAULT: StageNext(stage=TaskStage.COMPLETED),
                TaskAction.STOP: StageNext(stage=TaskStage.EXECUTING),
            },
        ),
        StageTransition[TaskStage, bool](
            stage=TaskStage.COMPLETED,
            agent=CodeExecutor,
            states={True: TaskStage.COMPLETED, False: TaskStage.DEBUGGING},
        ),
    ]
