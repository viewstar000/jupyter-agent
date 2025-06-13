"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from enum import Enum
from .base import BaseTaskFlow, StageTransition, StageNext, TaskAction
from ..bot_agents import (
    TaskPlannerAgent,
    TaskCodingAgent,
    CodeDebugerAgent,
    CodeExecutor,
    TaskVerifyAgent,
    TaskSummaryAgent,
    TaskPlannerState,
    TaskVerifyState,
)


class TaskStage(str, Enum):
    PLANNING = "planning"
    PLANNING_PAUSED = "planning_paused"
    CODING = "coding"
    EXECUTING = "executing"
    DEBUGGING = "debugging"
    VERIFYING = "verifying"
    SUMMARY = "summary"
    COMPLETED = "completed"


class TaskExecutorFlowV1(BaseTaskFlow):

    START_STAGE = TaskStage.PLANNING
    STOP_STAGES = [TaskStage.COMPLETED, TaskStage.PLANNING_PAUSED]
    STAGE_TRANSITIONS = [
        StageTransition[TaskStage, TaskPlannerState](
            stage=TaskStage.PLANNING,
            agent=TaskPlannerAgent,
            states={
                TaskPlannerState.PLANNED: TaskStage.CODING,
                TaskPlannerState.REQUEST_INFO: TaskStage.PLANNING_PAUSED,
                TaskPlannerState.GLOBAL_FINISHED: TaskStage.COMPLETED,
            },
        ),
        StageTransition[TaskStage, TaskPlannerState](
            stage=TaskStage.PLANNING_PAUSED,
            agent=TaskPlannerAgent,
            states={
                TaskPlannerState.PLANNED: TaskStage.CODING,
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
            states={True: TaskStage.VERIFYING, False: TaskStage.DEBUGGING},
        ),
        StageTransition[TaskStage, None](
            stage=TaskStage.DEBUGGING, agent=CodeDebugerAgent, next_stage=TaskStage.EXECUTING
        ),
        StageTransition[TaskStage, TaskVerifyState](
            stage=TaskStage.VERIFYING,
            agent=TaskVerifyAgent,
            states={
                TaskVerifyState.PASSED: TaskStage.SUMMARY,
                TaskVerifyState.FAILED: [
                    StageNext[TaskStage](action=TaskAction.CONTINUE, stage=TaskStage.PLANNING),
                    StageNext[TaskStage](action=TaskAction.SKIP, stage=TaskStage.SUMMARY),
                ],
            },
        ),
        StageTransition[TaskStage, None](
            stage=TaskStage.SUMMARY, agent=TaskSummaryAgent, next_stage=TaskStage.COMPLETED
        ),
        StageTransition[TaskStage, bool](
            stage=TaskStage.COMPLETED,
            agent=CodeExecutor,
            states={True: TaskStage.COMPLETED, False: TaskStage.DEBUGGING},
        ),
    ]
