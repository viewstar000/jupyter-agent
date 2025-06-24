"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from enum import Enum
from .base import (
    BaseTaskFlow,
    StageTransition,
    StageNext,
    TaskAction,
    TASK_STAGE_COMPLETED,
    TASK_STAGE_GLOBAL_FINISHED,
)
from ..bot_agents import (
    TaskPlannerAgentV3,
    TaskCodingAgent,
    CodeDebugerAgent,
    CodeExecutor,
    TaskStructureSummaryAgent,
    TaskStructureReasoningAgent,
    OutputTaskResult,
)
from ..bot_agents.task_planner_v3 import TaskPlannerState


class TaskStage(str, Enum):
    PLANNING = "planning"
    PLANNING_PAUSED = "planning_paused"
    CODING = "coding"
    EXECUTING = "executing"
    DEBUGGING = "debugging"
    REASONING = "reasoning"
    SUMMARY = "summary"
    OUTPUT_RESULT = "output_result"
    COMPLETED = TASK_STAGE_COMPLETED
    GLOBAL_FINISHED = TASK_STAGE_GLOBAL_FINISHED


class TaskExecutorFlowV3(BaseTaskFlow):

    START_STAGE = TaskStage.PLANNING
    STOP_STAGES = [TaskStage.COMPLETED, TaskStage.PLANNING_PAUSED, TaskStage.GLOBAL_FINISHED]
    STAGE_TRANSITIONS = [
        StageTransition[TaskStage, TaskPlannerState](
            stage=TaskStage.PLANNING,
            agent=TaskPlannerAgentV3,
            states={
                TaskPlannerState.CODING_PLANNED: TaskStage.CODING,
                TaskPlannerState.REASONING_PLANNED: TaskStage.REASONING,
                TaskPlannerState.REQUEST_INFO: TaskStage.PLANNING_PAUSED,
                TaskPlannerState.GLOBAL_FINISHED: TaskStage.GLOBAL_FINISHED,
            },
        ),
        StageTransition[TaskStage, TaskPlannerState](
            stage=TaskStage.PLANNING_PAUSED,
            agent=TaskPlannerAgentV3,
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
            stage=TaskStage.REASONING, agent=TaskStructureReasoningAgent, next_stage=TaskStage.COMPLETED
        ),
        StageTransition[TaskStage, None](
            stage=TaskStage.SUMMARY,
            agent=TaskStructureSummaryAgent,
            next_stage={
                TaskAction.DEFAULT: StageNext(stage=TaskStage.COMPLETED),
                TaskAction.STOP: StageNext(stage=TaskStage.EXECUTING),
            },
        ),
        StageTransition[TaskStage, bool](
            stage=TaskStage.COMPLETED,
            agent=CodeExecutor,
            states={True: TaskStage.OUTPUT_RESULT, False: TaskStage.DEBUGGING},
        ),
        StageTransition[TaskStage, None](
            stage=TaskStage.OUTPUT_RESULT, agent=OutputTaskResult, next_stage=TaskStage.COMPLETED
        ),
        StageTransition[TaskStage, None](
            stage=TaskStage.GLOBAL_FINISHED, agent=OutputTaskResult, next_stage=TaskStage.GLOBAL_FINISHED
        ),
    ]
