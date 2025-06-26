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
    RequestAboveUserSupplyAgent,
    RequestBelowUserSupplyAgent,
)
from ..bot_agents.task_planner_v3 import TaskPlannerState
from ..bot_agents.task_structrue_reasoner import TaskStructureReasonState
from ..bot_agents.task_structrue_summarier import TaskStructureSummaryState


class TaskStage(str, Enum):
    PLANNING = "planning"
    PLANNING_PAUSED = "planning_paused"
    REQUEST_INFO_ABOVE = "request_info_above"
    REQUEST_INFO_BELOW = "request_info_below"
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
                TaskPlannerState.REQUEST_INFO: TaskStage.REQUEST_INFO_ABOVE,
                TaskPlannerState.GLOBAL_FINISHED: TaskStage.GLOBAL_FINISHED,
            },
        ),
        StageTransition[TaskStage, None](
            stage=TaskStage.REQUEST_INFO_ABOVE, agent=RequestAboveUserSupplyAgent, next_stage=TaskStage.PLANNING_PAUSED
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
        StageTransition[TaskStage, TaskStructureReasonState](
            stage=TaskStage.REASONING,
            agent=TaskStructureReasoningAgent,
            states={
                TaskStructureReasonState.DONE: TaskStage.COMPLETED,
                TaskStructureReasonState.REQUEST_INFO: TaskStage.REQUEST_INFO_BELOW,
            },
        ),
        StageTransition[TaskStage, TaskStructureSummaryState](
            stage=TaskStage.SUMMARY,
            agent=TaskStructureSummaryAgent,
            states={
                TaskStructureSummaryState.DONE: {
                    TaskAction.DEFAULT: StageNext(stage=TaskStage.COMPLETED),
                    TaskAction.STOP: StageNext(stage=TaskStage.EXECUTING),
                },
                TaskStructureSummaryState.REQUEST_INFO: TaskStage.REQUEST_INFO_BELOW,
            },
        ),
        StageTransition[TaskStage, None](
            stage=TaskStage.REQUEST_INFO_BELOW, agent=RequestBelowUserSupplyAgent, next_stage=TaskStage.COMPLETED
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
