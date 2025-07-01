"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from enum import Enum
from .base import (
    BaseTaskFlow,
    StageNode,
    StageNext,
    TaskAction,
    TASK_STAGE_COMPLETED,
    TASK_STAGE_GLOBAL_FINISHED,
)
from ..bot_agents.task_planner_v3 import TaskPlannerAgentV3, TaskPlannerState
from ..bot_agents.task_coder import TaskCodingAgent
from ..bot_agents.task_debuger import CodeDebugerAgent
from ..bot_agents.task_code_executor import CodeExecutor
from ..bot_agents.task_structrue_summarier import TaskStructureSummaryAgent, TaskStructureSummaryState
from ..bot_agents.task_structrue_reasoner import TaskStructureReasoningAgent, TaskStructureReasonState
from ..bot_agents.output_task_result import OutputTaskResult
from ..bot_agents.request_user_supply import RequestAboveUserSupplyAgent, RequestBelowUserSupplyAgent
from ..bot_agents.prepare_next_cell import PrepareNextCell


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
    PREPARE_NEXT = "prepare_next"
    OUTPUT_RESULT = "output_result"
    COMPLETED = TASK_STAGE_COMPLETED
    GLOBAL_FINISHED = TASK_STAGE_GLOBAL_FINISHED


class TaskExecutorFlowV3(BaseTaskFlow):

    START_STAGE = TaskStage.PLANNING
    STOP_STAGES = [TaskStage.COMPLETED, TaskStage.PLANNING_PAUSED, TaskStage.GLOBAL_FINISHED]
    STAGE_NODES = [
        StageNode[TaskStage, TaskPlannerState](
            stage=TaskStage.PLANNING,
            agents=TaskPlannerAgentV3,
            states={
                TaskPlannerState.CODING_PLANNED: TaskStage.CODING,
                TaskPlannerState.REASONING_PLANNED: TaskStage.REASONING,
                TaskPlannerState.REQUEST_INFO: TaskStage.REQUEST_INFO_ABOVE,
                TaskPlannerState.GLOBAL_FINISHED: TaskStage.GLOBAL_FINISHED,
            },
        ),
        StageNode[TaskStage, None](
            stage=TaskStage.REQUEST_INFO_ABOVE,
            agents=RequestAboveUserSupplyAgent,
            next_stage=TaskStage.PLANNING_PAUSED,
        ),
        StageNode[TaskStage, TaskPlannerState](
            stage=TaskStage.PLANNING_PAUSED,
            agents=TaskPlannerAgentV3,
            states={
                TaskPlannerState.CODING_PLANNED: TaskStage.CODING,
                TaskPlannerState.REASONING_PLANNED: TaskStage.REASONING,
                TaskPlannerState.REQUEST_INFO: TaskStage.REQUEST_INFO_ABOVE,
                TaskPlannerState.GLOBAL_FINISHED: TaskStage.COMPLETED,
            },
        ),
        StageNode[TaskStage, None](stage=TaskStage.CODING, agents=TaskCodingAgent, next_stage=TaskStage.EXECUTING),
        StageNode[TaskStage, bool](
            stage=TaskStage.EXECUTING,
            agents=CodeExecutor,
            states={True: TaskStage.SUMMARY, False: TaskStage.DEBUGGING},
        ),
        StageNode[TaskStage, None](stage=TaskStage.DEBUGGING, agents=CodeDebugerAgent, next_stage=TaskStage.EXECUTING),
        StageNode[TaskStage, TaskStructureReasonState](
            stage=TaskStage.REASONING,
            agents=TaskStructureReasoningAgent,
            states={
                TaskStructureReasonState.DONE: TaskStage.COMPLETED,
                TaskStructureReasonState.REQUEST_INFO: TaskStage.REQUEST_INFO_BELOW,
            },
        ),
        StageNode[TaskStage, TaskStructureSummaryState](
            stage=TaskStage.SUMMARY,
            agents=TaskStructureSummaryAgent,
            states={
                TaskStructureSummaryState.DONE: {
                    TaskAction.DEFAULT: StageNext(stage=TaskStage.PREPARE_NEXT),
                    TaskAction.STOP: StageNext(stage=TaskStage.EXECUTING),
                },
                TaskStructureSummaryState.REQUEST_INFO: TaskStage.REQUEST_INFO_BELOW,
            },
        ),
        StageNode[TaskStage, None](
            stage=TaskStage.PREPARE_NEXT, agents=PrepareNextCell, next_stage=TaskStage.COMPLETED
        ),
        StageNode[TaskStage, None](
            stage=TaskStage.REQUEST_INFO_BELOW,
            agents=[PrepareNextCell, RequestBelowUserSupplyAgent],
            next_stage=TaskStage.COMPLETED,
        ),
        StageNode[TaskStage, bool](
            stage=TaskStage.COMPLETED,
            agents=CodeExecutor,
            states={True: TaskStage.OUTPUT_RESULT, False: TaskStage.DEBUGGING},
        ),
        StageNode[TaskStage, None](
            stage=TaskStage.OUTPUT_RESULT, agents=OutputTaskResult, next_stage=TaskStage.COMPLETED
        ),
        StageNode[TaskStage, None](
            stage=TaskStage.GLOBAL_FINISHED, agents=OutputTaskResult, next_stage=TaskStage.GLOBAL_FINISHED
        ),
    ]
