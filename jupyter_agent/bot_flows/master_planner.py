"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from .base import BaseTaskFlow, StageTransition, TASK_STAGE_START, TASK_STAGE_COMPLETED
from ..bot_evaluators.flow_global_planning import FlowGlobalPlanningEvaluator


class MasterPlannerFlow(BaseTaskFlow):

    STAGE_TRANSITIONS = [
        StageTransition(stage=TASK_STAGE_START, agent="MasterPlannerAgent", next_stage=TASK_STAGE_COMPLETED),
        StageTransition(stage=TASK_STAGE_COMPLETED, agent="OutputTaskResult", next_stage=TASK_STAGE_COMPLETED),
    ]
    STOP_STAGES = [TASK_STAGE_COMPLETED]
    FLOW_EVALUATOR = FlowGlobalPlanningEvaluator
