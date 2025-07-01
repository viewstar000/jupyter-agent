"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from .base import BaseTaskFlow, StageNode, TASK_STAGE_START, TASK_STAGE_COMPLETED
from ..bot_evaluators.flow_global_planning import FlowGlobalPlanningEvaluator
from ..bot_agents.master_planner import MasterPlannerAgent
from ..bot_agents.output_task_result import OutputTaskResult
from ..bot_agents.prepare_next_cell import PrepareNextCell
from ..bot_evaluators.dummy_task import DummyTaskEvaluator


class MasterPlannerFlow(BaseTaskFlow):

    STAGE_NODES = [
        StageNode(
            stage=TASK_STAGE_START,
            agents=MasterPlannerAgent,
            evaluators=DummyTaskEvaluator,
            next_stage=TASK_STAGE_COMPLETED,
        ),
        StageNode(stage=TASK_STAGE_COMPLETED, agents=OutputTaskResult, next_stage=TASK_STAGE_COMPLETED),
    ]
    STOP_STAGES = [TASK_STAGE_COMPLETED]
    FLOW_EVALUATOR = FlowGlobalPlanningEvaluator
