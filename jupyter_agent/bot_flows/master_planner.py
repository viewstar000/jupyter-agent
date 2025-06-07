"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from .base import BaseTaskFlow, StageTransition, TASK_STAGE_START, TASK_STAGE_COMPLETED


class MasterPlannerFlow(BaseTaskFlow):

    STAGE_TRANSITIONS = [
        StageTransition(stage=TASK_STAGE_START, agent="MasterPlannerAgent", next_stage=TASK_STAGE_START)
    ]
    STOP_STAGES = [TASK_STAGE_START]
