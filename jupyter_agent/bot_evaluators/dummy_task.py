"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import time

from .base import BaseEvaluator
from ..bot_evaluation import StageEvaluationRecord


class DummyTaskEvaluator(BaseEvaluator):

    def __call__(self, **kwargs):
        """
        Dummy evaluator that does nothing and returns a dummy response.
        """
        return StageEvaluationRecord(timestamp=time.time(), evaluator="dummy")
