"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import time

from .base import BaseEvaluator
from ..bot_outputs import StageEvalutionRecord


class DummyTaskEvaluator(BaseEvaluator):

    def __call__(self, **kwargs):
        """
        Dummy evaluator that does nothing and returns a dummy response.
        """
        return StageEvalutionRecord(timestamp=time.time(), evaluator="dummy")
