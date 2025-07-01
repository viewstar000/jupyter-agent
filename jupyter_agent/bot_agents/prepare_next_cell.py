"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import json
import datetime

from IPython.display import Markdown
from .base import BaseAgent
from ..bot_outputs import _D, _I, _W, _E, _F, _M, _B, _C, _O, ReplyType, markdown_block
from ..bot_actions import get_action_dispatcher, ActionSetCellContent, SetCellContentParams
from ..utils import get_env_capbilities


class PrepareNextCell(BaseAgent):

    def __call__(self):
        """执行代码逻辑"""
        if get_env_capbilities().set_cell_content:
            _I("set next cell content to generate the next task")
            get_action_dispatcher().send_action(
                ActionSetCellContent(
                    source=self.__class__.__name__,
                    params=SetCellContentParams(
                        index=1,
                        type="code",
                        source=(
                            "%%bot\n\n"
                            "# Execute this cell to generate the next task\n"
                            "# {}\n"
                            "# Special Note: Ensure the notebook is SAVED before executing this cell!\n"
                        ).format(datetime.datetime.now().isoformat()),
                    ),
                ),
                need_reply=False,
            )
        else:
            _M("Copy the following code to the next cell to generate the next task ...")
            _M(
                (
                    "```python\n"
                    "%%bot\n\n"
                    "# Execute this cell to generate the next task\n"
                    "# {}\n"
                    "# Special Note: Ensure the notebook is SAVED before executing this cell!\n"
                    "```"
                ).format(datetime.datetime.now().isoformat())
            )
        return False, None
