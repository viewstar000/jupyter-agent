"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import time
import shlex
import argparse
import ipynbname

from IPython.display import Markdown
from IPython.core.magic import Magics, magics_class, cell_magic
from traitlets import Unicode, Int
from traitlets.config.configurable import Configurable
from .bot_contexts import NotebookContext, TaskCellContext
from .bot_agents import AgentFactory
from .bot_flows import MasterPlannerFlow, TaskExecutorFlow
from .utils import DebugMixin


@magics_class
class BotMagics(Magics, Configurable, DebugMixin):
    """Jupyter cell magic for OpenAI bot integration"""

    # 配置项
    planner_api_url = Unicode(None, allow_none=True, help="Planner API URL").tag(config=True)
    planner_api_key = Unicode("API_KEY", help="Planner API Key").tag(config=True)
    planner_model = Unicode("", help="Planner Model Name").tag(config=True)
    coding_api_url = Unicode(None, allow_none=True, help="Coding API URL").tag(config=True)
    coding_api_key = Unicode("API_KEY", help="Coding API Key").tag(config=True)
    coding_model = Unicode("", help="Coding Model Name").tag(config=True)
    reasoning_api_url = Unicode(None, allow_none=True, help="Reasoning API URL").tag(config=True)
    reasoning_api_key = Unicode("API_KEY", help="Reasoning API Key").tag(config=True)
    reasoning_model = Unicode("", help="Reasoning Model Name").tag(config=True)
    notebook_path = Unicode(None, allow_none=True, help="Path to Notebook file").tag(config=True)
    debug_level = Int(0, help="Debug level for logging").tag(config=True)

    def parse_args(self, line):
        """解析命令行参数"""
        parser = argparse.ArgumentParser()
        parser.add_argument("--debug-level", type=int, default=int(self.debug_level), help="Debug level for logging")
        parser.add_argument("-P", "--planning", action="store_true", default=False, help="Run in planning mode")
        parser.add_argument("-s", "--stage", type=str, default=None, help="Task stage")
        parser.add_argument("-m", "--max-tries", type=int, default=3, help="Max tries")
        parser.add_argument("-S", "--step-mode", action="store_true", default=False, help="Run in single step mode")
        parser.add_argument("-Y", "--auto-confirm", action="store_true", default=False, help="Run without confirm")
        options, _ = parser.parse_known_args(shlex.split(line.strip()))
        self.debug_level = options.debug_level
        return options

    @cell_magic
    def bot(self, line, cell):
        """Jupyter cell magic: %%bot"""
        self.debug("Cell magic called with line:", line)
        self.debug("Cell magic called with cell:", repr(cell)[:50], "...")
        if not cell.strip():
            self._D(
                Markdown(
                    "The cell is **empty**, we can't do anything.\n\n"
                    "We will fill it with some random characters, please **RERUN** the cell again."
                )
            )
            self.shell.set_next_input(
                "%%bot {}\n\n# {}".format(line.strip(), time.strftime("%Y-%m-%d %H:%M:%S")), replace=True
            )
            return
        options = self.parse_args(line)
        self.debug("Cell magic called with options:", options)
        self.notebook_path = self.notebook_path or ipynbname.path()
        self.debug("Cell magic called with notebook path:", self.notebook_path)
        nb_context = NotebookContext(line, cell, notebook_path=self.notebook_path, debug_level=self.debug_level)
        cell_context = TaskCellContext(line, cell, notebook_path=self.notebook_path, debug_level=self.debug_level)
        agent_factory = AgentFactory(
            nb_context,
            cell_context,
            self.planner_api_url,
            self.planner_api_key,
            self.planner_model,
            self.coding_api_url,
            self.coding_api_key,
            self.coding_model,
            self.reasoning_api_url,
            self.reasoning_api_key,
            self.reasoning_model,
            debug_level=self.debug_level,
        )
        if options.planning:
            flow = MasterPlannerFlow(nb_context, cell_context, agent_factory, debug_level=self.debug_level)
        else:
            flow = TaskExecutorFlow(nb_context, cell_context, agent_factory, debug_level=self.debug_level)
        flow(options.stage, options.max_tries, not options.step_mode, not options.auto_confirm)


def load_ipython_extension(ipython):
    """Load the bot magic extension."""
    ipython.register_magics(BotMagics)
