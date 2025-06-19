"""
Copyright (c) 2025 viewstar000

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import time
import shlex
import argparse
import ipynbname
import traceback

from IPython.display import Markdown
from IPython.core.magic import Magics, magics_class, cell_magic
from traitlets import Unicode, Int, Bool
from traitlets.config.configurable import Configurable
from .bot_contexts import NotebookContext, AgentCellContext
from .bot_agents import AgentFactory
from .bot_agents.base import AgentModelType
from .bot_flows import MasterPlannerFlow, TaskExecutorFlowV1, TaskExecutorFlowV2, TaskExecutorFlowV3
from .bot_outputs import _D, _I, _W, _E, _F, _M, _B, _O, reset_output, set_logging_level


@magics_class
class BotMagics(Magics, Configurable):
    """Jupyter cell magic for OpenAI bot integration"""

    # 配置项
    logging_level = Unicode("INFO", help="Debug level for logging").tag(config=True)
    default_api_url = Unicode(None, allow_none=True, help="Default API URL").tag(config=True)
    default_api_key = Unicode("API_KEY", help="Default API Key").tag(config=True)
    default_model_name = Unicode("", help="Default Model Name").tag(config=True)
    planner_api_url = Unicode(None, allow_none=True, help="Planner API URL").tag(config=True)
    planner_api_key = Unicode("API_KEY", help="Planner API Key").tag(config=True)
    planner_model_name = Unicode("", help="Planner Model Name").tag(config=True)
    coding_api_url = Unicode(None, allow_none=True, help="Coding API URL").tag(config=True)
    coding_api_key = Unicode("API_KEY", help="Coding API Key").tag(config=True)
    coding_model_name = Unicode("", help="Coding Model Name").tag(config=True)
    reasoning_api_url = Unicode(None, allow_none=True, help="Reasoning API URL").tag(config=True)
    reasoning_api_key = Unicode("API_KEY", help="Reasoning API Key").tag(config=True)
    reasoning_model_name = Unicode("", help="Reasoning Model Name").tag(config=True)
    display_message = Bool(False, help="Display chat message").tag(config=True)
    display_think = Bool(True, help="Display chatthink response").tag(config=True)
    display_response = Bool(False, help="Display chat full response").tag(config=True)
    notebook_path = Unicode(None, allow_none=True, help="Path to Notebook file").tag(config=True)
    default_task_flow = Unicode("v3", allow_none=True, help="Default task flow").tag(config=True)
    support_save_meta = Bool(False, help="Support save metadata to cell").tag(config=True)

    def parse_args(self, line):
        """解析命令行参数"""
        parser = argparse.ArgumentParser()
        parser.add_argument("-l", "--logging-level", type=str, default=self.logging_level, help="level for logging")
        parser.add_argument("-P", "--planning", action="store_true", default=False, help="Run in planning mode")
        parser.add_argument("-s", "--stage", type=str, default=None, help="Task stage")
        parser.add_argument("-f", "--flow", type=str, default=self.default_task_flow, help="Flow name")
        parser.add_argument("-m", "--max-tries", type=int, default=3, help="Max tries")
        parser.add_argument("-S", "--step-mode", action="store_true", default=False, help="Run in single step mode")
        parser.add_argument("-Y", "--auto-confirm", action="store_true", default=False, help="Run without confirm")
        options, _ = parser.parse_known_args(shlex.split(line.strip()))

        return options

    @cell_magic
    def bot(self, line, cell):
        """Jupyter cell magic: %%bot"""
        try:
            AgentCellContext.SUPPORT_SAVE_META = self.support_save_meta
            reset_output(stage="Logging", logging_level=self.logging_level)
            _I("Cell magic %%bot executing ...")
            _D(f"Cell magic called with line: {line}")
            _D(f"Cell magic called with cell: {repr(cell)[:50]} ...")
            if not cell.strip():
                _O(
                    Markdown(
                        "The cell is **empty**, we can't do anything.\n\n"
                        "We will fill it with some random characters, please **RERUN** the cell again."
                    )
                )
                if self.shell is not None:
                    self.shell.set_next_input(
                        "%%bot {}\n\n# {}".format(line.strip(), time.strftime("%Y-%m-%d %H:%M:%S")), replace=True
                    )
                return
            options = self.parse_args(line)
            _D(f"Cell magic called with options: {options}")
            set_logging_level(options.logging_level)
            self.notebook_path = self.notebook_path or ipynbname.path()
            _D(f"Cell magic called with notebook path: {self.notebook_path}")
            nb_context = NotebookContext(line, cell, notebook_path=self.notebook_path)
            agent_factory = AgentFactory(
                nb_context,
                display_think=self.display_think,
                display_message=self.display_message,
                display_response=self.display_response,
            )
            agent_factory.config_model(
                AgentModelType.DEFAULT, self.default_api_url, self.default_api_key, self.default_model_name
            )
            agent_factory.config_model(
                AgentModelType.PLANNER, self.planner_api_url, self.planner_api_key, self.planner_model_name
            )
            agent_factory.config_model(
                AgentModelType.CODING, self.coding_api_url, self.coding_api_key, self.coding_model_name
            )
            agent_factory.config_model(
                AgentModelType.REASONING, self.reasoning_api_url, self.reasoning_api_key, self.reasoning_model_name
            )
            if options.planning:
                flow = MasterPlannerFlow(nb_context, agent_factory)
            else:
                if options.flow == "v1":
                    flow = TaskExecutorFlowV1(nb_context, agent_factory)
                elif options.flow == "v2":
                    flow = TaskExecutorFlowV2(nb_context, agent_factory)
                elif options.flow == "v3":
                    flow = TaskExecutorFlowV3(nb_context, agent_factory)
                else:
                    raise ValueError(f"Unknown flow: {options.flow}")
            flow(options.stage, options.max_tries, not options.step_mode, not options.auto_confirm)
        except Exception as e:
            traceback.print_exc()


def load_ipython_extension(ipython):
    """Load the bot magic extension."""
    ipython.register_magics(BotMagics)
