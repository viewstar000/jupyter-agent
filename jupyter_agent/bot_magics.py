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
from .bot_contexts import NotebookContext
from .bot_agents.base import AgentModelType, AgentFactory
from .bot_agents.request_user_supply import RequestUserSupplyAgent
from .bot_evaluators.base import EvaluatorFactory
from .bot_flows import MasterPlannerFlow, TaskExecutorFlowV3
from .bot_outputs import _D, _I, _W, _E, _F, _M, _B, _O, reset_output, set_logging_level, flush_output
from .bot_actions import close_action_dispatcher
from .utils import get_env_capbilities


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
    evaluating_api_url = Unicode(None, allow_none=True, help="Evaluating API URL").tag(config=True)
    evaluating_api_key = Unicode("API_KEY", help="Evaluating API Key").tag(config=True)
    evaluating_model_name = Unicode("", help="Evaluating Model Name").tag(config=True)
    reasoning_api_url = Unicode(None, allow_none=True, help="Reasoning API URL").tag(config=True)
    reasoning_api_key = Unicode("API_KEY", help="Reasoning API Key").tag(config=True)
    reasoning_model_name = Unicode("", help="Reasoning Model Name").tag(config=True)
    display_message = Bool(False, help="Display chat message").tag(config=True)
    display_think = Bool(True, help="Display chatthink response").tag(config=True)
    display_response = Bool(False, help="Display chat full response").tag(config=True)
    support_save_meta = Bool(False, help="Support save metadata to cell").tag(config=True)
    support_user_confirm = Bool(False, help="Support user confirm").tag(config=True)
    support_user_supply_info = Bool(False, help="Support user supply info").tag(config=True)
    support_set_cell_content = Bool(False, help="Support set cell content").tag(config=True)
    enable_evaluating = Bool(False, help="Enable evaluating task").tag(config=True)
    enable_supply_mocking = Bool(False, help="Enable supply mocking").tag(config=True)
    notebook_path = Unicode(None, allow_none=True, help="Path to Notebook file").tag(config=True)
    default_task_flow = Unicode("v3", allow_none=True, help="Default task flow").tag(config=True)
    default_max_tries = Int(2, help="Default max tries for task execution").tag(config=True)
    default_step_mode = Bool(False, help="Default step mode for task execution").tag(config=True)
    default_auto_confirm = Bool(True, help="Default auto confirm for task execution").tag(config=True)

    def parse_args(self, line):
        """解析命令行参数"""
        parser = argparse.ArgumentParser()
        parser.add_argument("-l", "--logging-level", type=str, default=self.logging_level, help="level for logging")
        parser.add_argument("-P", "--planning", action="store_true", default=False, help="Run in planning mode")
        parser.add_argument("-s", "--stage", type=str, default=None, help="Task stage")
        parser.add_argument("-f", "--flow", type=str, default=self.default_task_flow, help="Flow name")
        parser.add_argument("-m", "--max-tries", type=int, default=self.default_max_tries, help="Max tries")
        parser.add_argument(
            "-t",
            "--step-mode",
            action="store_true",
            dest="step_mode",
            default=self.default_step_mode,
            help="Run in single step mode",
        )
        parser.add_argument(
            "-T",
            "--not-step-mode",
            action="store_false",
            dest="step_mode",
            default=self.default_step_mode,
            help="Run in multi step mode",
        )
        parser.add_argument(
            "-y",
            "--auto-confirm",
            action="store_true",
            dest="auto_confirm",
            default=self.default_auto_confirm,
            help="Run without confirm",
        )
        parser.add_argument(
            "-Y",
            "--not-auto-confirm",
            action="store_false",
            dest="auto_confirm",
            default=self.default_auto_confirm,
            help="Run with confirm",
        )
        options, _ = parser.parse_known_args(shlex.split(line.strip()))
        return options

    @cell_magic
    def bot(self, line, cell):
        """Jupyter cell magic: %%bot"""
        try:
            reset_output(stage="Logging", logging_level=self.logging_level)
            _I("Cell magic %%bot executing ...")
            _D(f"Cell magic called with line: {line}")
            _D(f"Cell magic called with cell: {repr(cell)[:50]} ...")
            if not self.ensure_notebook_path():
                _O(
                    Markdown(
                        "The notebook path is **empty**, we can't do anything.\n\n"
                        "Please set the notebook path in the configuration, and **RERUN** the cell again.\n\n"
                        'For example: `%config BotMagics.notebook_path = globals()["__vsc_ipynb_file__"]`'
                    )
                )
                return
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
            get_env_capbilities().save_metadata = self.support_save_meta
            get_env_capbilities().user_confirm = self.support_user_confirm
            get_env_capbilities().user_supply_info = self.support_user_supply_info
            get_env_capbilities().set_cell_content = self.support_set_cell_content
            RequestUserSupplyAgent.MOCK_USER_SUPPLY = self.enable_supply_mocking
            options = self.parse_args(line)
            set_logging_level(options.logging_level)
            _D(f"Cell magic called with options: {options}")
            nb_context = NotebookContext(line, cell, notebook_path=self.notebook_path)
            agent_factory = self.get_agent_factory(nb_context)
            evaluator_factory = self.get_evaluator_factory(nb_context)
            if options.planning:
                flow = MasterPlannerFlow(nb_context, agent_factory, evaluator_factory)
            elif options.flow == "v3":
                flow = TaskExecutorFlowV3(nb_context, agent_factory, evaluator_factory)
            else:
                raise ValueError(f"Unknown flow: {options.flow}")
            flow(
                options.stage,
                options.max_tries,
                not options.step_mode,
                not options.auto_confirm,
            )
        except Exception as e:
            traceback.print_exc()
        finally:
            close_action_dispatcher()
            flush_output()

    def ensure_notebook_path(self):
        if self.notebook_path:
            return self.notebook_path
        result = self.shell and self.shell.run_cell(
            "globals().get('__vsc_ipynb_file__') or globals().get('__evaluation_ipynb_file__')"
        )
        if result and result.success and result.result:
            self.notebook_path = result.result
            return self.notebook_path
        try:
            self.notebook_path = str(ipynbname.path())
            return self.notebook_path
        except Exception as e:
            _F(f"Failed to get notebook path: {e}")
            return None

    def get_agent_factory(self, nb_context):
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
            AgentModelType.EVALUATING, self.evaluating_api_url, self.evaluating_api_key, self.evaluating_model_name
        )
        agent_factory.config_model(
            AgentModelType.REASONING, self.reasoning_api_url, self.reasoning_api_key, self.reasoning_model_name
        )
        return agent_factory

    def get_evaluator_factory(self, nb_context):
        if self.enable_evaluating:
            evaluator_factory = EvaluatorFactory(
                nb_context,
                display_think=self.display_think,
                display_message=self.display_message,
                display_response=self.display_response,
            )
            evaluator_factory.config_model(
                AgentModelType.DEFAULT, self.default_api_url, self.default_api_key, self.default_model_name
            )
            evaluator_factory.config_model(
                AgentModelType.PLANNER, self.planner_api_url, self.planner_api_key, self.planner_model_name
            )
            evaluator_factory.config_model(
                AgentModelType.CODING, self.coding_api_url, self.coding_api_key, self.coding_model_name
            )
            evaluator_factory.config_model(
                AgentModelType.EVALUATING, self.evaluating_api_url, self.evaluating_api_key, self.evaluating_model_name
            )
            evaluator_factory.config_model(
                AgentModelType.REASONING, self.reasoning_api_url, self.reasoning_api_key, self.reasoning_model_name
            )
        else:
            evaluator_factory = None
        return evaluator_factory


def load_ipython_extension(ipython):
    """Load the bot magic extension."""
    ipython.register_magics(BotMagics)
