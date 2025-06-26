import pytest
from unittest.mock import MagicMock, patch
from enum import Enum
from pydantic import BaseModel
from jupyter_agent.bot_agents.base import BaseAgent
from jupyter_agent.bot_flows import base


# Dummy Enum and Agent for testing
class DummyStage(str, Enum):
    START = "start"
    MIDDLE = "middle"
    END = "completed"


class DummyAgent(BaseAgent):
    EVALUATORS = {"success": "dummy_evaluator"}

    def __call__(self):
        return False, "success"


class DummyFailAgent(BaseAgent):
    def __call__(self):
        return True, "fail"


class DummyNotebookContext:
    def __init__(self):
        self.cur_task = MagicMock()
        self.cur_task.cell_idx = 0
        self.cur_task.agent_stage = DummyStage.START
        self.cur_task.update_cell = MagicMock()
        self.cells = []


class DummyEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self):
        class Result:
            timestamp = None
            evaluator = None
            cell_index = 0
            flow = "TestFlow"
            stage = "start"
            agent = "DummyAgent"
            execution_duration = 0.1
            is_success = True

        return Result()


class DummyStageNext(base.StageNext[DummyStage]):
    pass


class DummyStageTransition(base.StageTransition[DummyStage, str]):
    pass


@pytest.fixture
def notebook_context():
    return DummyNotebookContext()


@pytest.fixture
def agent_factory():
    def factory(agent_type):
        if agent_type == DummyAgent:
            return DummyAgent(DummyNotebookContext())
        elif agent_type == DummyFailAgent:
            return DummyFailAgent(DummyNotebookContext())
        return agent_type()

    return factory


@pytest.fixture
def evaluator_factory():
    def factory(evaluator_type):
        return DummyEvaluator()

    return factory


def make_simple_flow():
    class SimpleFlow(base.BaseTaskFlow):
        STAGE_TRANSITIONS = [
            DummyStageTransition(
                stage=DummyStage.START,
                agent=DummyAgent,
                states={
                    "success": DummyStageNext(stage=DummyStage.END, message="Done!"),
                    "fail": DummyStageNext(stage=DummyStage.START, message="Retry!"),
                },
            ),
            DummyStageTransition(stage=DummyStage.END, agent=DummyAgent, states={}),
        ]
        START_STAGE = DummyStage.START
        STOP_STAGES = [DummyStage.END]

    return SimpleFlow


def test_prepare_stage_transitions(notebook_context, agent_factory):
    flow_cls = make_simple_flow()
    flow = flow_cls(notebook_context, agent_factory)
    assert DummyStage.START in flow.stage_transitions
    assert DummyStage.END in flow.stage_transitions


def test_get_stage_agent_returns_agent(notebook_context, agent_factory):
    flow_cls = make_simple_flow()
    flow = flow_cls(notebook_context, agent_factory)
    agent = flow.get_stage_agent(DummyStage.START)
    assert isinstance(agent, DummyAgent)


def test_get_stage_agent_raises_for_invalid_stage(notebook_context, agent_factory):
    flow_cls = make_simple_flow()
    flow = flow_cls(notebook_context, agent_factory)
    with pytest.raises(ValueError):
        flow.get_stage_agent("nonexistent")


def test_get_next_stage_trans_returns_stage_next(notebook_context, agent_factory):
    flow_cls = make_simple_flow()
    flow = flow_cls(notebook_context, agent_factory)
    ns = flow._get_next_stage_trans(DummyStage.START, "success")
    assert isinstance(ns, base.StageNext)


def test_match_action():
    flow_cls = make_simple_flow()
    flow = flow_cls(DummyNotebookContext(), lambda x: DummyAgent(DummyNotebookContext()))
    assert flow.match_action("") == base.TaskAction.CONTINUE
    assert flow.match_action("c") == base.TaskAction.CONTINUE
    assert flow.match_action("continue") == base.TaskAction.CONTINUE
    assert flow.match_action("r") == base.TaskAction.RETRY
    assert flow.match_action("retry") == base.TaskAction.RETRY
    assert flow.match_action("k") == base.TaskAction.SKIP
    assert flow.match_action("skip") == base.TaskAction.SKIP
    assert flow.match_action("s") == base.TaskAction.STOP
    assert flow.match_action("stop") == base.TaskAction.STOP
    with pytest.raises(ValueError):
        flow.match_action("unknown")


def test_get_prompt_message_success(notebook_context, agent_factory):
    flow_cls = make_simple_flow()
    flow = flow_cls(notebook_context, agent_factory)
    msg = flow.get_prompt_message(DummyStage.START, "success", failed=False)
    assert "Continue to stage" in msg


def test_get_prompt_message_failed(notebook_context, agent_factory):
    flow_cls = make_simple_flow()
    flow = flow_cls(notebook_context, agent_factory)
    msg = flow.get_prompt_message(DummyStage.START, "fail", failed=True)
    assert "FAILED" in msg or "Retry!" in msg


@patch("jupyter_agent.bot_flows.base.set_stage")
@patch("jupyter_agent.bot_flows.base._M")
@patch("jupyter_agent.bot_flows.base.output_evaluation")
@patch("jupyter_agent.bot_flows.base.flush_output")
def test_call_success(
    mock_flush, mock_output_eval, mock_M, mock_set_stage, notebook_context, agent_factory, evaluator_factory
):
    flow_cls = make_simple_flow()
    flow = flow_cls(notebook_context, agent_factory, evaluator_factory)
    # Patch input to always continue
    with patch("builtins.input", return_value="c"):
        result = flow(stage=DummyStage.START, max_tries=2, stage_continue=True, stage_confirm=False)
    assert result == DummyStage.END


@patch("jupyter_agent.bot_flows.base.set_stage")
@patch("jupyter_agent.bot_flows.base._M")
@patch("jupyter_agent.bot_flows.base.output_evaluation")
@patch("jupyter_agent.bot_flows.base.flush_output")
def test_call_with_failure_and_retry(
    mock_flush, mock_output_eval, mock_M, mock_set_stage, notebook_context, agent_factory, evaluator_factory
):
    # Flow with a failing agent
    class FailFlow(base.BaseTaskFlow):
        STAGE_TRANSITIONS = [
            DummyStageTransition(
                stage=DummyStage.START,
                agent=DummyFailAgent,
                states={
                    "fail": DummyStageNext(stage=DummyStage.START, message="Retry!"),
                },
            ),
        ]
        START_STAGE = DummyStage.START
        STOP_STAGES = [DummyStage.END]

    flow = FailFlow(notebook_context, agent_factory, evaluator_factory)
    with patch("builtins.input", return_value="c"):
        result = flow(stage=DummyStage.START, max_tries=1, stage_continue=True, stage_confirm=False)
    assert result == DummyStage.START
