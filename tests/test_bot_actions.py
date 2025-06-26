import pytest
import time
import threading
import queue
from jupyter_agent import bot_actions
from pydantic import ValidationError
from bottle import default_app


def test_actionbase_and_subclasses():
    # Test ActionBase
    ab = bot_actions.ActionBase(timestamp=time.time(), action="test_action")
    assert ab.action == "test_action"
    assert isinstance(ab.timestamp, float)

    # Test SetNextCellParams and ActionSetNextCell
    params = bot_actions.SetNextCellParams(index=2, type="markdown", source="Hello", tags=["tag1"])
    action = bot_actions.ActionSetNextCell(timestamp=time.time(), params=params)
    assert action.action == "set_next_cell"
    assert action.params.index == 2
    assert action.params.type == "markdown"

    # Test ConfirmChoiceItem and ActionRequestUserConfirm
    choice = bot_actions.ConfirmChoiceItem(label="Yes", value="yes")
    params = bot_actions.RequestUserConfirmParams(prompt="Continue?", choices=[choice], default="yes")
    action = bot_actions.ActionRequestUserConfirm(timestamp=time.time(), params=params)
    assert action.action == "request_user_confirm"
    assert action.params.choices[0].value == "yes"

    # Test UserSupplyInfoReply and ActionReceiveUserSupplyInfo
    reply = bot_actions.UserSupplyInfoReply(prompt="Please confirm", reply="OK")
    params = bot_actions.ReceiveUserSupplyInfoParams(replies=[reply])
    action = bot_actions.ActionReceiveUserSupplyInfo(timestamp=time.time(), params=params)
    assert action.action == "receive_user_supply_info"
    assert action.params.replies[0].reply == "OK"


def test_get_action_class():
    klass = bot_actions.get_action_class("set_next_cell")
    assert klass is bot_actions.ActionSetNextCell
    klass2 = bot_actions.get_action_class("request_user_confirm")
    assert klass2 is bot_actions.ActionRequestUserConfirm
    with pytest.raises(ValueError):
        bot_actions.get_action_class("nonexistent_action")


def test_action_dispatcher_queue_and_reply(monkeypatch):
    dispatcher = bot_actions.ActionDispatcher()
    action = bot_actions.ActionSetNextCell(timestamp=time.time())
    dispatcher.send_action(action)
    # Fetch from queue
    queued = dispatcher.action_queue.get(timeout=1)
    assert queued["action"] == "set_next_cell"

    # Test reply mechanism
    reply_action = bot_actions.ActionRequestUserConfirm(timestamp=time.time())
    reply_uuid = "test-uuid"
    reply = bot_actions.ActionReply(
        reply_timestamp=time.time(),
        uuid=reply_uuid,
        reply=reply_action,
    )
    dispatcher.action_replies[reply_uuid] = reply

    # Simulate get_action_reply
    class DummyReplyAction(bot_actions.ReplyActionBase):
        uuid: str = reply_uuid

    dummy = DummyReplyAction(timestamp=time.time(), action="dummy", uuid=reply_uuid)
    result = dispatcher.get_action_reply(dummy, wait=False)
    assert result == reply_action

    dispatcher.close()


def test_action_dispatcher_context_manager():
    with bot_actions.ActionDispatcher() as dispatcher:
        assert dispatcher.is_alive()
    assert not dispatcher.is_alive()


def test_get_and_close_action_dispatcher():
    d1 = bot_actions.get_action_dispatcher()
    assert isinstance(d1, bot_actions.ActionDispatcher)
    bot_actions.close_action_dispatcher()
    assert bot_actions._default_action_dispatcher is None or not bot_actions._default_action_dispatcher.is_alive()


def test_request_user_reply(monkeypatch):
    prompts = [
        bot_actions.RequestUserSupplyInfo(prompt="What is your name?", example="Alice"),
        bot_actions.RequestUserSupplyInfo(prompt="What is your age?", example="30"),
    ]
    monkeypatch.setattr("builtins.input", lambda prompt: "test")
    replies = bot_actions.request_user_reply(prompts)
    assert len(replies) == 2
    assert replies[0].reply == "test"
