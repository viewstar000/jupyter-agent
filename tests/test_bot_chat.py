import pytest
from unittest.mock import patch, MagicMock
from jupyter_agent import bot_chat


class DummyLogger:
    def __call__(self, *args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def patch_loggers(monkeypatch):
    # Patch all logger functions to dummy
    monkeypatch.setattr(bot_chat, "_D", DummyLogger())
    monkeypatch.setattr(bot_chat, "_I", DummyLogger())
    monkeypatch.setattr(bot_chat, "_W", DummyLogger())
    monkeypatch.setattr(bot_chat, "_E", DummyLogger())
    monkeypatch.setattr(bot_chat, "_F", DummyLogger())
    monkeypatch.setattr(bot_chat, "_B", DummyLogger())
    monkeypatch.setattr(bot_chat, "_M", DummyLogger())


def test_chatmessages_add_and_get():
    cm = bot_chat.ChatMessages(contexts={"name": "Alice"})
    cm.add("Hello, {{ name }}!", role="user")
    messages = cm.get()
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["text"] == "Hello, Alice!"


def test_chatmessages_add_multiple_roles():
    cm = bot_chat.ChatMessages()
    cm.add("Hi", role="user")
    cm.add("Hello", role="assistant")
    cm.add("How are you?", role="user")
    messages = cm.get()
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"


def test_chatmessages_clear():
    cm = bot_chat.ChatMessages()
    cm.add("Hi", role="user")
    cm.clear()
    assert cm.get() == []


def test_chatmessages_add_unsupported_content_type():
    cm = bot_chat.ChatMessages()
    with pytest.raises(NotImplementedError):
        cm.add("data", content_type="image")


def test_botchat_parse_reply_text():
    bc = bot_chat.BotChat("http://test", "key", "gpt-4")
    reply = "Hello, world!"
    result = list(bc.parse_reply(reply))
    assert result[0]["type"] == "text"
    assert "Hello, world!" in result[0]["content"]


def test_botchat_parse_reply_think_block():
    bc = bot_chat.BotChat("http://test", "key", "gpt-4")
    reply = "<think>This is a thought.</think>"
    result = list(bc.parse_reply(reply, ret_think_block=True))
    assert any(r["type"] == "think" for r in result)
    assert any("This is a thought." in r["content"] for r in result if r["type"] == "think")


def test_botchat_parse_reply_code_block():
    bc = bot_chat.BotChat("http://test", "key", "gpt-4")
    reply = "```python\nprint('hi')\n```"
    result = list(bc.parse_reply(reply))
    assert any(r["type"] == "code" and r["lang"] == "python" for r in result)
    assert any("print('hi')" in r["content"] for r in result if r["type"] == "code")


def test_botchat_parse_reply_fence_block():
    bc = bot_chat.BotChat("http://test", "key", "gpt-4")
    reply = "```\njust text\n```"
    result = list(bc.parse_reply(reply))
    assert any(r["type"] == "fence" for r in result)
    assert any("just text" in r["content"] for r in result if r["type"] == "fence")


def test_botchat_create_messages():
    bc = bot_chat.BotChat("http://test", "key", "gpt-4")
    cm = bc.create_messages(contexts={"foo": "bar"})
    assert isinstance(cm, bot_chat.ChatMessages)
    cm.add("Hello, {{ foo }}!")
    assert cm.get()[0]["content"][0]["text"] == "Hello, bar!"


@patch("openai.OpenAI")
def test_botchat_chat_success(mock_openai):
    # Setup mock OpenAI response
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Hello, world!"
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client

    bc = bot_chat.BotChat("http://test", "key", "gpt-4")
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    result = bc.chat(messages)
    assert any(r["type"] == "text" for r in result)
    assert any("Hello, world!" in r["content"] for r in result)


@patch("openai.OpenAI")
def test_botchat_chat_no_response(mock_openai):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = []
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client

    bc = bot_chat.BotChat("http://test", "key", "gpt-4")
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    result = bc.chat(messages)
    assert result == []
