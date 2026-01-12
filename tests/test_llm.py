"""Tests for the LLM layer."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from atlas.llm.base import Message, ToolCall, StreamChunk


class TestMessage:
    """Tests for Message dataclass."""

    def test_basic_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls is None
        assert msg.tool_call_id is None
        assert msg.name is None

    def test_assistant_message(self):
        msg = Message(role="assistant", content="Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_message_with_tool_calls(self):
        tool_call = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"location": "NYC"},
        )
        msg = Message(
            role="assistant",
            content="",
            tool_calls=[tool_call],
        )
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "get_weather"

    def test_tool_response_message(self):
        msg = Message(
            role="tool",
            content='{"temp": 72}',
            tool_call_id="call_123",
            name="get_weather",
        )
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_123"
        assert msg.name == "get_weather"

    def test_to_dict_basic(self):
        msg = Message(role="user", content="Hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Hello"}

    def test_to_dict_with_tool_calls(self):
        tool_call = ToolCall(
            id="call_1",
            name="test_func",
            arguments={"key": "value"},
        )
        msg = Message(
            role="assistant",
            content="",
            tool_calls=[tool_call],
        )
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == ""
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["id"] == "call_1"
        assert d["tool_calls"][0]["type"] == "function"
        assert d["tool_calls"][0]["function"]["name"] == "test_func"

    def test_to_dict_with_tool_call_id(self):
        msg = Message(
            role="tool",
            content="result",
            tool_call_id="call_abc",
        )
        d = msg.to_dict()
        assert d["tool_call_id"] == "call_abc"

    def test_to_dict_with_name(self):
        msg = Message(
            role="tool",
            content="result",
            name="my_function",
        )
        d = msg.to_dict()
        assert d["name"] == "my_function"


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_basic_tool_call(self):
        tc = ToolCall(
            id="call_xyz",
            name="search",
            arguments={"query": "python"},
        )
        assert tc.id == "call_xyz"
        assert tc.name == "search"
        assert tc.arguments == {"query": "python"}

    def test_empty_arguments(self):
        tc = ToolCall(id="call_1", name="no_args", arguments={})
        assert tc.arguments == {}


class TestStreamChunk:
    """Tests for StreamChunk dataclass."""

    def test_default_values(self):
        chunk = StreamChunk()
        assert chunk.delta == ""
        assert chunk.tool_calls is None
        assert chunk.done is False
        assert chunk.finish_reason is None

    def test_content_chunk(self):
        chunk = StreamChunk(delta="Hello")
        assert chunk.delta == "Hello"
        assert chunk.done is False

    def test_final_chunk(self):
        chunk = StreamChunk(done=True, finish_reason="stop")
        assert chunk.done is True
        assert chunk.finish_reason == "stop"

    def test_chunk_with_tool_calls(self):
        tool_call = ToolCall(
            id="call_1",
            name="func",
            arguments={"x": 1},
        )
        chunk = StreamChunk(
            done=True,
            finish_reason="tool_calls",
            tool_calls=[tool_call],
        )
        assert chunk.tool_calls is not None
        assert len(chunk.tool_calls) == 1


class TestZAIClientCreation:
    """Tests for ZAIClient creation."""

    def test_creation_without_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            from atlas.llm.client import ZAIClient
            with pytest.raises(ValueError, match="LLM API key required"):
                ZAIClient()

    def test_creation_with_zai_api_key(self):
        with patch.dict(os.environ, {"ZAI_API_KEY": "test-key"}, clear=True):
            from atlas.llm.client import ZAIClient
            # Need to mock AsyncOpenAI to avoid actual connection
            with patch("atlas.llm.client.AsyncOpenAI") as mock_openai:
                client = ZAIClient()
                assert client.name == "zai"
                mock_openai.assert_called_once()

    def test_creation_with_custom_params(self):
        with patch("atlas.llm.client.AsyncOpenAI") as mock_openai:
            from atlas.llm.client import ZAIClient
            client = ZAIClient(
                api_key="custom-key",
                base_url="https://custom.api/v1",
                model="custom-model",
            )
            assert client.model == "custom-model"
            mock_openai.assert_called_with(
                api_key="custom-key",
                base_url="https://custom.api/v1",
            )


class TestZAIClientMethods:
    """Tests for ZAIClient methods."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked ZAIClient."""
        with patch("atlas.llm.client.AsyncOpenAI") as mock_openai:
            from atlas.llm.client import ZAIClient
            client = ZAIClient(api_key="test-key")
            client._client = mock_openai.return_value
            return client

    @pytest.mark.asyncio
    async def test_complete_basic(self, mock_client):
        """Test basic completion."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.tool_calls = None

        mock_client._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        messages = [Message(role="user", content="Hi")]
        result = await mock_client.complete(messages)

        assert result.role == "assistant"
        assert result.content == "Hello!"
        assert result.tool_calls is None

    @pytest.mark.asyncio
    async def test_complete_with_tool_calls(self, mock_client):
        """Test completion with tool calls."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_abc"
        mock_tool_call.function.name = "test_func"
        mock_tool_call.function.arguments = '{"key": "value"}'

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_response.choices[0].message.tool_calls = [mock_tool_call]

        mock_client._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        messages = [Message(role="user", content="Call a function")]
        result = await mock_client.complete(messages)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_abc"
        assert result.tool_calls[0].name == "test_func"
        assert result.tool_calls[0].arguments == {"key": "value"}

    def test_parse_arguments_valid_json(self, mock_client):
        """Test parsing valid JSON arguments."""
        result = mock_client._parse_arguments('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_arguments_empty(self, mock_client):
        """Test parsing empty arguments."""
        result = mock_client._parse_arguments("")
        assert result == {}

    def test_parse_arguments_invalid_json(self, mock_client):
        """Test parsing invalid JSON returns raw."""
        result = mock_client._parse_arguments("not json")
        assert result == {"raw": "not json"}

    def test_to_api_message(self, mock_client):
        """Test converting Message to API format."""
        msg = Message(role="user", content="Test")
        result = mock_client._to_api_message(msg)
        assert result == {"role": "user", "content": "Test"}

    def test_to_api_tool(self, mock_client):
        """Test converting tool spec to API format."""
        tool = {
            "name": "my_func",
            "description": "Does something",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
            },
        }
        result = mock_client._to_api_tool(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "my_func"
        assert result["function"]["description"] == "Does something"


class TestGetLLMClient:
    """Tests for the global LLM client singleton."""

    def test_get_llm_client_without_key_raises(self):
        """Test that getting client without API key raises error."""
        # Reset the global client
        import atlas.llm.client as client_module
        client_module._llm_client = None

        with patch.dict(os.environ, {}, clear=True):
            from atlas.llm.client import get_llm_client
            with pytest.raises(ValueError):
                get_llm_client()
