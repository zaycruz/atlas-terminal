"""Tests for the environment configuration module."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from atlas.environment import (
    load_dotenv,
    get_llm_api_key,
    get_llm_base_url,
    get_llm_model,
    get_system_prompt,
    get_mcp_endpoint,
    get_searxng_endpoint,
    get_polygon_api_key,
    get_fred_api_key,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_SYSTEM_PROMPT,
)


class LoadDotenvTests(unittest.TestCase):
    """Tests for load_dotenv function."""

    def test_load_dotenv_sets_missing_values(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            env_path = Path(tmp) / ".env"
            env_path.write_text("FOO=bar\nZAI_API_KEY=test-key\n")
            load_dotenv(env_path)
            self.assertEqual(os.environ["FOO"], "bar")
            self.assertEqual(os.environ["ZAI_API_KEY"], "test-key")

    def test_load_dotenv_does_not_override_existing(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"FOO": "initial"}, clear=True
        ):
            env_path = Path(tmp) / ".env"
            env_path.write_text("FOO=bar\nBAR=baz\n")
            load_dotenv(env_path)
            self.assertEqual(os.environ["FOO"], "initial")
            self.assertEqual(os.environ["BAR"], "baz")

    def test_load_dotenv_handles_quoted_values(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            env_path = Path(tmp) / ".env"
            env_path.write_text('DOUBLE="double quoted"\nSINGLE=\'single quoted\'\n')
            load_dotenv(env_path)
            self.assertEqual(os.environ["DOUBLE"], "double quoted")
            self.assertEqual(os.environ["SINGLE"], "single quoted")

    def test_load_dotenv_skips_comments_and_empty(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            env_path = Path(tmp) / ".env"
            env_path.write_text("# comment\n\nFOO=bar\n  # another comment\n")
            load_dotenv(env_path)
            self.assertEqual(os.environ.get("FOO"), "bar")
            self.assertNotIn("#", os.environ)

    def test_load_dotenv_missing_file_is_noop(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            env_path = Path(tmp) / "nonexistent.env"
            load_dotenv(env_path)  # Should not raise


class LLMConfigTests(unittest.TestCase):
    """Tests for LLM configuration functions."""

    def test_get_llm_api_key_from_zai(self) -> None:
        with patch.dict(os.environ, {"ZAI_API_KEY": "zai-key"}, clear=True):
            self.assertEqual(get_llm_api_key(), "zai-key")

    def test_get_llm_api_key_from_atlas(self) -> None:
        with patch.dict(os.environ, {"ATLAS_LLM_API_KEY": "atlas-key"}, clear=True):
            self.assertEqual(get_llm_api_key(), "atlas-key")

    def test_get_llm_api_key_prefers_zai(self) -> None:
        with patch.dict(
            os.environ,
            {"ZAI_API_KEY": "zai-key", "ATLAS_LLM_API_KEY": "atlas-key"},
            clear=True,
        ):
            self.assertEqual(get_llm_api_key(), "zai-key")

    def test_get_llm_api_key_returns_none_if_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(get_llm_api_key())

    def test_get_llm_base_url_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_llm_base_url(), DEFAULT_LLM_BASE_URL)

    def test_get_llm_base_url_from_env(self) -> None:
        with patch.dict(
            os.environ, {"ATLAS_LLM_BASE_URL": "https://custom.api/"}, clear=True
        ):
            self.assertEqual(get_llm_base_url(), "https://custom.api")

    def test_get_llm_model_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_llm_model(), DEFAULT_LLM_MODEL)

    def test_get_llm_model_from_env(self) -> None:
        with patch.dict(os.environ, {"ATLAS_LLM_MODEL": "custom-model"}, clear=True):
            self.assertEqual(get_llm_model(), "custom-model")

    def test_get_llm_model_override(self) -> None:
        with patch.dict(os.environ, {"ATLAS_LLM_MODEL": "env-model"}, clear=True):
            self.assertEqual(get_llm_model("override-model"), "override-model")

    def test_get_system_prompt_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_system_prompt(), DEFAULT_SYSTEM_PROMPT)

    def test_get_system_prompt_from_env(self) -> None:
        with patch.dict(
            os.environ, {"ATLAS_SYSTEM_PROMPT": "Custom prompt"}, clear=True
        ):
            self.assertEqual(get_system_prompt(), "Custom prompt")


class MCPConfigTests(unittest.TestCase):
    """Tests for MCP configuration functions."""

    def test_get_mcp_endpoint_default(self) -> None:
        # Note: MCP_ENDPOINT is evaluated at module load time
        # so we test with fresh import in integration tests
        endpoint = get_mcp_endpoint()
        self.assertTrue(endpoint.startswith("http"))


class SearchConfigTests(unittest.TestCase):
    """Tests for search configuration functions."""

    def test_get_searxng_endpoint_default(self) -> None:
        endpoint = get_searxng_endpoint()
        self.assertTrue(endpoint.startswith("http"))
        self.assertFalse(endpoint.endswith("/"))


class DataFeedConfigTests(unittest.TestCase):
    """Tests for data feed configuration functions."""

    def test_get_polygon_api_key(self) -> None:
        with patch.dict(os.environ, {"POLYGON_API_KEY": "poly-key"}, clear=True):
            self.assertEqual(get_polygon_api_key(), "poly-key")

    def test_get_polygon_api_key_none(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(get_polygon_api_key())

    def test_get_fred_api_key(self) -> None:
        with patch.dict(os.environ, {"FRED_API_KEY": "fred-key"}, clear=True):
            self.assertEqual(get_fred_api_key(), "fred-key")


if __name__ == "__main__":
    unittest.main()
