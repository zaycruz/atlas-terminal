import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from atlas.environment import (
    get_ai_model,
    get_ai_system_prompt,
    get_ollama_host,
    load_dotenv,
    resolve_environment,
)


class EnvironmentTests(unittest.TestCase):
    def test_resolve_environment_defaults_to_paper(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_environment(None), "paper")

    def test_resolve_environment_respects_env_var(self) -> None:
        with patch.dict(os.environ, {"ATLAS_ENV": "live"}, clear=True):
            self.assertEqual(resolve_environment(None), "live")

    def test_resolve_environment_rejects_invalid(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                resolve_environment("demo")

    def test_load_dotenv_sets_missing_values(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            env_path = Path(tmp) / ".env"
            env_path.write_text("FOO=bar\nALPACA_API_KEY_ID=abc\n")
            load_dotenv(env_path)
            self.assertEqual(os.environ["FOO"], "bar")
            self.assertEqual(os.environ["ALPACA_API_KEY_ID"], "abc")

    def test_load_dotenv_does_not_override_existing(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"FOO": "initial"}, clear=True
        ):
            env_path = Path(tmp) / ".env"
            env_path.write_text("FOO=bar\nBAR=baz\n")
            load_dotenv(env_path)
            self.assertEqual(os.environ["FOO"], "initial")
            self.assertEqual(os.environ["BAR"], "baz")

    def test_get_ai_model_defaults_and_override(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_ai_model(None), "llama3.2")
            self.assertEqual(get_ai_model("qwen"), "qwen")

    def test_get_ai_model_env(self) -> None:
        with patch.dict(os.environ, {"ATLAS_AI_MODEL": "phi"}, clear=True):
            self.assertEqual(get_ai_model(None), "phi")

    def test_get_ai_system_prompt(self) -> None:
        with patch.dict(os.environ, {"ATLAS_AI_SYSTEM_PROMPT": "hi"}, clear=True):
            self.assertEqual(get_ai_system_prompt(), "hi")

    def test_get_ollama_host(self) -> None:
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://localhost:1234/"}, clear=True):
            self.assertEqual(get_ollama_host(), "http://localhost:1234")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
