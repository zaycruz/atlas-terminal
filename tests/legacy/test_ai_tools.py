from dataclasses import dataclass

from atlas.ai.tools import ToolResult


@dataclass
class Dummy:
    value: int


def test_tool_result_serializes_dataclass() -> None:
    result = ToolResult(True, "dummy", Dummy(5), "ok")
    payload = result.to_model_dict()
    assert payload["data"] == {"value": 5}
    assert payload["success"] is True
    assert payload["tool"] == "dummy"
