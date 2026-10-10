import json
from types import SimpleNamespace as NS

from harness_lab.controller_v1_2 import run_harness
from harness_lab.tools import calculate


CALCULATE_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform a basic arithmetic operation.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                },
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
            "additionalProperties": False,
        },
    },
}


def tool_call(name, arguments, call_id="call_1"):
    return NS(
        id=call_id,
        function=NS(name=name, arguments=arguments),
        type="function",
    )


def response(content=None, tool_calls=None):
    return NS(
        choices=[
            NS(
                message=NS(
                    content=content,
                    tool_calls=tool_calls or [],
                )
            )
        ]
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.chat = NS(
            completions=NS(create=self.create)
        )

    def create(self, **kwargs):
        self.calls += 1
        if not self.responses:
            raise AssertionError("Unexpected LLM call")
        return self.responses.pop(0)


def run(
    client,
    messages=None,
    tools=None,
    functions=None,
    limit=3,
):
    if messages is None:
        messages = []

    return run_harness(
        client=client,
        messages=messages,
        tools=[CALCULATE_TOOL] if tools is None else tools,
        tool_functions=functions or {},
        max_iterations=limit,
        model="test-model",
    )


def test_direct_answer():
    client = FakeClient([
        response("The answer is 42.")
    ])

    result = run(client)

    assert result["status"] == "completed"
    assert result["answer"] == "The answer is 42."
    assert result["llm_calls"] == 1
    assert result["tool_calls"] == 0


def test_tool_error_is_returned_to_model():
    client = FakeClient([
        response(tool_calls=[
            tool_call(
                "calculate",
                '{"operation":"divide","a":10,"b":0}',
            )
        ]),
        response("Division by zero is not allowed."),
    ])

    messages = []
    result = run(
        client,
        messages=messages,
        functions={"calculate": calculate},
    )

    tool_message = next(
        m for m in messages
        if isinstance(m, dict) and m.get("role") == "tool"
    )
    tool_result = json.loads(tool_message["content"])

    assert tool_result["ok"] is False
    assert tool_result["error"] == "Division by zero"
    assert result["status"] == "completed"
    assert result["tool_calls"] == 1


def test_unknown_tool_is_not_executed():
    executed = []

    def forbidden_tool(**kwargs):
        executed.append(kwargs)
        return {"ok": True}

    client = FakeClient([
        response(tool_calls=[
            tool_call("forbidden_tool", '{"x":1}')
        ]),
        response("The tool was rejected."),
    ])

    result = run(
        client,
        functions={"forbidden_tool": forbidden_tool},
    )

    assert executed == []
    assert result["status"] == "completed"
    assert result["tool_calls"] == 1


def test_invalid_json_is_handled():
    client = FakeClient([
        response(tool_calls=[
            tool_call("calculate", '{"operation":')
        ]),
        response("The tool arguments were invalid."),
    ])

    messages = []
    result = run(
        client,
        messages=messages,
        functions={"calculate": calculate},
    )

    tool_message = next(
        m for m in messages
        if isinstance(m, dict) and m.get("role") == "tool"
    )
    tool_result = json.loads(tool_message["content"])

    assert tool_result["ok"] is False
    assert "JSONDecodeError" in tool_result["error"]
    assert result["status"] == "completed"


def test_iteration_limit_prevents_tool_execution():
    executed = []

    def tracked(**kwargs):
        executed.append(kwargs)
        return {"ok": True}

    tracked_tool_schema = {
        "type": "function",
        "function": {
            "name": "tracked",
            "description": "Test tool",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "number"}},
                "required": ["x"],
                "additionalProperties": False,
            },
        },
    }

    client = FakeClient([
        response(tool_calls=[
            tool_call("tracked", '{"x":1}')
        ]),
    ])

    result = run(
        client,
        tools=[tracked_tool_schema],
        functions={"tracked": tracked},
        limit=1,
    )

    assert executed == []
    assert result["status"] == "iteration_limit"
    assert result["tool_calls"] == 0


def test_multiple_tool_calls_are_processed():
    client = FakeClient([
        response(tool_calls=[
            tool_call(
                "calculate",
                '{"operation":"add","a":2,"b":3}',
                "call_1",
            ),
            tool_call(
                "calculate",
                '{"operation":"multiply","a":4,"b":5}',
                "call_2",
            ),
        ]),
        response("The results are 5 and 20."),
    ])

    messages = []
    result = run(
        client,
        messages=messages,
        functions={"calculate": calculate},
    )

    tool_messages = [
        m for m in messages
        if isinstance(m, dict) and m.get("role") == "tool"
    ]
    tool_results = [
        json.loads(m["content"]) for m in tool_messages
    ]

    assert len(tool_messages) == 2
    assert [r["result"] for r in tool_results] == [5, 20]
    assert result["status"] == "completed"
    assert result["tool_calls"] == 2


# V1.2-specific tests


def test_missing_required_argument_is_rejected():
    executed = []

    def tracked(**kwargs):
        executed.append(kwargs)
        return {"ok": True}

    client = FakeClient([
        response(tool_calls=[
            tool_call(
                "calculate",
                '{"operation":"add","a":2}',
            )
        ]),
        response("Invalid arguments were rejected."),
    ])

    messages = []
    result = run(
        client,
        messages=messages,
        functions={"calculate": tracked},
    )

    tool_message = next(
        m for m in messages
        if isinstance(m, dict) and m.get("role") == "tool"
    )

    assert executed == []
    assert "Missing required arguments" in tool_message["content"]
    assert result["status"] == "completed"


def test_api_error_is_reported_without_retry():
    class BrokenClient:
        def __init__(self):
            self.calls = 0
            self.chat = NS(
                completions=NS(create=self.create)
            )

        def create(self, **kwargs):
            self.calls += 1
            raise RuntimeError("Simulated API failure")

    client = BrokenClient()
    result = run(client, limit=3)

    assert result["status"] == "api_error"
    assert result["llm_calls"] == 1
    assert client.calls == 1


def test_duplicate_tool_call_ids_are_rejected():
    client = FakeClient([
        response(tool_calls=[
            tool_call(
                "calculate",
                '{"operation":"add","a":1,"b":2}',
                "duplicate",
            ),
            tool_call(
                "calculate",
                '{"operation":"add","a":3,"b":4}',
                "duplicate",
            ),
        ]),
    ])

    result = run(
        client,
        functions={"calculate": calculate},
    )

    assert result["status"] == "protocol_error"
    assert result["tool_calls"] == 0
    assert result["error_type"] == (
        "Invalid or duplicate tool_call_id"
    )