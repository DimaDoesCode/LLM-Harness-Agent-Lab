import json
from types import SimpleNamespace as NS

from harness_lab.controller_v1 import run_harness
from harness_lab.tools import calculate


def tool_call(name, arguments, call_id="call_1"):
    return NS(
        id=call_id,
        function=NS(name=name, arguments=arguments),
    )


def response(content=None, tool_calls=None):
    message = NS(content=content, tool_calls=tool_calls or [])
    return NS(choices=[NS(message=message)])


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
        self.chat = NS(
            completions=NS(create=self.create)
        )

    def create(self, **kwargs):
        self.calls += 1
        return next(self.responses)


def run(client, messages=None, tools=None, functions=None, limit=3):
    if messages is None:
        messages = []

    return run_harness(
        client=client,
        messages=messages,
        tools=tools or [],
        tool_functions=functions or {},
        max_iterations=limit,
        model="test-model",
    )

def test_direct_answer():
    client = FakeClient([response("Hello!")])

    result = run(client)

    assert result["status"] == "completed"
    assert result["answer"] == "Hello!"
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

    payload = json.loads(tool_message["content"])

    assert payload["ok"] is False
    assert "Division by zero" in payload["error"]
    assert result["status"] == "completed"


def test_unknown_tool_is_not_executed():
    executed = []

    def forbidden_tool(**kwargs):
        executed.append(kwargs)
        return {"ok": True}

    client = FakeClient([
        response(tool_calls=[
            tool_call("unknown_tool", "{}")
        ]),
        response("Recovered."),
    ])

    messages = []
    result = run(
        client,
        messages=messages,
        functions={"forbidden_tool": forbidden_tool},
    )

    tool_message = next(
        m for m in messages
        if isinstance(m, dict) and m.get("role") == "tool"
    )

    assert executed == []
    assert "Unknown tool" in tool_message["content"]
    assert result["status"] == "completed"


def test_invalid_json_is_handled():
    client = FakeClient([
        response(tool_calls=[
            tool_call("calculate", "{invalid json")
        ]),
        response("The tool call failed."),
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

    assert "JSONDecodeError" in tool_message["content"]
    assert result["status"] == "completed"


def test_iteration_limit_prevents_tool_execution():
    executed = []

    def tracked_tool(**kwargs):
        executed.append(kwargs)
        return {"ok": True}

    client = FakeClient([
        response(tool_calls=[
            tool_call("tracked", '{"x":1}')
        ]),
    ])

    result = run(
        client,
        functions={"tracked": tracked_tool},
        limit=1,
    )

    assert result["status"] == "iteration_limit"
    assert result["llm_calls"] == 1
    assert executed == []
    assert client.calls == 1


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
        response("Both calculations completed."),
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

    assert len(tool_messages) == 2
    assert [m["tool_call_id"] for m in tool_messages] == [
        "call_1", "call_2"
    ]
    assert json.loads(tool_messages[0]["content"])["result"] == 5
    assert json.loads(tool_messages[1]["content"])["result"] == 20
    assert result["tool_calls"] == 2
    assert result["status"] == "completed"