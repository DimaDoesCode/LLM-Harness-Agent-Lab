import json
from types import SimpleNamespace

from harness_lab.controller_v2 import run_harness


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer"},
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_order_total",
            "parameters": {
                "type": "object",
                "properties": {
                    "quantity": {"type": "integer"},
                    "unit_price": {"type": "number"},
                },
                "required": ["quantity", "unit_price"],
                "additionalProperties": False,
            },
        },
    },
]


def tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments),
        ),
    )


def response(content=None, tool_calls=None):
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        # Snapshot the message list at the time of this LLM call.
        self.calls.append({
            **kwargs,
            "messages": list(kwargs["messages"]),
        })

        if not self.responses:
            raise AssertionError("Unexpected LLM call")

        return self.responses.pop(0)


def order_details(order_id):
    assert order_id == 1042
    return {
        "ok": True,
        "order_id": 1042,
        "quantity": 3,
        "unit_price": 120,
    }


def calculate_total(quantity, unit_price):
    return {
        "ok": True,
        "total": quantity * unit_price,
    }


TOOL_FUNCTIONS = {
    "get_order_details": order_details,
    "calculate_order_total": calculate_total,
}


def initial_messages():
    return [
        {"role": "system", "content": "Test sequential tool calls."},
        {"role": "user", "content": "Find the total for order 1042."},
    ]


def test_sequential_tool_calls_preserve_history_and_order():
    client = FakeClient([
        response(tool_calls=[
            tool_call(
                "call_1",
                "get_order_details",
                {"order_id": 1042},
            )
        ]),
        response(tool_calls=[
            tool_call(
                "call_2",
                "calculate_order_total",
                {"quantity": 3, "unit_price": 120},
            )
        ]),
        response(content="Order 1042 has a total of 360."),
    ])

    result = run_harness(
        client=client,
        messages=initial_messages(),
        tools=TOOLS,
        tool_functions=TOOL_FUNCTIONS,
        max_iterations=3,
    )

    assert result["status"] == "completed"
    assert result["llm_calls"] == 3
    assert result["tool_calls"] == 2

    assert [event["tool"] for event in result["events"]] == [
        "get_order_details",
        "calculate_order_total",
    ]

    assert result["events"][0]["arguments"] == {"order_id": 1042}
    assert result["events"][0]["result"]["quantity"] == 3
    assert result["events"][0]["result"]["unit_price"] == 120

    assert result["events"][1]["arguments"] == {
        "quantity": 3,
        "unit_price": 120,
    }
    assert result["events"][1]["result"]["total"] == 360

    # The second LLM request must include the first tool result.
    second_request_messages = client.calls[1]["messages"]

    first_tool_result = [
        message
        for message in second_request_messages
        if isinstance(message, dict)
        and message.get("role") == "tool"
    ]

    assert len(first_tool_result) == 1
    assert first_tool_result[0]["tool_call_id"] == "call_1"

    first_result_content = json.loads(first_tool_result[0]["content"])
    assert first_result_content["quantity"] == 3
    assert first_result_content["unit_price"] == 120

    # The third LLM request must include the second tool result too.
    third_request_messages = client.calls[2]["messages"]

    tool_results = [
        message
        for message in third_request_messages
        if isinstance(message, dict)
        and message.get("role") == "tool"
    ]

    assert len(tool_results) == 2
    assert [message["tool_call_id"] for message in tool_results] == [
        "call_1",
        "call_2",
    ]

    second_result_content = json.loads(tool_results[1]["content"])
    assert second_result_content["total"] == 360

    assert result["answer"] == "Order 1042 has a total of 360."


def test_iteration_limit_prevents_last_tool_execution():
    client = FakeClient([
        response(tool_calls=[
            tool_call(
                "call_1",
                "get_order_details",
                {"order_id": 1042},
            )
        ]),
        response(tool_calls=[
            tool_call(
                "call_2",
                "calculate_order_total",
                {"quantity": 3, "unit_price": 120},
            )
        ]),
    ])

    result = run_harness(
        client=client,
        messages=initial_messages(),
        tools=TOOLS,
        tool_functions=TOOL_FUNCTIONS,
        max_iterations=2,
    )

    assert result["status"] == "iteration_limit"
    assert result["llm_calls"] == 2
    assert result["tool_calls"] == 1
    assert [event["tool"] for event in result["events"]] == [
        "get_order_details"
    ]


def test_multiple_tool_calls_in_one_turn_are_rejected():
    client = FakeClient([
        response(tool_calls=[
            tool_call(
                "call_1",
                "get_order_details",
                {"order_id": 1042},
            ),
            tool_call(
                "call_2",
                "calculate_order_total",
                {"quantity": 3, "unit_price": 120},
            ),
        ]),
    ])

    result = run_harness(
        client=client,
        messages=initial_messages(),
        tools=TOOLS,
        tool_functions=TOOL_FUNCTIONS,
        max_iterations=3,
    )

    assert result["status"] == "protocol_error"
    assert result["tool_calls"] == 0
    assert result["error_type"] == (
        "Expected one tool call per assistant turn"
    )


def test_api_error_is_reported():
    client = FakeClient([])
    client.responses = []

    def fail(**kwargs):
        client.calls.append(kwargs)
        raise RuntimeError("Simulated API failure")

    client.chat.completions.create = fail

    result = run_harness(
        client=client,
        messages=initial_messages(),
        tools=TOOLS,
        tool_functions=TOOL_FUNCTIONS,
        max_iterations=3,
    )

    assert result["status"] == "api_error"
    assert result["llm_calls"] == 1
    assert result["tool_calls"] == 0
    assert result["error_type"] == "RuntimeError"