import json
import os

from dotenv import load_dotenv
from groq import Groq

from harness_lab.controller_v2 import run_harness


load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_ITERATIONS = 3

client = Groq(api_key=os.environ["GROQ_API_KEY"])


def get_order_details(order_id: int) -> dict:
    """Return details for the test order."""

    orders = {
        1042: {
            "order_id": 1042,
            "quantity": 3,
            "unit_price": 120,
        }
    }

    if order_id not in orders:
        return {
            "ok": False,
            "error": f"Order {order_id} not found",
        }

    return {
        "ok": True,
        **orders[order_id],
    }


def calculate_order_total(
    quantity: int,
    unit_price: float,
) -> dict:
    """Calculate an order total from supplied values."""

    return {
        "ok": True,
        "total": quantity * unit_price,
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": "Retrieve details for an order by ID.",
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
            "description": (
                "Calculate the total price using the quantity "
                "and unit price provided."
            ),
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

TOOL_FUNCTIONS = {
    "get_order_details": get_order_details,
    "calculate_order_total": calculate_order_total,
}


def main():
    messages = [
        {
            "role": "system",
            "content": (
                "You are testing sequential tool use. "
                "For order 1042, first call get_order_details. "
                "Wait for its result before deciding the next action. "
                "Then call calculate_order_total using the quantity "
                "and unit_price returned by the first tool. "
                "Do not calculate the total yourself. "
                "Do not invent or assume tool results. "
                "After receiving the calculation result, answer briefly "
                "with the order ID and total. "
                "Make only one tool call per assistant turn."
            ),
        },
        {
            "role": "user",
            "content": (
                "Find the total price for order 1042. "
                "Use the available tools."
            ),
        },
    ]

    result = run_harness(
        client=client,
        messages=messages,
        tools=TOOLS,
        tool_functions=TOOL_FUNCTIONS,
        max_iterations=MAX_ITERATIONS,
        model=MODEL,
    )

    print("\n[RUN SUMMARY]")
    print(f"Model: {MODEL}")
    print(f"Status: {result['status']}")
    print(f"LLM calls: {result['llm_calls']}")
    print(f"Tool calls: {result['tool_calls']}")

    print("\n[TOOL EVENTS]")
    print(json.dumps(result["events"], indent=2, default=str))

    print("\n[FINAL ANSWER]")
    print(result["answer"])

    if result.get("error_type"):
        print(f"Error type: {result['error_type']}")

    # Keep the exit status useful for automated runs.
    if result["status"] != "completed":
        raise SystemExit(1)

    expected_tools = [
        "get_order_details",
        "calculate_order_total",
    ]

    actual_tools = [
        event["tool"] for event in result["events"]
    ]

    second_args = (
        result["events"][1]["arguments"]
        if len(result["events"]) > 1
        else {}
    )

    answer = result.get("answer") or ""

    scenario_passed = (
        actual_tools == expected_tools
        and len(result["events"]) == 2
        and second_args.get("quantity") == 3
        and second_args.get("unit_price") == 120
        and result["events"][0]["result"].get("quantity") == 3
        and result["events"][0]["result"].get("unit_price") == 120
        and result["events"][1]["result"].get("total") == 360
        and result["llm_calls"] == 3
        and "1042" in answer
        and "360" in answer
    )
    
    print(f"\n[SCENARIO RESULT] {'PASS' if scenario_passed else 'FAIL'}")

    if not scenario_passed:
        raise SystemExit(1)

    return result


if __name__ == "__main__":
    main()