import json
import os

from dotenv import load_dotenv
from groq import Groq

from harness_lab.tools import calculate


load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_ITERATIONS = 3

client = Groq(api_key=os.environ["GROQ_API_KEY"])

TOOLS = [
    {
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
]

TOOL_FUNCTIONS = {
    "calculate": calculate,
}


def main():
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Use the calculate tool for arithmetic. "
                "After receiving the tool result, answer the user."
            ),
        },
        {
            "role": "user",
            "content": "Calculate 137 * 246 and explain the result briefly.",
        },
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"\n[LLM CALL {iteration + 1}]")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )

        message = response.choices[0].message
        messages.append(message)

        tool_calls = message.tool_calls or []

        if not tool_calls:
            print("\n[FINAL ANSWER]")
            print(message.content or "(No text response)")
            return

        for tool_call in tool_calls:
            name = tool_call.function.name
            print(f"[TOOL REQUEST] {name}")

            try:
                arguments = json.loads(tool_call.function.arguments)

                if name not in TOOL_FUNCTIONS:
                    result = {
                        "ok": False,
                        "error": f"Unknown tool: {name}",
                    }
                else:
                    result = TOOL_FUNCTIONS[name](**arguments)

            except (ValueError, TypeError, KeyError) as exc:
                result = {
                    "ok": False,
                    "error": f"Invalid tool arguments: {exc}",
                }

            print("[TOOL RESULT]", result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    print(
        "\n[STOP] Maximum iterations reached. "
        "No additional LLM call was made."
    )


if __name__ == "__main__":
    main()