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


from harness_lab.controller import run_harness

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

    result = run_harness(
        client=client,
        messages=messages,
        tools=TOOLS,
        tool_functions=TOOL_FUNCTIONS,
        max_iterations=3,
        model=MODEL,
    )

    print("\n[RUN SUMMARY]")
    print(result)


if __name__ == "__main__":
    main()