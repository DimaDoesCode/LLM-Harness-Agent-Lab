import json


def run_harness(
    client,
    messages,
    tools,
    tool_functions,
    max_iterations=3,
    model="openai/gpt-oss-120b",
):
    """Run a bounded tool-calling loop."""

    tool_call_count = 0

    for iteration in range(max_iterations):
        print(f"\n[LLM CALL {iteration + 1}]")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
        )

        message = response.choices[0].message
        messages.append(message)

        tool_calls = message.tool_calls or []

        if not tool_calls:
            print("\n[FINAL ANSWER]")
            print(message.content or "(No text response)")

            return {
                "status": "completed",
                "answer": message.content,
                "llm_calls": iteration + 1,
                "tool_calls": tool_call_count,
            }

        # Do not execute tools if no further LLM call is allowed.
        if iteration == max_iterations - 1:
            print("[STOP] Iteration limit reached before tool execution.")

            return {
                "status": "iteration_limit",
                "answer": None,
                "llm_calls": iteration + 1,
                "tool_calls": tool_call_count,
            }

        for tool_call in tool_calls:
            name = tool_call.function.name
            print(f"[TOOL REQUEST] {name}")

            try:
                arguments = json.loads(
                    tool_call.function.arguments
                )

                if not isinstance(arguments, dict):
                    raise ValueError(
                        "Tool arguments must be a JSON object"
                    )

                if name not in tool_functions:
                    result = {
                        "ok": False,
                        "error": f"Unknown tool: {name}",
                    }
                else:
                    result = tool_functions[name](**arguments)

            except Exception as exc:
                result = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

            tool_call_count += 1
            print("[TOOL RESULT]", result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str),
                }
            )

    # Defensive fallback.
    return {
        "status": "iteration_limit",
        "answer": None,
        "llm_calls": max_iterations,
        "tool_calls": tool_call_count,
    }