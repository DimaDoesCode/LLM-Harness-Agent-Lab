import json


def validate_arguments(arguments, schema):
    """Validate the supported subset of JSON Schema."""

    if not isinstance(arguments, dict):
        raise ValueError("Arguments must be a JSON object")

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    missing = set(required) - set(arguments)
    if missing:
        raise ValueError(
            f"Missing required arguments: {sorted(missing)}"
        )

    if schema.get("additionalProperties") is False:
        extra = set(arguments) - set(properties)
        if extra:
            raise ValueError(
                f"Unexpected arguments: {sorted(extra)}"
            )

    for name, value in arguments.items():
        if name not in properties:
            continue

        rule = properties[name]
        expected_type = rule.get("type")

        if expected_type == "number":
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            )
        elif expected_type == "integer":
            valid = (
                isinstance(value, int)
                and not isinstance(value, bool)
            )
        elif expected_type == "string":
            valid = isinstance(value, str)
        elif expected_type == "boolean":
            valid = isinstance(value, bool)
        elif expected_type == "object":
            valid = isinstance(value, dict)
        elif expected_type == "array":
            valid = isinstance(value, list)
        else:
            valid = True

        if not valid:
            raise ValueError(
                f"Argument {name!r} must have type {expected_type!r}"
            )

        if "enum" in rule and value not in rule["enum"]:
            raise ValueError(
                f"Argument {name!r} must be one of {rule['enum']}"
            )


def run_harness(
    client,
    messages,
    tools,
    tool_functions,
    max_iterations=3,
    model="openai/gpt-oss-120b",
):
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    schemas = {
        item["function"]["name"]: item["function"]["parameters"]
        for item in tools
        if item.get("type") == "function"
    }

    tool_call_count = 0

    for iteration in range(max_iterations):
        print(f"\n[LLM CALL {iteration + 1}]")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
            )
        except Exception as exc:
            print(f"[API ERROR] {type(exc).__name__}: {exc}")
            return {
                "status": "api_error",
                "answer": None,
                "llm_calls": iteration + 1,
                "tool_calls": tool_call_count,
                "error_type": type(exc).__name__,
            }

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

        call_ids = [call.id for call in tool_calls]

        if (
            any(not call_id for call_id in call_ids)
            or len(call_ids) != len(set(call_ids))
        ):
            return {
                "status": "protocol_error",
                "answer": None,
                "llm_calls": iteration + 1,
                "tool_calls": tool_call_count,
                "error_type": "Invalid or duplicate tool_call_id",
            }

        # Do not execute tools if no follow-up LLM call is available.
        if iteration == max_iterations - 1:
            print(
                "[STOP] Iteration limit reached before tool execution."
            )
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

                if name not in schemas or name not in tool_functions:
                    raise ValueError(
                        f"Unknown or unregistered tool: {name}"
                    )

                validate_arguments(arguments, schemas[name])
                result = tool_functions[name](**arguments)

            except (ValueError, TypeError, KeyError) as exc:
                result = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            except Exception as exc:
                result = {
                    "ok": False,
                    "error": (
                        f"Tool execution failed: {type(exc).__name__}"
                    ),
                }

            tool_call_count += 1
            print("[TOOL RESULT]", result)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str),
            })

    return {
        "status": "iteration_limit",
        "answer": None,
        "llm_calls": max_iterations,
        "tool_calls": tool_call_count,
    }