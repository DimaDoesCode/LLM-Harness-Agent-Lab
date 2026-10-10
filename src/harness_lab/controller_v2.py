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
    """Run an LLM harness for sequential tool-call experiments."""

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    schemas = {
        item["function"]["name"]: item["function"]["parameters"]
        for item in tools
        if item.get("type") == "function"
    }

    events = []
    tool_call_count = 0

    def result_dict(status, answer, llm_calls, error_type=None):
        result = {
            "status": status,
            "answer": answer,
            "llm_calls": llm_calls,
            "tool_calls": tool_call_count,
            "events": events,
        }
        if error_type:
            result["error_type"] = error_type
        return result

    for iteration in range(max_iterations):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
            )
        except Exception as exc:
            return result_dict(
                "api_error",
                None,
                iteration + 1,
                type(exc).__name__,
            )

        message = response.choices[0].message
        messages.append(message)

        tool_calls = message.tool_calls or []

        if not tool_calls:
            return result_dict(
                "completed",
                message.content,
                iteration + 1,
            )

        call_ids = [call.id for call in tool_calls]

        if (
            any(not call_id for call_id in call_ids)
            or len(call_ids) != len(set(call_ids))
        ):
            return result_dict(
                "protocol_error",
                None,
                iteration + 1,
                "Invalid or duplicate tool_call_id",
            )

        # This experiment requires each dependent action to be
        # selected after the previous tool result becomes available.
        if len(tool_calls) != 1:
            return result_dict(
                "protocol_error",
                None,
                iteration + 1,
                "Expected one tool call per assistant turn",
            )

        # A tool result must be followed by another LLM call.
        # Do not execute a tool when no iteration remains.
        if iteration == max_iterations - 1:
            return result_dict(
                "iteration_limit",
                None,
                iteration + 1,
            )

        tool_call = tool_calls[0]
        name = tool_call.function.name

        event = {
            "tool_call_id": tool_call.id,
            "tool": name,
            "arguments": None,
            "result": None,
        }

        try:
            arguments = json.loads(tool_call.function.arguments)
            event["arguments"] = arguments

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
                "error": f"Tool execution failed: {type(exc).__name__}",
            }

        event["result"] = result
        events.append(event)
        tool_call_count += 1

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result, default=str),
        })

    return result_dict(
        "iteration_limit",
        None,
        max_iterations,
    )