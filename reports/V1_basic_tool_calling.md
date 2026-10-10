# V1 — Basic Tool Calling

## 1. Objective

V1 implements a minimal tool-calling harness in Python using the Groq API.

The purpose is to understand the basic control flow between an LLM and external Python tools without relying on agent frameworks such as LangChain or LangGraph.

The central question is:

**Can a Python controller receive a tool call from an LLM, execute the requested function, return the result to the model, and obtain a final answer?**

V1 establishes the baseline for subsequent harness experiments.

## 2. Architecture

The harness implements the following execution cycle:

1. Send the conversation history and available tool definitions to the LLM.
2. Inspect the model response.
3. If the model returns a final answer, terminate the loop.
4. If the model requests one or more tools, parse their arguments and execute the registered Python functions.
5. Append the tool results to the conversation history.
6. Call the LLM again so it can use the results.
7. Stop when the model returns a final answer or the iteration limit is reached.

The controller is responsible for orchestration. The model selects the requested tool, while Python executes the actual function.

## 3. Implementation

Main files:

- `src/harness_lab/controller_v1.py` — tool-calling loop and control flow.
- `src/harness_lab/tools.py` — implementation of the `calculate` tool.
- `scripts/run_v1_tool_calling.py` — live API execution script.
- `tests/test_controller_v1.py` — controller unit tests.
- `tests/test_tools.py` — tool unit tests.

### Tool: `calculate`

The tool supports four arithmetic operations:

- `add`
- `subtract`
- `multiply`
- `divide`

Division by zero and unsupported operations return structured error results.

The tool is a deliberately simple example. It makes the interaction between the model, controller, and Python function easy to inspect.

## 4. Test Results

The V1 test suite contains six controller tests and four tool tests.

| Test file | Tests | Purpose |
|---|---:|---|
| `tests/test_controller_v1.py` | 6 | Basic controller behaviour and tool-calling flow |
| `tests/test_tools.py` | 4 | Arithmetic operations and tool errors |
| **Total** | **10** | |

The controller tests cover:
- Direct answers without tool execution.
- Returning tool errors to the model.
- Rejecting unknown tools.
- Handling invalid JSON arguments.
- Preventing tool execution when the iteration limit is reached.
- Processing multiple tool calls.

The tool tests cover arithmetic operations, division by zero, and unsupported operations.

### Test isolation

The tests use a fake client with predefined responses.

Consequently, pytest verifies the Python implementation against known scenarios without making real LLM API calls.

Passing these tests does not establish the reliability or correctness of the model itself.

## 5. Live API Smoke Test

The script `scripts/run_v1_tool_calling.py` was executed using the Groq API.

Configuration:
- Model: `openai/gpt-oss-120b`
- Temperature: `0`
- Maximum iterations: `3`
- Tool: `calculate`

User request:

`Calculate 137 * 246 and explain the result briefly.`

Observed result:

| Metric | Result |
|---|---:|
| LLM calls | 2 |
| Tool calls | 1 |
| Tool result | `33702` |
| Execution status | Completed |

The observed sequence was:

1. The LLM requested the `calculate` tool.
2. The Python controller executed the function.
3. The tool returned `33702`.
4. The controller passed the tool result back to the LLM.
5. The LLM produced a final answer.

This confirms that the basic tool-calling loop works in a real API interaction.

It is a smoke test, not a statistical evaluation.

## 6. Known Limitations

V1 is a minimal baseline, not a production-ready agent framework.

Known limitations include:

1. **Limited argument validation.** Tool arguments are not validated against a strict supported subset of JSON Schema before execution.

2. **Basic protocol checks.** Tool-call IDs and the complete message history are not comprehensively validated.

3. **Basic API error handling.** API failures are not handled with a dedicated structured error policy.

4. **Limited failure classification.** Different classes of input, protocol, API, and execution errors are not comprehensively separated.

5. **Iteration-limit behaviour.** The controller has a limit on LLM calls, but V1 does not include the additional V1.2 safeguard that prevents executing a tool when no follow-up LLM call remains.

6. **No systematic model evaluation.** A successful live run does not establish consistent model behaviour across different prompts or tool-call scenarios.

## 7. Conclusion

V1 demonstrates the essential tool-calling mechanism:

**LLM → Python tool → tool result → LLM → final answer.**

The implementation provides a small, understandable baseline for studying harness behaviour.

The test suite checks ten defined scenarios across the controller and arithmetic tool. A separate live API smoke test confirms that the model can request the tool and use its result in a final response.

V1 is considered complete as the initial baseline.

V1.2 builds on this baseline by adding basic argument validation, API error handling, tool-call ID checks, and iteration-limit protection.

The next experiment, V2 — Sequential Tool Calls, will investigate whether the harness can reliably orchestrate multiple dependent tool calls rather than only the basic tool-calling cycle.