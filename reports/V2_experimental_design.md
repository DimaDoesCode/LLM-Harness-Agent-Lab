# V2 — Sequential Tool Calls: Experimental Design

## 1. Research Question

**Can a minimal LLM harness correctly orchestrate sequential tool calls when later actions depend on earlier tool results?**

The experiment evaluates whether an LLM can use the result of an earlier tool call to select and parameterize a subsequent tool call, while the harness correctly manages the interaction protocol.

## 2. Hypothesis

A minimal harness can correctly execute dependent tool calls when it preserves the conversation history, returns tool results using the expected protocol, and allows the LLM to determine the next action.

The experiment distinguishes harness correctness from model behaviour.

## 3. Experimental Scenario

The model must retrieve order details and then calculate the total price.

### Tool 1: `get_order_details(order_id)`

Input:

`order_id = 1042`

Expected result:

```json
{
  "order_id": 1042,
  "quantity": 3,
  "unit_price": 120
}
```

### Tool 2: `calculate_order_total(quantity, unit_price)`

Expected arguments:

- `quantity = 3`
- `unit_price = 120`

Expected result:

`360`

The second tool call must depend on the first tool's actual result.

The harness must not extract fields from the first result and construct the second call's arguments itself. The LLM must receive the result and generate the next tool call.

### Expected sequence

1. The LLM requests `get_order_details`.
2. The harness executes the tool and records its result.
3. The harness sends the updated conversation history to the LLM.
4. The LLM requests `calculate_order_total` with arguments derived from the returned data.
5. The harness executes the second tool and records its result.
6. The LLM produces a final answer.
7. The harness terminates with a completed status.

## 4. What Is Being Tested?

### Harness correctness

- Correct execution order.
- Preservation of assistant messages and tool calls.
- Correct association between tool calls and tool results using `tool_call_id`.
- Delivery of previous tool results to subsequent LLM requests.
- Correct loop termination.
- Enforcement of the iteration limit.

### Model behaviour

- Selection of the second tool after receiving the first result.
- Correct extraction of the required arguments from that result.
- Correct use of the second tool's output in the final answer.

These are separate evaluation dimensions. Passing deterministic unit tests does not establish that the live model will always select the correct tool or arguments.

## 5. Test Strategy

### Unit tests

Use `pytest` and `FakeClient` to provide deterministic LLM responses.

Tests should verify:

1. The first tool is executed.
2. Its result is appended to the conversation history.
3. The next LLM request contains the earlier tool result.
4. The second tool receives the expected arguments.
5. Tool execution occurs in the correct order.
6. Tool results match their corresponding tool-call IDs.
7. The loop terminates after the final assistant response.
8. The iteration limit prevents execution beyond the allowed boundary.

Negative tests should cover relevant protocol and iteration-limit failures without expanding the scope of the experiment.

Unit tests validate harness logic; they do not measure live LLM reliability.

### Live Groq test

Run the scenario using:

- Model: `openai/gpt-oss-120b`
- Temperature: `0`
- Provider: Groq API

Record the actual sequence of LLM requests, tool calls, tool arguments, tool results, and final response.

The live test evaluates whether the model completes the intended sequence through the real API.

## 6. Metrics

| Metric | Evaluation |
|---|---|
| Harness protocol correctness | Pass/fail from unit tests |
| Tool execution order | First tool precedes second tool |
| Argument dependency | Second call uses values returned by the first tool |
| Final result correctness | Expected total is `360` |
| Loop termination | Final response ends the loop |
| Iteration-limit enforcement | No tool executes beyond the permitted boundary |
| Live scenario outcome | Pass/fail for the complete observed run |

Record LLM request count and tool-call count as diagnostic information, not as independent measures of reliability.

## 7. Success Criteria

The harness implementation passes when all required unit tests pass.

The live scenario passes when:

- The model calls the first tool.
- The model subsequently calls the second tool with arguments derived from the first result.
- Both tool results are correctly returned through the conversation protocol.
- The final result is correct.
- The loop terminates normally.

A failed live run must be classified using the observed trace. Potential causes include model behaviour, malformed tool calls, API failures, and harness protocol errors.

One successful run demonstrates feasibility for that run only. It does not establish general reliability.

## 8. Scope and Limitations

- Two dependent tools.
- One primary scenario.
- Deterministic tool implementations.
- `pytest` with `FakeClient`.
- A separate live Groq test.
- No LangChain, LangGraph, or other agent framework.
- No retries, parallel tool execution, multi-agent coordination, or production-grade orchestration features unless strictly required by the existing harness interface.

The experiment focuses on sequential dependency rather than broad agent capabilities.

## 9. Planned Files

- `src/harness_lab/controller_v2.py`
- `scripts/run_v2_sequential_tool_calls.py`
- `tests/test_controller_v2.py`
- `reports/V2_sequential_tool_calls.md`

Existing V1 and V1.2 files remain unchanged.

## 10. Interpretation

The experiment is designed to answer two distinct questions:

1. Can the harness reliably preserve and execute the sequential tool-calling protocol under controlled tests?
2. Can the live model use earlier tool results to make the next tool call correctly?

Only the combination of deterministic tests and observed live behaviour provides evidence about the complete scenario. Neither alone proves general reliability.
