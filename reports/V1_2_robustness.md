# V1.2 — Harness Robustness

## 1. Objective

V1.2 extends the initial tool-calling harness with basic robustness controls.

The goal is to make tool execution more predictable and to handle common failure conditions without introducing a large framework or automatic retry logic.

This is a small engineering experiment, not a production-readiness assessment.

## 2. Implementation

The following controls were added in `src/harness_lab/controller_v1_2.py`.

### 2.1. Tool argument validation

Tool arguments are validated against a supported subset of JSON Schema before execution.

Implemented checks:
- Arguments must be a JSON object.
- Required arguments must be present.
- Unexpected arguments are rejected when `additionalProperties` is `false`.
- Basic argument types are checked.
- Supported `enum` constraints are enforced.

This is intentionally a limited schema validator, not a complete JSON Schema implementation.

### 2.2. API error handling

API exceptions are caught and returned as a structured result with `status="api_error"`.

Automatic retries are not performed.

This keeps failure handling explicit and avoids introducing additional API calls or retry policies into the experiment.

### 2.3. Tool call ID validation

The controller rejects tool-call batches containing missing or duplicate `tool_call_id` values.

This is a basic protocol safeguard. It does not validate IDs against the entire message history.

### 2.4. Iteration-limit protection

The controller does not execute a requested tool when no further LLM call is available within the configured iteration limit.

This prevents a tool from running when the harness cannot continue the normal tool-result → LLM-response cycle.

## 3. Test Results

The test suite was executed with `pytest`.

| Test file | Tests | Result |
|---|---:|---|
| `tests/test_tools.py` | 4 | Passed |
| `tests/test_controller_v1.py` | 6 | Passed |
| `tests/test_controller_v1_2.py` | 9 | Passed |
| **Total** | **19** | **19 passed** |

Execution time: 0.16 seconds.

The V1.2 tests cover:
- Direct model answers
- Tool execution errors
- Unknown tools
- Invalid JSON arguments
- Iteration-limit protection
- Multiple tool calls
- Missing required arguments
- API error handling without retries
- Duplicate tool-call ID rejection

### Test isolation

The pytest suite uses a `FakeClient` that returns predefined responses.

Therefore:
- Tests run locally.
- Tests do not call the Groq API.
- Tests do not measure actual model quality or reliability.
- The tests verify the controller's behaviour for the scenarios explicitly defined in the test code.

## 4. Live API Smoke Test

A separate script, `scripts/run_v1_2_tool_calling.py`, was executed against the Groq API.

Configuration:
- Model: `openai/gpt-oss-120b`
- Temperature: `0`
- Maximum LLM calls: `3`
- Tool: `calculate`

User request:

`Calculate 137 * 246 and explain the result briefly.`

Observed execution:

| Metric | Result |
|---|---:|
| LLM calls | 2 |
| Tool calls | 1 |
| Tool result | `33702` |
| Final status | `completed` |

The model requested the calculation tool, the harness executed it, and the model returned a final answer based on the tool result.

This confirms that the controller works in one real API interaction. It is a smoke test, not a statistical evaluation.

## 5. Known Limitations

V1.2 remains a minimal educational harness.

1. **Partial JSON Schema support.** Nested constraints, numeric bounds, string patterns, and other schema features are not comprehensively validated.

2. **Limited protocol validation.** The controller checks missing and duplicate tool-call IDs within a single batch, but does not validate the complete message history.

3. **Limited response validation.** Unexpected or malformed API response structures are not comprehensively handled.

4. **Broad tool exception handling.** Some exceptions raised inside a tool can be classified similarly to input-validation errors.

5. **No automatic retry policy.** Transient API failures terminate the current run.

6. **No security sandbox.** Registered tools execute as ordinary Python functions. Argument validation alone does not make arbitrary tools safe.

7. **No systematic adversarial testing.** The current suite covers selected expected failure modes, not an exhaustive set of malformed responses or protocol violations.

## 6. Conclusion

V1.2 adds several useful safeguards while keeping the architecture small and understandable.

The implementation has:
- 19 passing local tests;
- one successful live Groq smoke test;
- basic argument validation;
- explicit API error handling;
- basic tool-call ID checks;
- protection against tool execution at the iteration limit.

The result is a more robust baseline than V1, but it should not be described as production-ready.

**V1.2 is considered complete for the current learning objective.**

The next experiment should focus on a new behavioural question rather than adding more defensive checks to the same controller.