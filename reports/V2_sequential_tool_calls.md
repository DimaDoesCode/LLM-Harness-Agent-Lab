# V2 — Sequential Tool Calls

## Objective

Evaluate whether a minimal LLM harness can orchestrate dependent tool calls, where the arguments of the second call must be derived by the model from the first tool's result.

## Experimental Setup

- **Model:** `openai/gpt-oss-120b` via Groq
- **Temperature:** 0
- **Scenario:** Retrieve order details and calculate the order total
- **Expected sequence:** `get_order_details` → `calculate_order_total` → final answer
- **Expected values:** quantity = 3, unit price = 120, total = 360

## Evaluation Dimensions

1. **Harness correctness:** message history, tool-result delivery, tool-call IDs, execution order, and termination.
2. **Model behaviour:** selection of the second tool and derivation of its arguments from the first tool result.
3. **End-to-end outcome:** correct final result and normal termination.

## Live Run Results

**Status: PASS**

The live Groq run completed the sequential tool-calling scenario successfully.

- **Model:** `openai/gpt-oss-120b`
- **Temperature:** 0
- **LLM calls:** 3
- **Tool calls:** 2
- **Final status:** `completed`

Observed sequence:

1. The model called `get_order_details` with `order_id=1042`.
2. The tool returned `quantity=3` and `unit_price=120`.
3. The model subsequently called `calculate_order_total` with `quantity=3` and `unit_price=120`.
4. The second tool returned `total=360`.
5. The model produced the final answer: `Order 1042 total: 360.`

All scenario-level success criteria were satisfied.

This result demonstrates one successful live execution. It does not establish repeatability or general reliability across different scenarios.
## Interpretation

Passing deterministic tests demonstrates that the harness handles the tested protocol correctly under controlled responses. It does not establish that the model will reliably choose the correct sequence or derive the correct arguments.

A successful live run demonstrates one successful execution, not general reliability. Repeated runs with the same frozen scenario would be needed to estimate run-to-run stability.

## Limitations

- One primary scenario
- Two dependent tools
- One model and API provider
- No retries, parallel tool execution, or multi-agent orchestration
- A single successful live run is insufficient to estimate reliability
