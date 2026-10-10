# LLM Harness & Agent Lab

A hands-on project for exploring LLM harnesses, tool calling, and agent execution through small, reproducible experiments.

## Goals

- Understand the LLM tool-calling protocol.
- Implement a minimal harness in Python without agent frameworks.
- Test sequential tool execution, execution limits, and error handling.
- Separate harness correctness from model behaviour.

## Experiments

### V1 — Minimal Tool Calling

A Groq-hosted LLM requests a Python arithmetic tool. The harness executes the function, returns its result to the model, and collects the final answer.

### V1.2 — Tool Calling with Validation

Extends the basic harness with tool-argument validation and additional error-handling tests.

### V2 — Sequential Tool Calls

Tests whether an LLM can use the result of one tool call to generate the arguments for a subsequent tool call. The harness passes tool results back through the conversation history rather than constructing the next call itself.

## Stack

Python, Groq API, official Groq SDK, pytest.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
Copy-Item .env.example .env
```

Add your Groq API key to `.env`.

## Run

```powershell
python scripts/run_v1_tool_calling.py
python scripts/run_v1_2_tool_calling.py
python scripts/run_v2_sequential_tool_calls.py
```

Run local tests without API calls:

```powershell
pytest
```

## Scope and Limitations

The project uses a manually implemented harness without agent frameworks or multi-agent architectures.

Experiments distinguish deterministic harness tests from live LLM runs. A successful live run demonstrates that a scenario can work, but does not establish general model reliability.