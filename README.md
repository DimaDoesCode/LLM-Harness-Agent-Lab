# LLM Harness & Agent Lab

A hands-on project for understanding LLM harnesses, tool calling, and agent execution through small, reproducible experiments.

## Goals

- Understand the LLM tool-calling protocol.
- Implement a minimal harness in Python without agent frameworks.
- Explore agent loops, execution limits, and error handling.
- Evaluate reliability and token usage on controlled tasks.

## Experiments

### V1 — Minimal Tool Calling

A Groq-hosted LLM requests a Python arithmetic tool. The harness executes the function, returns its result to the model, and collects the final answer.

**Stack:** Python, Groq API, official Groq SDK.

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
```

Run local tests without API calls:

```powershell
pytest
```

## Scope

This project starts with a manually implemented tool-calling harness. Agent frameworks and multi-agent architectures are intentionally excluded from the initial experiments.