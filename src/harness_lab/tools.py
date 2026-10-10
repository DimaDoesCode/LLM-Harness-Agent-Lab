def calculate(operation: str, a: float, b: float) -> dict:
    """Execute one of four supported arithmetic operations."""

    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return {"ok": False, "error": "Division by zero"}
        result = a / b
    else:
        return {
            "ok": False,
            "error": f"Unsupported operation: {operation}",
        }

    return {"ok": True, "result": result}