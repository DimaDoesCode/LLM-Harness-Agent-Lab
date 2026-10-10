from harness_lab.tools import calculate


def test_add():
    assert calculate("add", 2, 3) == {
        "ok": True,
        "result": 5,
    }


def test_multiply():
    assert calculate("multiply", 4, 5)["result"] == 20


def test_division_by_zero():
    result = calculate("divide", 10, 0)
    assert result["ok"] is False


def test_unsupported_operation():
    result = calculate("power", 2, 3)
    assert result["ok"] is False