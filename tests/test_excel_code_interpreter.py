"""Tests for the Excel code-interpreter agent tool (backend/extraction/excel/
excel_tool.py) — runs LLM-generated Python/pandas code against real spreadsheet
data in a sandboxed subprocess. Registered in build_agent_registry() and the
agent's system prompt MANDATES using it for Excel questions, so this is a real,
reachable production surface for LLM-generated code, not a toy.

Previously had a manual __main__ self-test script only (never ran under CI) and
zero pytest coverage. Added here after a live security audit (4-Aug) found a
real, exploitable sandbox escape — see test_class_hierarchy_walk_escape_blocked.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import time

import pandas as pd
import pytest

from backend.extraction.excel.excel_tool import (
    ExcelTool,
    _df_cache,
    _find_dunder_access,
    get_sheets,
    run_code,
    serialize_result,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    _df_cache.clear()
    yield
    _df_cache.clear()


# --- Security: dunder-access blocking -----------------------------------------
# Real live finding (4-Aug): the __builtins__ whitelist blocks calling banned
# NAMES (import, eval, exec, open, getattr) but does NOT gate attribute access
# at all -- ().__class__.__bases__[0].__subclasses__() walked the live class
# hierarchy and found subprocess.Popen with zero blocked builtins involved,
# confirmed executing successfully before _find_dunder_access existed. This is
# the textbook reason "restrict __builtins__" alone is not a real Python
# sandbox; these tests lock in the fix.

def test_class_hierarchy_walk_escape_blocked():
    code = """
result = None
for c in ().__class__.__bases__[0].__subclasses__():
    if 'Popen' in c.__name__:
        result = c.__name__
        break
"""
    r = run_code(code, {})
    assert r["success"] is False
    assert "dunder" in r["error"].lower()


def test_dunder_import_name_blocked():
    r = run_code("result = __import__('os').listdir('.')", {})
    assert r["success"] is False
    assert "dunder" in r["error"].lower()


def test_dunder_globals_access_blocked():
    r = run_code("result = (lambda: 1).__globals__", {})
    assert r["success"] is False
    assert "dunder" in r["error"].lower()


def test_dunder_class_attribute_blocked():
    r = run_code('s = ""\nresult = s.__class__', {})
    assert r["success"] is False
    assert "dunder" in r["error"].lower()


def test_bare_import_still_blocked_by_builtins_whitelist():
    # The original, pre-existing defense -- still exercised, not replaced.
    r = run_code("import os\nresult = os.listdir('.')", {})
    assert r["success"] is False


def test_open_eval_exec_still_blocked_by_builtins_whitelist():
    for code in ("result = open('/etc/passwd').read()",
                 "result = eval('1+1')",
                 "exec('x=1')\nresult = x",
                 "result = getattr(1, 'real')"):
        r = run_code(code, {})
        assert r["success"] is False, f"expected block for: {code}"


def test_find_dunder_access_returns_none_for_clean_code():
    assert _find_dunder_access("result = df['x'].sum()") is None
    assert _find_dunder_access("result = [p for p in df['x'] if p > 1]") is None


def test_find_dunder_access_ignores_unparseable_code():
    # A real SyntaxError should surface normally from compile(), not be masked
    # by the static pre-check.
    assert _find_dunder_access("this is not : valid python(") is None


# --- Legitimate usage is unaffected by the security fix ------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Part No": ["A01", "A02", "A03"],
        "Qty": [3, 5, 2],
        "Price": [10.5, 20.0, 7.25],
    })


def test_sum_column(sample_df):
    r = run_code("result = df['Qty'].sum()", {"df": sample_df})
    assert r["success"] is True
    assert r["result"] == 10


def test_filter_and_count(sample_df):
    r = run_code("result = len(df[df['Qty'] > 2])", {"df": sample_df})
    assert r["success"] is True
    assert r["result"] == 2


def test_string_filtering(sample_df):
    r = run_code("result = df[df['Part No'].str.startswith('A0')]", {"df": sample_df})
    assert r["success"] is True


def test_last_expression_auto_assigned_to_result(sample_df):
    # No explicit `result =` -- the AST rewrite in _execute_in_worker should
    # still capture the trailing expression.
    r = run_code("df['Qty'].sum()", {"df": sample_df})
    assert r["success"] is True
    assert r["result"] == 10


# --- Timeout, truncation, caching (ported from the old manual self-test) -------

def test_infinite_loop_times_out_fast():
    t0 = time.time()
    r = run_code("while True:\n    pass", {}, timeout_sec=2)
    elapsed = time.time() - t0
    assert r["success"] is False
    assert "timeout" in r["error"].lower()
    assert elapsed < 5


def test_infinite_loop_worker_process_actually_terminated():
    # Real, live-confirmed bug (4-Aug): the previous implementation reported a
    # clean TimeoutError to the caller while the underlying OS process kept
    # running an infinite loop at ~100% CPU for 6+ minutes afterward, with no
    # cleanup. This locks in the fix by inspecting the real Process object,
    # not just the returned error message.
    import backend.extraction.excel.excel_tool as et
    real_ctx = mp.get_context("spawn")
    created: list = []

    class _RecordingCtx:
        def Pipe(self, *a, **kw):
            return real_ctx.Pipe(*a, **kw)

        def Process(self, *a, **kw):
            p = real_ctx.Process(*a, **kw)
            created.append(p)
            return p

    with pytest.MonkeyPatch.context() as mpatch:
        mpatch.setattr(et.mp, "get_context", lambda *a, **kw: _RecordingCtx())
        r = et.run_code("while True:\n    pass", {}, timeout_sec=2)

    assert r["success"] is False
    assert len(created) == 1
    assert created[0].is_alive() is False, "worker process was not actually terminated"


def test_output_truncation():
    r = run_code('result = "A" * 8000', {})
    serialized = serialize_result(r.get("result"), max_chars=100)
    assert "truncated" in serialized
    assert len(serialized) < 200


def test_cache_invalidates_on_mtime_change():
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        pd.DataFrame({"A": [1]}).to_excel(tmp_path, index=False)
        s1 = get_sheets(tmp_path)
        val1 = list(s1.values())[0].iloc[0, 0]

        time.sleep(1.1)
        pd.DataFrame({"A": [99]}).to_excel(tmp_path, index=False)
        s2 = get_sheets(tmp_path)
        val2 = list(s2.values())[0].iloc[0, 0]

        assert val1 == 1
        assert val2 == 99
    finally:
        os.remove(tmp_path)


# --- ExcelTool.run() end-to-end (real temp file, not the missing external fixture) --

def test_run_end_to_end_with_real_temp_file():
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        pd.DataFrame({"Qty": [1, 2, 3]}).to_excel(tmp_path, sheet_name="Sheet1", index=False)
        tool = ExcelTool()
        res = tool.run(tmp_path, "result = df['Qty'].sum()", sheet_name="Sheet1")
        assert res["success"] is True
        assert "6" in res["result"]
    finally:
        os.remove(tmp_path)


def test_run_reports_missing_sheet_clearly():
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        pd.DataFrame({"A": [1]}).to_excel(tmp_path, sheet_name="Sheet1", index=False)
        tool = ExcelTool()
        res = tool.run(tmp_path, "result = 1", sheet_name="NoSuchSheet")
        assert res["success"] is False
        assert "not found" in res["error"].lower()
    finally:
        os.remove(tmp_path)


def test_run_missing_file_returns_error_not_exception():
    tool = ExcelTool()
    res = tool.run("/nonexistent/path/does_not_exist.xlsx", "result = 1")
    assert res["success"] is False
    assert "error" in res
