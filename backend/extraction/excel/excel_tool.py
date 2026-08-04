"""Excel Code Interpreter Agent Tool.

Runs LLM-generated Python/Pandas code against real Excel DataFrames in a
sandboxed subprocess with timeout protection.

Known limits:
- Fixed limits: 15s timeout, 20 rows, 4000 chars - not exposed to the agent.
- Two layers of sandboxing: a __builtins__ whitelist (blocks calling banned
  names like import/eval/exec/open) PLUS a static AST check that rejects any
  dunder attribute/name access (blocks the classic __class__.__bases__[0]
  .__subclasses__() escape, which reaches dangerous classes like
  subprocess.Popen without calling a single blocked builtin -- confirmed live,
  4-Aug, executing successfully before the AST check existed). Together these
  close every escape found so far, but this is still defense-in-depth, not a
  proven-complete sandbox against all possible CPython escapes.
- On timeout the worker OS process is terminated (escalating to kill() if
  needed) -- fixed 4-Aug after live-confirming the previous ProcessPoolExecutor
  version's "reaped by the OS" claim was false: a `while True: pass` worker
  kept running at ~100% CPU for 6+ minutes after its reported timeout.
- No in-worker memory cap; OOM kills the worker and returns a generic error.
- Cold start ~200-500ms per call (process spawn + module imports). By design.
"""
from __future__ import annotations

import io
import os
import sys
import json
import math
import datetime
import collections
import re
from collections import OrderedDict
from dotenv import load_dotenv

# Load environment variables (.env) for DB connection string
load_dotenv()
import multiprocessing as mp
from typing import Any, Optional
from contextlib import redirect_stdout, redirect_stderr

import pandas as pd
import numpy as np
import sqlite3

# ── FIXED SERVER-SIDE CONSTANTS (not exposed to the agent) ────────────────────
_TIMEOUT_SEC = 15
_MAX_ROWS = 20
_MAX_CHARS = 4000

# ── SECURE BUILTINS WHITELIST ─────────────────────────────────────────────────

SAFE_BUILTINS = {
    "len": len, "range": range, "sum": sum, "min": min, "max": max,
    "abs": abs, "round": round, "sorted": sorted, "list": list,
    "dict": dict, "str": str, "int": int, "float": float, "bool": bool,
    "enumerate": enumerate, "zip": zip, "print": print,
    "set": set, "tuple": tuple, "any": any, "all": all,
    "isinstance": isinstance, "type": type,
}


def build_namespace(dataframes: dict) -> dict:
    """Prepare a restricted execution namespace with whitelisted modules."""
    ns = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "sqlite3": sqlite3,
        "math": math,
        "datetime": datetime,
        "collections": collections,
        "re": re,
        "json": json,
    }
    ns.update(dataframes)
    return ns


def _find_dunder_access(code: str) -> str | None:
    """Static AST check: reject any code referencing a dunder name/attribute
    (__class__, __bases__, __subclasses__, __globals__, __import__, etc).

    REAL, EXPLOITABLE vulnerability found live (4-Aug): the __builtins__
    whitelist below blocks calling banned functions (import, eval, exec, open,
    getattr, __import__ as a NAME) -- but attribute access is not gated by
    __builtins__ at all, so
        ().__class__.__bases__[0].__subclasses__()
    walks the live class hierarchy and finds subprocess.Popen (confirmed: this
    exact payload executed successfully and returned "Popen" before this check
    existed) -- one step from arbitrary shell command execution AS THE BACKEND
    PROCESS, with zero blocked builtins involved. This is the well-known,
    textbook reason "restrict __builtins__" is not a real Python sandbox on its
    own; blocking the dunder-attribute vector is the standard mitigation.
    Legitimate pandas/numpy data-analysis code never needs to reference a
    dunder, so this has no real false-positive cost."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None  # let compile() raise the real, informative syntax error
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            return f"blocked: dunder attribute access '.{node.attr}' is not allowed"
        if isinstance(node, ast.Name) and node.id.startswith("__") and node.id.endswith("__"):
            return f"blocked: dunder name '{node.id}' is not allowed"
    return None


def _execute_in_worker(code: str, dataframes: dict) -> dict:
    """Core executor running inside a separate process worker.

    DataFrames are unpickled automatically by the multiprocessing engine.
    """
    import ast

    dunder_err = _find_dunder_access(code)
    if dunder_err:
        return {"success": False, "error": dunder_err, "stdout": "", "stderr": ""}

    ns = build_namespace(dataframes)
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        # Preprocess AST to assign the last expression statement to 'result' if 'result' is not assigned
        try:
            tree = ast.parse(code)
            if tree.body:
                has_result_assign = False
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == "result":
                                has_result_assign = True
                                break
                if not has_result_assign and isinstance(tree.body[-1], ast.Expr):
                    last_expr = tree.body[-1].value
                    new_node = ast.Assign(
                        targets=[ast.Name(id='result', ctx=ast.Store())],
                        value=last_expr
                    )
                    ast.copy_location(new_node, tree.body[-1])
                    tree.body[-1] = new_node
                    ast.fix_missing_locations(tree)
                compiled = compile(tree, "<agent_code>", "exec")
            else:
                compiled = compile(code, "<agent_code>", "exec")
        except Exception:
            # Fall back to standard compilation if AST manipulation fails
            compiled = compile(code, "<agent_code>", "exec")

        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(compiled, ns)

        result = ns.get("result", None)
        return {
            "success": True,
            "result": result,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
        }
    except Exception as e:
        exc_type = type(e)
        err_msg = f"{exc_type.__name__}: {e}"
        if isinstance(e, KeyError) or "Column not found" in str(e):
            col_info = []
            for k, df in dataframes.items():
                if isinstance(df, pd.DataFrame):
                    col_info.append(f"columns in df: {list(df.columns)}")
                elif isinstance(df, dict):
                    for sheet_k, sheet_df in df.items():
                        if isinstance(sheet_df, pd.DataFrame):
                            col_info.append(f"columns in dfs['{sheet_k}']: {list(sheet_df.columns)}")
            if col_info:
                err_msg += f". Available columns: {', '.join(col_info)}"
        return {
            "success": False,
            "error": err_msg,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
        }


# ── CODE EXECUTION WITH TIMEOUT ──────────────────────────────────────────────

def _worker_entry(conn, code: str, dataframes: dict) -> None:
    """Process entry point (must be module-level to be picklable under the
    'spawn' start method). Runs the code and sends the result back over conn."""
    try:
        result = _execute_in_worker(code, dataframes)
        conn.send(result)
    except Exception:
        pass  # parent times this out / treats a silent worker as a crash
    finally:
        conn.close()


_TIMED_OUT = object()  # sentinel, distinguishes "no result yet" from a real None result


def run_code(code: str, dataframes: dict, timeout_sec: int = _TIMEOUT_SEC) -> dict:
    """Run code in a dedicated worker process with a hard timeout that actually
    KILLS the OS process on timeout.

    REAL, LIVE-CONFIRMED BUG (4-Aug) in the previous ProcessPoolExecutor-based
    version: its own docstring claimed a timed-out worker is "reaped by the
    OS" -- false. A `while True: pass` worker was directly observed still
    running at ~100% CPU for 6+ minutes after its 2-second timeout had already
    been reported to the caller as a clean TimeoutError, with no automatic
    cleanup; it only stopped when manually killed. Every timeout on this path
    permanently leaked a CPU-spinning process, and code that runs arbitrary
    LLM-generated content WILL trigger timeouts sometimes -- a real, easily
    triggered resource-exhaustion risk, not a corner case.

    Switched from ProcessPoolExecutor (which offered no clean way to reach the
    real OS process behind a timed-out Future) to a raw multiprocessing.Process
    -- this call site never benefited from pooling anyway (one process per
    call, pool shut down immediately after), so this is strictly simpler too.
    On timeout: terminate(), then escalate to kill() if it's still alive after
    a grace period.
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker_entry, args=(child_conn, code, dataframes), daemon=True)
    proc.start()
    child_conn.close()  # parent only reads

    try:
        if parent_conn.poll(timeout_sec):
            try:
                result = parent_conn.recv()
            except (EOFError, OSError):
                result = None  # worker crashed/was killed mid-send
        else:
            result = _TIMED_OUT
    finally:
        proc.join(timeout=1)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)
        parent_conn.close()

    if result is _TIMED_OUT:
        return {
            "success": False,
            "error": f"TimeoutError: Execution timed out after {timeout_sec} seconds.",
            "stdout": "",
            "stderr": "",
        }
    if result is None:
        return {
            "success": False,
            "error": "Worker process ended without a result (crashed).",
            "stdout": "",
            "stderr": "",
        }
    return result


# ── LRU CACHE (keyed by resolved file path) ──────────────────────────────────

_MAX_CACHED_DOCS = 20
_df_cache: OrderedDict[str, dict] = OrderedDict()


def get_column_variations(name: str) -> list[str]:
    """Generate case, space, underscore, and stem variations for an English column name."""
    variations = [name]
    
    # Stem before parentheses/brackets
    if "(" in name:
        stem = name.split("(")[0].strip()
        if stem:
            variations.append(stem)
    if "[" in name:
        stem = name.split("[")[0].strip()
        if stem:
            variations.append(stem)
            
    expanded = []
    for var in variations:
        expanded.append(var)
        var_lower = var.lower()
        expanded.append(var_lower)
        
        var_nospace = var.replace(" ", "")
        expanded.append(var_nospace)
        expanded.append(var_nospace.lower())
        
        var_under = var.replace(" ", "_")
        expanded.append(var_under)
        expanded.append(var_under.lower())
        
        # 'set up' -> 'setup' variations
        if "set up" in var_lower:
            setup_name = var.replace("set up", "setup").replace("Set up", "Setup").replace("SET UP", "SETUP")
            expanded.append(setup_name)
            expanded.append(setup_name.lower())
            expanded.append(setup_name.replace(" ", "_"))
            expanded.append(setup_name.replace(" ", "_").lower())
            expanded.append(setup_name.replace(" ", ""))
            expanded.append(setup_name.replace(" ", "").lower())
            
    return list(set(expanded))


def post_process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Helper to detect English translation header in row 0, clean the dataframe,
    coerce numeric types, and map columns so both Japanese and English headers work.
    """
    if df.empty or len(df) < 1:
        return df

    # 1. Detect if row 0 contains English translation labels.
    first_row = df.iloc[0].astype(str).tolist()
    english_keywords = {
        "description", "specification", "modify", "drawing", "qty",
        "measure", "material", "machine", "price", "local", "reference"
    }
    
    has_english_translation = False
    for val in first_row:
        val_lower = str(val).lower()
        if any(kw in val_lower for kw in english_keywords):
            has_english_translation = True
            break
            
    if has_english_translation:
        # Drop the first row (translation labels) and copy
        df_clean = df.iloc[1:].copy()
        df_clean.reset_index(drop=True, inplace=True)
        
        # Try to parse numeric types now that text labels are removed
        for col in df_clean.columns:
            try:
                df_clean[col] = pd.to_numeric(df_clean[col])
            except (ValueError, TypeError):
                if df_clean[col].dtype == object:
                    df_clean[col] = df_clean[col].apply(lambda x: str(x).strip() if pd.notna(x) else x)
        
        # Duplicate columns under English labels and variations
        for ja_col, en_col in zip(list(df.columns), first_row):
            if pd.notna(ja_col) and pd.notna(en_col):
                ja_str = str(ja_col).strip()
                en_str = str(en_col).strip()
                if ja_str and en_str and ja_str != en_str:
                    df_clean[en_str] = df_clean[ja_str]
                    
                    # Generate and duplicate variations
                    variations = get_column_variations(en_str)
                    for var_name in variations:
                        if var_name not in df_clean.columns:
                            df_clean[var_name] = df_clean[ja_str]
                            
        return df_clean
        
    return df


def get_sheets(
    resolved_path: str, sheet_name: Optional[str] = None
) -> dict[str, pd.DataFrame]:
    """Load sheets lazily, caching in-memory with LRU eviction.

    - Keyed by canonical resolved file path (not raw user input).
    - If a specific sheet is requested, only that sheet is read from disk.
    - If mtime changes, the entire cache entry is dropped and reloaded fresh.
    - Cache is capped at _MAX_CACHED_DOCS entries; oldest is evicted on overflow.
    """
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Excel file not found: {resolved_path}")

    mtime = os.path.getmtime(resolved_path)
    cached = _df_cache.get(resolved_path)

    # If mtime changed, drop stale entry entirely
    if cached and cached["mtime"] != mtime:
        del _df_cache[resolved_path]
        cached = None

    # Create fresh entry if needed
    if cached is None:
        _df_cache[resolved_path] = {"mtime": mtime, "sheets": {}}
        cached = _df_cache[resolved_path]

    # Move to end (most recently used)
    _df_cache.move_to_end(resolved_path)
    sheets = cached["sheets"]

    if sheet_name and sheet_name != "all":
        # Lazy: load only the requested sheet if not already cached
        if sheet_name not in sheets:
            df = pd.read_excel(resolved_path, sheet_name=sheet_name)
            sheets[sheet_name] = post_process_dataframe(df)
        result = {sheet_name: sheets[sheet_name]}
    else:
        # Load all sheets if not already done
        if "__all_loaded__" not in sheets:
            all_sheets = pd.read_excel(resolved_path, sheet_name=None)
            for k, df in all_sheets.items():
                sheets[k] = post_process_dataframe(df)
            sheets["__all_loaded__"] = True
        result = {k: v for k, v in sheets.items() if k != "__all_loaded__"}

    # Evict oldest entries if over the cap
    while len(_df_cache) > _MAX_CACHED_DOCS:
        _df_cache.popitem(last=False)

    return result


# ── PATH RESOLUTION ───────────────────────────────────────────────────────────

def resolve_document_path(filename_or_id: str) -> str:
    """Find the original file path using the DB, falling back to local files."""
    # 1. Clean the input filename/path to extract raw basename and strip UUID prefixes
    base_name = os.path.basename(filename_or_id)
    clean_name = base_name
    parts = base_name.split("_", 1)
    if len(parts) == 2 and len(parts[0]) == 36:  # UUID prefix pattern length (36 chars)
        clean_name = parts[1]

    # Try resolving via database
    try:
        from backend.storage.postgres_store import PostgresStore
        store = PostgresStore()
        try:
            row = store.conn.execute(
                """
                SELECT file_path, document_id FROM documents
                WHERE document_id::text = %s
                   OR filename = %s
                   OR LOWER(filename) = LOWER(%s)
                   OR filename = %s
                   OR LOWER(filename) = LOWER(%s)
                   OR file_path = %s
                   OR LOWER(file_path) LIKE LOWER(%s)
                """,
                (
                    filename_or_id, 
                    filename_or_id, filename_or_id,
                    clean_name, clean_name,
                    filename_or_id,
                    f"%{base_name}"
                ),
            ).fetchone()
            if row and row[0]:
                resolved = row[0]
                doc_id = str(row[1]) if row[1] else "temp"
                if resolved.startswith("supabase://"):
                    parts = resolved[11:].split("/", 1)
                    bucket = parts[0]
                    key = parts[1]
                    local_path = os.path.join("uploads", f"temp_{doc_id}_{os.path.basename(key)}")
                    if not os.path.exists(local_path):
                        os.makedirs("uploads", exist_ok=True)
                        from backend.storage.supabase_store import download_from_supabase
                        download_from_supabase(bucket, key, local_path)
                    return os.path.abspath(local_path)
                if os.path.exists(resolved):
                    return os.path.abspath(resolved)
                # Try finding under typical paths relative to workspace root
                for prefix in (".", "uploads", "staged"):
                    for p in (os.path.join(prefix, resolved), os.path.join(prefix, os.path.basename(resolved))):
                        if os.path.exists(p):
                            return os.path.abspath(p)
        finally:
            store.close()
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Database path resolution failed: %s", exc)

    # 2. Direct path fallback
    if os.path.exists(filename_or_id):
        return os.path.abspath(filename_or_id)

    # Workspace-relative
    root_path = os.path.join(".", filename_or_id)
    if os.path.exists(root_path):
        return os.path.abspath(root_path)

    # Check directly inside uploads/ or staged/ directories
    for folder in ("uploads", "staged"):
        p = os.path.join(folder, base_name)
        if os.path.exists(p):
            return os.path.abspath(p)
        # Also try matching with clean name
        p_clean = os.path.join(folder, clean_name)
        if os.path.exists(p_clean):
            return os.path.abspath(p_clean)

    # Walk workspace (skip heavy dirs)
    skip = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache"}
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            # Match clean name or raw base name
            if f.lower() in (filename_or_id.lower(), base_name.lower(), clean_name.lower()):
                return os.path.abspath(os.path.join(root, f))

    raise FileNotFoundError(
        f"Could not find document '{filename_or_id}' in database or workspace."
    )


# ── OUTPUT SERIALIZATION & TRUNCATION ─────────────────────────────────────────

def serialize_result(
    result: Any, max_rows: int = _MAX_ROWS, max_chars: int = _MAX_CHARS
) -> str:
    """Format and cap the returned value to prevent context-window blowup."""
    if isinstance(result, pd.DataFrame):
        summary = f"[{len(result)} rows x {len(result.columns)} cols]\n"
        return summary + result.head(max_rows).to_string()
    elif isinstance(result, pd.Series):
        summary = f"Series (len={len(result)})\n"
        return summary + result.head(max_rows).to_string()

    text = str(result)
    if len(text) > max_chars:
        return text[:max_chars] + f"\n...[truncated: exceeded {max_chars} chars]"
    return text


# ── AGENT TOOL ────────────────────────────────────────────────────────────────

class ExcelTool:
    """Agent-callable tool: runs Python/Pandas code on ingested Excel sheets."""

    name = "excel_tool"
    description = (
        "Execute Python code (using pandas, numpy, sqlite3) to filter, calculate, "
        "sum, or query columns in an Excel sheet.\n"
        "RULES:\n"
        "1. Assign your final answer to a variable named `result`.\n"
        "2. The active sheet is loaded as `df`.\n"
        "3. A dict of sheet DataFrames is available as `dfs` (only when sheet_name='all').\n"
        "4. Pre-imported: pd, np, sqlite3, math, datetime, re, json. No other imports."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "filename_or_id": {
                "type": "string",
                "description": "Excel filename (e.g. 'Fiber BOQ.xlsx') or document ID.",
            },
            "code": {
                "type": "string",
                "description": "Python code to execute. Must assign final answer to `result`.",
            },
            "sheet_name": {
                "type": "string",
                "description": "Sheet to load as `df`. Omit or 'all' to load all sheets.",
            },
        },
        "required": ["filename_or_id", "code"],
    }

    def run(
        self, filename_or_id: str, code: str, sheet_name: Optional[str] = None
    ) -> dict[str, Any]:
        try:
            file_path = resolve_document_path(filename_or_id)
            sheets = get_sheets(file_path, sheet_name)

            if not sheets:
                return {"success": False, "error": "No sheets found in file."}

            # Build namespace — only pickle what the worker actually needs
            dataframes: dict[str, Any] = {}
            if sheet_name and sheet_name != "all":
                if sheet_name not in sheets:
                    avail = list(sheets.keys())
                    return {
                        "success": False,
                        "error": f"Sheet '{sheet_name}' not found. Available: {avail}",
                    }
                dataframes["df"] = sheets[sheet_name]
                dataframes["dfs"] = {sheet_name: sheets[sheet_name]}
            else:
                first = list(sheets.keys())[0]
                dataframes["df"] = sheets[first]
                dataframes["dfs"] = sheets

            outcome = run_code(code, dataframes)

            if not outcome["success"]:
                return {
                    "success": False,
                    "error": outcome.get("error", "Unknown error."),
                    "stdout": outcome.get("stdout", ""),
                }

            return {
                "success": True,
                "result": serialize_result(outcome.get("result")),
                "stdout": outcome.get("stdout", ""),
            }

        except Exception as e:
            return {"success": False, "error": f"ToolError: {e}"}

    __call__ = run


# ── SELF-TESTS ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    import tempfile

    tool = ExcelTool()
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        global passed, failed
        if condition:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            print(f"  [FAIL] {name} -- {detail}")

    # ── Find test file ────────────────────────────────────────────────
    test_excel = "Fiber BOQ.xlsx"
    test_path = None
    try:
        test_path = resolve_document_path(test_excel)
        print(f"Found test file: {test_path}\n")
    except FileNotFoundError:
        print(f"Could not find {test_excel}. Skipping file-dependent tests.\n")

    # ── Test 1: List sheets ───────────────────────────────────────────
    print("--- Test 1: List sheets ---")
    if test_path:
        res = tool.run(test_excel, "result = list(dfs.keys())", sheet_name="all")
        check("success", res["success"], str(res.get("error")))
        check("has sheets", "Vendor A" in str(res.get("result", "")))
    else:
        print("  [SKIP]")

    # ── Test 2: Single-sheet calculation ──────────────────────────────
    print("--- Test 2: Single-sheet row count ---")
    if test_path:
        code = "result = f'rows={len(df.dropna(how=\"all\"))}'"
        res = tool.run(test_excel, code, sheet_name="Vendor A")
        check("success", res["success"], str(res.get("error")))
        check("has row count", "rows=" in str(res.get("result", "")))
    else:
        print("  [SKIP]")

    # ── Test 3: Blocked import ────────────────────────────────────────
    print("--- Test 3: Sandbox blocks import os ---")
    res = run_code("import os\nresult = os.listdir('.')", {})
    check("blocked", not res["success"])
    check("error type", "ImportError" in res.get("error", "") or "NameError" in res.get("error", ""),
          res.get("error", ""))

    # ── Test 4: Infinite loop timeout ─────────────────────────────────
    print("--- Test 4: Infinite loop timeout ---")
    t0 = time.time()
    res = run_code("while True:\n    pass", {}, timeout_sec=2)
    elapsed = time.time() - t0
    check("returned error", not res["success"])
    check("is timeout", "TimeoutError" in res.get("error", ""), res.get("error", ""))
    check("returned fast", elapsed < 5, f"took {elapsed:.1f}s")

    # ── Test 5: Truncation ────────────────────────────────────────────
    print("--- Test 5: Output truncation ---")
    res = run_code('result = "A" * 8000', {})
    serialized = serialize_result(res.get("result"), max_chars=100)
    check("truncated", "truncated" in serialized, f"len={len(serialized)}")

    # ── Test 6: Cache mtime invalidation ──────────────────────────────
    print("--- Test 6: Cache mtime invalidation ---")
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        pd.DataFrame({"A": [1]}).to_excel(tmp_path, index=False)
        s1 = get_sheets(tmp_path)
        val1 = list(s1.values())[0].iloc[0, 0]

        time.sleep(1.1)  # ensure mtime changes
        pd.DataFrame({"A": [99]}).to_excel(tmp_path, index=False)
        s2 = get_sheets(tmp_path)
        val2 = list(s2.values())[0].iloc[0, 0]

        check("initial value", val1 == 1, f"got {val1}")
        check("reloaded value", val2 == 99, f"got {val2}")
    finally:
        os.remove(tmp_path)

    # ── Test 7: LRU eviction ──────────────────────────────────────────
    print("--- Test 7: LRU cache eviction ---")
    _df_cache.clear()
    tmp_files = []
    try:
        for i in range(21):
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                tmp_files.append(tmp.name)
            pd.DataFrame({"X": [i]}).to_excel(tmp_files[-1], index=False)
            get_sheets(tmp_files[-1])

        check("cache size <= 20", len(_df_cache) <= _MAX_CACHED_DOCS,
              f"size={len(_df_cache)}")
        check("oldest evicted", tmp_files[0] not in _df_cache,
              f"first key still present")
    finally:
        for f in tmp_files:
            try:
                os.remove(f)
            except OSError:
                pass
        _df_cache.clear()

    # ── Test 8: Selective pickling ────────────────────────────────────
    print("--- Test 8: Selective pickling (single sheet) ---")
    if test_path:
        res = tool.run(test_excel, "result = list(dfs.keys())", sheet_name="Vendor A")
        check("success", res["success"], str(res.get("error")))
        check("only one sheet in dfs", res.get("result", "").strip() == "['Vendor A']",
              res.get("result", ""))
    else:
        print("  [SKIP]")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n=== RESULTS: {passed} passed, {failed} failed ===")
    sys.exit(1 if failed else 0)
