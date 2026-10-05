"""
Unified tool executor for CLARA.

Single entry point for all tool calls from both FAST and DELIBERATE paths.
Routes to native Python functions or MCP servers based on tool registry lookup.
Normalizes all results to strings. All error handling centralized here.

FAST path:       execute_fast(tool_name, args_dict, registry, mcp_client)
DELIBERATE path: execute_deliberate(tool_name, query_str, registry, mcp_client, encode_fn)

Two entry points exist because FAST uses structured args (dict from Interpreter)
while DELIBERATE uses a flat query string parsed from Action: [...] blocks.
In the streaming migration (future brief), both will converge to structured args.

CRITICAL: mcp_client.call() is async — call it directly with await.
Never wrap async MCP calls in asyncio.to_thread or asyncio.run.
"""

import asyncio
import json
import os
import re
from .session_logger import slog
from .resource_ledger import resource_ledger
from . import admissibility

# ── Filesystem tools whose paths we track in filesystem_map ──────────────────
_FS_PATH_TOOLS = frozenset({
    "read_file", "write_file", "list_directory", "create_directory",
    "get_more_search_results",
})

# The agent's LIVE crud instance, injected by api.py at startup (set_db). The old code
# did `from .crud import crud as _crud` — importing the CLASS and calling instance
# methods on it, so every merge raised TypeError into the defensive except and the
# Phase-B fsmap auto-population NEVER actually ran (Brief 36 C-35, found during the
# Brief 37 double-check). A fresh crud() here would be equally wrong: it would load its
# own stale memory snapshot and clobber the agent's in-RAM state on save.
_db = None


def set_db(db) -> None:
    """Called once by api.py after the agent (and its crud) is created."""
    global _db
    _db = db


def _update_filesystem_map(tool_name: str, args: dict, result: str) -> None:
    """
    After a successful filesystem tool call, update the filesystem_map tree.
    Called from both execute_fast and execute_deliberate — never raises.
    All merges are batched (save=False) into ONE _save_memory at the end —
    a full-file fsync per child path was the hottest write source (B-2).
    """
    if tool_name not in _FS_PATH_TOOLS or _db is None:
        return
    if not isinstance(result, str):
        return
    lowered = result.lstrip().lower()
    # G37 (2026-08-11) — REQUIRE POSITIVE EVIDENCE, don't merely check for an "error:" prefix.
    #
    # The old guard was `startswith("error:")` only, and the recorded path comes from the tool ARGS.
    # So any failure phrased differently — an empty result, a chunk-limit note, a differently-worded
    # refusal — was treated as confirmation the path exists, and a path the model merely GUESSED got
    # written into long-term memory as established fact. That is how `E:\ML PROJECTS` (with a space,
    # a directory that does not exist) acquired a fabricated four-child subtree and started being
    # injected into every request as [FILE SYSTEM MAP], costing a wasted turn whenever the agent had
    # to disambiguate two plausible project roots.
    #
    # Corroborated in the same run that found it: she reported `list_directory` "returned empty output
    # even for directories I know are populated" — precisely the case the old guard waved through.
    if not lowered:
        return                                    # empty result proves nothing
    if any(t in lowered[:200] for t in ("error:", "tool error:", "not found", "no such file",
                                        "cannot find", "does not exist", "access is denied",
                                        "permission denied", "enoent", "failed")):
        return
    try:
        if tool_name in ("read_file", "write_file"):
            path = args.get("path", "")
            if path:
                _db.merge_filesystem_path(path, is_file=True, save=False)

        elif tool_name == "create_directory":
            path = args.get("path", "")
            if path:
                _db.merge_filesystem_path(path, is_file=False, save=False)

        elif tool_name == "list_directory":
            path = args.get("path", "")
            if path:
                _db.merge_filesystem_path(path, is_file=False, save=False)
                _parse_list_directory_into_map(path, result, _db)

        elif tool_name == "get_more_search_results":
            _parse_search_paths_into_map(result, _db)

        _db._save_memory()

    except Exception:
        pass  # never disrupt tool execution


def _parse_list_directory_into_map(parent_path: str, result: str, crud) -> None:
    """Parse list_directory output and add children to filesystem_map."""
    parent = parent_path.rstrip("\\").rstrip("/")
    # Try JSON array first (DC may return structured JSON)
    try:
        data = json.loads(result)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    is_dir = item.get("isDirectory", item.get("type", "") == "directory")
                    if name:
                        crud.merge_filesystem_path(f"{parent}\\{name}", is_file=not is_dir, save=False)
            return
    except (json.JSONDecodeError, Exception):
        pass
    # Text fallback — common DC format: lines with filenames/dirnames
    for line in result.splitlines():
        line = line.strip()
        if not line or ":" in line[:3]:  # skip header lines like "Contents of E:\..."
            continue
        is_dir = line.endswith("/") or line.endswith("\\") or "(directory)" in line.lower() or "[dir]" in line.lower()
        name = re.split(r'\s{2,}|\t|\s+\(', line)[0].strip().rstrip("/\\")
        if name and ("." in name or is_dir):
            try:
                crud.merge_filesystem_path(f"{parent}\\{name}", is_file=not is_dir, save=False)
            except Exception:
                pass


def _parse_search_paths_into_map(result: str, crud) -> None:
    """Extract Windows file paths from search result text and add to filesystem_map."""
    # Match paths like E:\something\file.py or C:\Users\...
    for match in re.finditer(r'[A-Za-z]:\\(?:[^\s:\n\r"\'<>|?*]+\\)*[^\s:\n\r"\'<>|?*]+\.[A-Za-z0-9]{1,10}', result):
        try:
            crud.merge_filesystem_path(match.group(0), is_file=True, save=False)
        except Exception:
            pass


def _extract_param(query: str, *param_names: str, fallback: str = "") -> tuple:
    """
    Extract named params from a possibly-JSON query string produced by _validate_actions.
    When the model uses named params, the whole action item is serialized as JSON.
    Returns a tuple of values in the order of param_names.
    Falls back to (query, fallback, fallback, ...) if not JSON.
    """
    if query.strip().startswith("{"):
        try:
            parsed = json.loads(query)
            return tuple(str(parsed.get(p, fallback)) for p in param_names)
        except json.JSONDecodeError:
            pass
    return (query,) + (fallback,) * (len(param_names) - 1)


# ── Atomic search (Fix 1) ─────────────────────────────────────────────────────
# start_search is a two-phase tool: the first call returns a session handle with
# "Status: RUNNING / Total results: 0", and the real results only arrive from
# get_more_search_results. A model reading "Total results: 0" while RUNNING cannot
# distinguish it from "0 matches found" — the cause of confident false negatives
# (Q05 MAX_ATTEMPTS, Q06 resource_callback: both real strings reported as absent).
# We make search atomic here: start, then poll get_more until COMPLETED, returning
# only the terminal result so the caller never sees the ambiguous mid-state. This
# also lets FAST run searches correctly — it could never make the second call.
SEARCH_POLL_INTERVAL = 0.35   # seconds between get_more polls
MAX_SEARCH_POLLS     = 20     # ~7s ceiling; DC searches usually finish in 1-2 polls,
                              # headroom so a slow large search is not cut off mid-scan
_SESSION_ID_RE = re.compile(r'(search_\d+_\d+)')


async def _atomic_search(server: str, mcp_client, args: dict) -> str:
    """
    Run start_search and poll get_more_search_results until the search reports
    COMPLETED (or a timeout / error). Returns the final aggregated result text.
    Never surfaces the 'RUNNING / 0 results' intermediate state to the caller.
    """
    # A filePattern of "*"/"**"/"" means "all files" — already the default — so passing
    # it is redundant and (observed in the 2026-05-31 os.replace false negative) can make
    # the search return a spurious 0. Drop it so the search uses its correct default.
    if isinstance(args, dict) and str(args.get("filePattern", "")).strip() in ("*", "**", ""):
        args = {k: v for k, v in args.items() if k != "filePattern"}

    start_raw = await mcp_client.call(server, "start_search", args)
    start_str = start_raw if isinstance(start_raw, str) else str(start_raw)

    if start_str.lower().startswith(("error", "tool error")):
        return start_str
    # If DC did not ask us to paginate, the start result is already terminal.
    if "get_more_search_results" not in start_str and "Status: RUNNING" not in start_str:
        return start_str
    m = _SESSION_ID_RE.search(start_str)
    if not m:
        return start_str

    session_id = m.group(1)
    last = start_str
    completed = False
    for _ in range(MAX_SEARCH_POLLS):
        await asyncio.sleep(SEARCH_POLL_INTERVAL)
        more_raw = await mcp_client.call(
            server, "get_more_search_results",
            {"sessionId": session_id, "offset": 0, "length": 100},
        )
        more_str = more_raw if isinstance(more_raw, str) else str(more_raw)
        if more_str.lower().startswith(("error", "tool error")):
            return more_str
        last = more_str
        # "Status: COMPLETED" is the terminal marker; "Total results found" appears
        # on the completed payload. Either means polling can stop.
        if "Status: COMPLETED" in more_str or "Total results found" in more_str:
            completed = True
            break
    if not completed:
        # Timed out while still RUNNING. Make the incompleteness explicit so a slow
        # search is never read as "no matches" — that would resurrect the very
        # false-negative this function exists to prevent.
        last += (
            "\n\n[NOTE: this search did not finish within the time budget. The results "
            "above are PARTIAL and INCONCLUSIVE — this is NOT confirmation that no "
            "matches exist. Call get_more_search_results with this sessionId to continue.]"
        )
    elif re.search(r"\b0 (?:matches|results)\b|results found:\s*0\b|results:\s*0\b", last, re.I):
        # A completed search with zero results is the classic false-negative trap (wrong
        # path/scope/pattern or a tool hiccup). Attach the Rule-19 reminder to the tool
        # output itself, so it is seen in the moment — far more effective than the prompt
        # rule alone (which was violated in the 2026-05-31 os.replace false negative).
        last += (
            "\n\n[NOTE: 0 results. This is NOT proof the string is absent — a wrong path, "
            "scope, or pattern can cause a false 0. If you expected matches, re-check the "
            "path/pattern and read a known file to confirm before concluding it is absent.]"
        )
    return last

# ── Native tool imports ───────────────────────────────────────────────────────
from .tools import (
    web_search,
    run_python_code,
    get_time_date,
    consult_archive,
)
from .tools import analyze_image_grok

# ── NATIVE_TOOLS — handled by Python functions, not MCP ──────────────────────
NATIVE_TOOLS = frozenset({
    "web_search", "python_repl", "date_time", "vision_tool",
    "consult_archive", "query_task_status", "tool_search", "ambient_recall",
    "episodic_search", "whatsapp_missed", "ocr_pdf",
})


def _demo_pack():
    """The optional demo toolpack manifest (module named by the DEMO_TOOLPACK env var), or None.
    Re-reads the env each call so tests can toggle it; importlib caches the module so repeat calls
    are cheap. Deal-specific packs live OUTSIDE the tracked tree — core stays generic and never
    names a prospect. A tool only reaches the dispatch below if the registry registered it, which
    only happens when the same env var is set."""
    pack = os.getenv("DEMO_TOOLPACK", "").strip()
    if not pack:
        return None
    try:
        import importlib
        return importlib.import_module(pack)
    except Exception:  # noqa: BLE001
        return None


def _is_demo_tool(tool_name: str) -> bool:
    p = _demo_pack()
    return bool(p) and tool_name in getattr(p, "TOOL_NAMES", frozenset())


def _admit(tool_name: str, args: dict, task_id: str = None):
    """Admissibility gate (BRIEF_54 phase 0) for ONE action, called BEFORE any dispatch or lock.
    Returns an Error-string when an ENFORCED decision blocks the action, else None.

    Mutating tools only; µs when off/shadow (local adapters). In enforce mode a DENY/REVIEW blocks
    the action and returns an Error-string, which rides the EXISTING failure machinery
    (FAST→DELIBERATE escalation / ReAct adaptation), so the agent is never halted, only the action.

    BRIEF_66 (2026-10-04): this used to live inline in _execute_mcp, the only call site, so the
    native python_repl branches below never reached the gate at all. Every path that executes a
    mutating tool calls this one helper. `tests/test_brief66_dispatch_gate.py` drives the real
    dispatch functions and fails if any of them stops reaching it."""
    decision = admissibility.gate(tool_name, args, task_id=task_id)
    if not decision["enforced"]:
        return None
    slog.info(f">> [Admissibility] {decision['verdict']} '{tool_name}' — {decision['reason']} "
              f"(receipt {decision['receipt_id']})")
    if decision["verdict"] == admissibility.DENY:
        return (f"Error: Action denied by admissibility gate — {decision['reason']} "
                f"(receipt {decision['receipt_id']}). This action was NOT executed.")
    # REVIEW: hold + best-effort Telegram note (no-ops if Telegram unconfigured).
    try:
        from .telegram_bot import notifier
        asyncio.create_task(notifier.send(
            f"[Admissibility] Action HELD for review: {tool_name} — {decision['reason']} "
            f"(receipt {decision['receipt_id']})"))
    except Exception:
        pass
    return (f"Error: Action held for review by admissibility gate — {decision['reason']} "
            f"(receipt {decision['receipt_id']}). This action was NOT executed; "
            f"Alkama has been notified for approval.")


async def _execute_mcp(server: str, tool_name: str, args: dict, mcp_client, task_id: str = None) -> str:
    """Single MCP dispatch path shared by execute_fast and execute_deliberate
    (previously ~60 duplicated lines — Brief 36 C-17). Owns, in order:
    the admissibility gate, write-ledger protection, atomic search, read-hash recording,
    fsmap update, and read_file line-stamping.

    Ledger ordering fix (C-16): check_write now runs INSIDE the held write lock.
    The old order (check → acquire → write) let two tasks both pass the hash check
    before either wrote — the second then silently clobbered the first, the exact
    read-modify-write hazard the ledger exists to stop.
    """
    blocked = _admit(tool_name, args, task_id=task_id)
    if blocked:
        return blocked

    if task_id and tool_name == "write_file":
        path = args.get("path", "")
        if path:
            write_lock = await resource_ledger.acquire_write(path, task_id)
            try:
                ok, reason = resource_ledger.check_write(task_id, path)
                if not ok:
                    return f"Error: {reason}"
                result = await mcp_client.call(server, tool_name, args)
            finally:
                write_lock.release()
        else:
            result = await mcp_client.call(server, tool_name, args)
    elif tool_name == "start_search":
        result = await _atomic_search(server, mcp_client, args)
    else:
        result = await mcp_client.call(server, tool_name, args)

    result_str = result if isinstance(result, str) else str(result)

    # Resource ledger: record read hash after successful read_file
    if task_id and tool_name == "read_file":
        path = args.get("path", "")
        if path and not result_str.lower().startswith(("error:", "tool error:")):
            resource_ledger.record_read(task_id, path, result_str)

    _update_filesystem_map(tool_name, args, result_str)
    # Coordinate-drift fix: stamp absolute line numbers (after ledger/raw use).
    if tool_name == "read_file" and not result_str.lower().startswith(("error:", "tool error:")):
        result_str = _number_read_file_lines(result_str, args.get("offset", 0))
    return result_str


def _number_read_file_lines(raw: str, offset) -> str:
    """Stamp correct ABSOLUTE line numbers onto Desktop Commander read_file output.

    DC's read_file returns '[Reading N lines from line M ...]' + a blank line + raw
    content with NO per-line numbers, AND its offset is 0-indexed while the header
    prints the raw offset — so the header is off by one and the model then derives line
    numbers from a wrong base and drifts (the long-standing coordinate-drift bug:
    369->368, 377->378, etc.). We re-stamp each content line as 'ABS: <line>' where
    ABS = offset + i (i 1-indexed → first shown line is offset+1, matching the file's
    1-indexed lines, verified: offset 15 shows line 16). Only touches DC '[Reading' output;
    must be applied to the RETURNED value AFTER resource_ledger.record_read (which needs
    the raw bytes so its hash matches check_write's hash of the on-disk file). (2026-06-02)
    """
    try:
        off = int(offset) if offset is not None else 0
    except (TypeError, ValueError):
        off = 0
    parts = raw.split("\n")
    if not parts or not parts[0].lstrip().startswith("[Reading"):
        return raw
    header, body = parts[0], parts[1:]
    lead = ""
    if body and body[0].strip() == "":
        lead, body = "\n", body[1:]
    numbered = "\n".join(f"{off + i + 1}: {ln}" for i, ln in enumerate(body))
    return f"{header}{lead}\n{numbered}" if numbered else raw


async def execute_fast(tool_name: str, args: dict, registry, mcp_client, task_id: str = None) -> str:
    """
    Execute a tool called from the FAST path.
    args: structured dict from Interpreter output.
    Returns string result or "Error: ..." on failure.

    Routing:
    1. tool_name in NATIVE_TOOLS → run Python function
    2. Otherwise → look up server in registry → direct await mcp_client.call()
    3. Not in registry → return error (triggers FAST→DELIBERATE escalation)
    """
    try:
        if tool_name == "web_search":
            res = await asyncio.to_thread(web_search, args.get("query", ""))
            return res.get("answer", "No results found.")

        elif tool_name == "python_repl":
            code = args.get("code", "")
            blocked = _admit("python_repl", {"code": code}, task_id=task_id)   # BRIEF_66
            if blocked:
                return blocked
            return await asyncio.to_thread(run_python_code, code)

        elif tool_name == "date_time":
            return await asyncio.to_thread(get_time_date, int(args.get("offset_days", 0) or 0),
                                           int(args.get("offset_minutes", 0) or 0))  # Brief 50 + G25 time-delta

        elif tool_name == "consult_archive":
            return await asyncio.to_thread(consult_archive, args.get("query", ""))

        elif tool_name == "vision_tool":
            if not os.getenv("GEMINI_API_KEY", ""):
                return "Error: GEMINI_API_KEY not set in .env. Vision unavailable."
            path     = args.get("path", "")
            question = args.get("question", "Describe this image.")
            paths    = args.get("paths", None)
            result   = await asyncio.to_thread(
                analyze_image_grok, None, path, question, paths
            )
            if path and "temp_image_" in path:
                import pathlib
                pathlib.Path(path).unlink(missing_ok=True)
            return result

        elif tool_name == "query_task_status":
            from .tools import query_task_status as _qts
            return await asyncio.to_thread(_qts, args.get("keyword", ""))

        elif tool_name == "ambient_recall":
            from .tools import ambient_recall as _ar
            return await asyncio.to_thread(
                _ar, args.get("window", "24"), args.get("query", ""), args.get("date", "")
            )

        elif tool_name == "episodic_search":
            from .tools import episodic_search as _es
            return await asyncio.to_thread(_es, args.get("query", ""), args.get("k", 5))

        elif tool_name == "whatsapp_missed":
            from .tools import whatsapp_missed as _wm
            return await asyncio.to_thread(_wm, args.get("query", ""), args.get("limit", 20))

        elif tool_name == "ocr_pdf":
            from .tools import ocr_pdf as _ocr
            return await asyncio.to_thread(_ocr, args.get("path", "") or args.get("query", ""),
                                           args.get("max_pages", 10), args.get("question", ""))

        elif tool_name == "tool_search":
            # tool_search in FAST is an edge case — route to DELIBERATE
            return "Error: tool_search requires DELIBERATE mode. Escalating."

        elif _is_demo_tool(tool_name):
            return await asyncio.to_thread(_demo_pack().dispatch, tool_name, args)

        # ── MCP tools ─────────────────────────────────────────────────────────
        elif registry is not None:
            server = registry.get_server(tool_name)
            if server and server != "native" and mcp_client is not None:
                return await _execute_mcp(server, tool_name, args, mcp_client, task_id=task_id)
            elif server is None:
                return f"Error: Tool '{tool_name}' not found in registry."
            else:
                return f"Error: Tool '{tool_name}' server '{server}' not connected."

        else:
            return f"Error: Unknown tool '{tool_name}' and no registry available."

    except Exception as e:
        return f"Error: {e}"


async def execute_deliberate(
    tool_name: str,
    query: str,
    registry,
    mcp_client,
    encode_fn=None,
    task_id: str = None,
) -> str:
    """
    Execute a tool called from the DELIBERATE ReAct loop.
    query: flat string from Action: [{"tool": "...", "query": "..."}]
    Returns string result.

    Special case: tool_search encodes query and returns schema observation string.
    For MCP tools, query string is mapped to primary required arg via _build_args.
    """
    try:
        if tool_name == "tool_search":
            if registry is None or not registry.is_ready:
                return "Tool registry not available."
            if encode_fn is None:
                return "Encoding function unavailable for tool_search."
            import torch
            from .tool_registry import format_tool_schemas_for_glint
            q_emb = await encode_fn(query, convert_to_tensor=True)
            q_emb_cpu = q_emb.to("cpu")
            if q_emb_cpu.dim() == 1:
                q_emb_cpu = q_emb_cpu.unsqueeze(0)
            schemas = registry.search(q_emb_cpu, top_k=4)
            return format_tool_schemas_for_glint(schemas)

        elif tool_name == "web_search":
            (q,) = _extract_param(query, "query")
            res = await asyncio.to_thread(web_search, q or query)
            return res.get("answer", "No results found.")

        elif tool_name == "python_repl":
            (code,) = _extract_param(query, "code")
            code = code or query
            blocked = _admit("python_repl", {"code": code}, task_id=task_id)   # BRIEF_66
            if blocked:
                return blocked
            return await asyncio.to_thread(run_python_code, code)

        elif tool_name == "date_time":
            return await asyncio.to_thread(get_time_date)

        elif tool_name == "consult_archive":
            (q,) = _extract_param(query, "query")
            return await asyncio.to_thread(consult_archive, q or query)

        elif tool_name == "vision_tool":
            if not os.getenv("GEMINI_API_KEY", ""):
                return "Error: GEMINI_API_KEY not set in .env. Vision unavailable."
            if query.strip().startswith("{"):
                try:
                    parsed = json.loads(query)
                    path = parsed.get("path", "")
                    q = parsed.get("question", parsed.get("query", "Describe this image."))
                except json.JSONDecodeError:
                    parts = query.split(",", 1)
                    path = parts[0].strip().strip('"').strip("'")
                    q = parts[1].strip() if len(parts) > 1 else "Describe this image."
            else:
                parts = query.split(",", 1)
                path = parts[0].strip().strip('"').strip("'")
                q = parts[1].strip() if len(parts) > 1 else "Describe this image."
            result = await asyncio.to_thread(analyze_image_grok, None, path, q)
            if "temp_image_" in path:
                import pathlib
                pathlib.Path(path).unlink(missing_ok=True)
            return result

        elif tool_name == "query_task_status":
            from .tools import query_task_status as _qts
            (keyword,) = _extract_param(query, "keyword")
            return await asyncio.to_thread(_qts, keyword or query)

        elif tool_name == "ambient_recall":
            from .tools import ambient_recall as _ar
            w, q, dt = _extract_param(query, "window", "query", "date")
            # flat non-JSON string = treat as the window/date ("2h", "June 11") or keyword
            return await asyncio.to_thread(_ar, w or "24", q or "", dt or "")

        elif tool_name == "episodic_search":
            from .tools import episodic_search as _es
            (q,) = _extract_param(query, "query")
            return await asyncio.to_thread(_es, q or query, 5)

        elif tool_name == "whatsapp_missed":
            from .tools import whatsapp_missed as _wm
            (q,) = _extract_param(query, "query")
            return await asyncio.to_thread(_wm, q or "", 20)

        elif tool_name == "ocr_pdf":
            from .tools import ocr_pdf as _ocr
            (pth,) = _extract_param(query, "path")
            return await asyncio.to_thread(_ocr, pth or query.strip(), 10, "")

        elif _is_demo_tool(tool_name):
            p = _demo_pack()
            return await asyncio.to_thread(p.dispatch, tool_name, p.args_from_query(tool_name, query))

        # ── MCP tools ─────────────────────────────────────────────────────────
        elif registry is not None:
            server = registry.get_server(tool_name)
            if server and server != "native" and mcp_client is not None:
                schema = registry.get_schema(tool_name)
                mcp_args = _build_args_from_query(tool_name, query, schema)
                # Multi-arg rejection (C-20): _build_args returns {"error": guidance}
                # when a multi-required-arg tool got a flat string. Previously that
                # dict was sent to the MCP server AS ARGUMENTS — the model saw DC's
                # terse validation error instead of the crafted guidance. Return it.
                if isinstance(mcp_args, dict) and set(mcp_args.keys()) == {"error"}:
                    return f"Error: {mcp_args['error']}"
                return await _execute_mcp(server, tool_name, mcp_args, mcp_client, task_id=task_id)
            elif server is None:
                return (
                    f"Tool '{tool_name}' not found. "
                    f"Call tool_search to discover available tools."
                )
            else:
                return f"Error: Tool '{tool_name}' server '{server}' not connected."

        else:
            return f"Error: Unknown tool '{tool_name}'."

    except Exception as e:
        return f"Tool error: {e}"


def _build_args_from_query(tool_name: str, query: str, schema) -> dict:
    """
    Build an args dict for an MCP tool from a flat DELIBERATE query string.

    Strategy:
    1. Check if tool has 0 required args → return {} (no-arg tool)
    2. ENFORCE: Multi-arg tools must provide JSON format
    3. Try JSON parse — DELIBERATE may pass structured args inline
    4. Fall back to first required arg from schema (single-arg tools only)
    5. If no schema, use {"query": query}

    Transitional function — goes away when Pattern B (streaming) lands.
    """
    # list_directory special-case: DELIBERATE sometimes passes depth as
    # comma-separated suffix ("E:\\path,3"). Must run before JSON parse.
    if tool_name == "list_directory" and not query.strip().startswith("{"):
        # Legacy flat format: "E:\\path" or "E:\\path,2" (depth as comma suffix)
        parts = query.strip().rsplit(",", 1)
        if len(parts) == 2:
            path_part = parts[0].strip()
            depth_part = parts[1].strip()
            if depth_part.isdigit():
                return {"path": path_part, "depth": int(depth_part)}
        return {"path": query.strip()}

    # No-arg tools (list_searches, get_more_search_results, etc.)
    # should return {} regardless of query value. Prevents empty query errors.
    if schema is not None:
        input_schema = schema.get("inputSchema", {})
        required = input_schema.get("required", [])
        if not required:
            slog.debug(f"   [Executor] '{tool_name}' is a no-arg tool — ignoring query.")
            return {}
        
        # ENFORCEMENT: Multi-arg tools MUST use JSON format with named parameters
        if len(required) > 1 and not query.strip().startswith("{"):
            param_list = ", ".join(required)
            slog.error(
                f"   [Executor] REJECTED '{tool_name}': requires {len(required)} args "
                f"({param_list}) but received flat query string. Must use JSON format with named parameters."
            )
            return {
                "error": (
                    f"Tool '{tool_name}' requires {len(required)} parameters ({param_list}). "
                    f"Provide as JSON with named parameters, e.g.: "
                    f"{{'{required[0]}': '...', '{required[1]}': '...'}}"
                )
            }

    # Defaults applied regardless of whether args came in as JSON or flat string.
    # Only fills args not already present — never overwrites explicit values.
    TOOL_ARG_DEFAULTS = {
        "start_process":         {"timeout_ms": 10000},
        "read_process_output":   {"timeout_ms": 5000},
        "interact_with_process": {"timeout_ms": 8000},
        "list_directory":        {"depth": 0},
        "write_file":            {"mode": "rewrite"},
    }

    # Normalize known wrong values regardless of how args arrived.
    TOOL_ARG_NORMALIZERS = {
        "write_file": {"mode": {"w": "rewrite", "a": "append"}},
    }

    stripped = query.strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            slog.debug(f"   [Executor] '{tool_name}' JSON-parsed query → {list(parsed.keys())}")
            for arg, default_val in TOOL_ARG_DEFAULTS.get(tool_name, {}).items():
                if arg not in parsed:
                    parsed[arg] = default_val
            for arg, mapping in TOOL_ARG_NORMALIZERS.get(tool_name, {}).items():
                if arg in parsed and parsed[arg] in mapping:
                    parsed[arg] = mapping[parsed[arg]]
            return parsed
        except json.JSONDecodeError as e:
            slog.warning(f"   [Executor] '{tool_name}' JSON parse failed: {e}. Falling back to flat string.")

    if schema is None:
        return {"query": query}

    input_schema = schema.get("inputSchema", {})
    required     = input_schema.get("required", [])
    properties   = input_schema.get("properties", {})

    if not required:
        first_prop = next(iter(properties.keys()), "query")
        return {first_prop: query}

    first_required = required[0]
    if len(required) > 1:
        slog.warning(
            f"   [Executor] '{tool_name}' needs {len(required)} required args "
            f"but only flat query available. Missing: {required[1:]}."
        )
    result = {first_required: query}

    for arg, default_val in TOOL_ARG_DEFAULTS.get(tool_name, {}).items():
        if arg not in result:
            result[arg] = default_val

    return result
