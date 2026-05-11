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
import re
from .session_logger import slog
from .resource_ledger import resource_ledger

# ── Filesystem tools whose paths we track in filesystem_map ──────────────────
_FS_PATH_TOOLS = frozenset({
    "read_file", "write_file", "list_directory", "create_directory",
    "get_more_search_results",
})


def _update_filesystem_map(tool_name: str, args: dict, result: str) -> None:
    """
    After a successful filesystem tool call, update the filesystem_map tree.
    Called from both execute_fast and execute_deliberate — never raises.
    """
    if tool_name not in _FS_PATH_TOOLS:
        return
    if not isinstance(result, str):
        return
    lowered = result.lstrip().lower()
    if lowered.startswith("error:") or lowered.startswith("tool error:"):
        return
    try:
        from .crud import crud as _crud
        if tool_name in ("read_file", "write_file"):
            path = args.get("path", "")
            if path:
                _crud.merge_filesystem_path(path, is_file=True)

        elif tool_name == "create_directory":
            path = args.get("path", "")
            if path:
                _crud.merge_filesystem_path(path, is_file=False)

        elif tool_name == "list_directory":
            path = args.get("path", "")
            if path:
                _crud.merge_filesystem_path(path, is_file=False)
                _parse_list_directory_into_map(path, result, _crud)

        elif tool_name == "get_more_search_results":
            _parse_search_paths_into_map(result, _crud)

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
                        crud.merge_filesystem_path(f"{parent}\\{name}", is_file=not is_dir)
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
                crud.merge_filesystem_path(f"{parent}\\{name}", is_file=not is_dir)
            except Exception:
                pass


def _parse_search_paths_into_map(result: str, crud) -> None:
    """Extract Windows file paths from search result text and add to filesystem_map."""
    # Match paths like E:\something\file.py or C:\Users\...
    for match in re.finditer(r'[A-Za-z]:\\(?:[^\s:\n\r"\'<>|?*]+\\)*[^\s:\n\r"\'<>|?*]+\.[A-Za-z0-9]{1,10}', result):
        try:
            crud.merge_filesystem_path(match.group(0), is_file=True)
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

# ── Native tool imports ───────────────────────────────────────────────────────
from .tools import (
    web_search,
    run_python_code,
    get_time_date,
    consult_archive,
)
from .tools import analyze_image_grok
from . import tools as _tools_module  # live reference so _tools_module._xai_client_ref is read at call time

# ── NATIVE_TOOLS — handled by Python functions, not MCP ──────────────────────
NATIVE_TOOLS = frozenset({
    "web_search", "python_repl", "date_time", "vision_tool",
    "consult_archive", "query_task_status", "tool_search",
})


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
            return await asyncio.to_thread(run_python_code, args.get("code", ""))

        elif tool_name == "date_time":
            return await asyncio.to_thread(get_time_date)

        elif tool_name == "consult_archive":
            return await asyncio.to_thread(consult_archive, args.get("query", ""))

        elif tool_name == "vision_tool":
            if _tools_module._xai_client_ref is None:
                return "Error: Vision client not yet initialized. Retry in a moment."
            path     = args.get("path", "")
            question = args.get("question", "Describe this image.")
            paths    = args.get("paths", None)
            result   = await asyncio.to_thread(
                analyze_image_grok, _tools_module._xai_client_ref, path, question, paths
            )
            if path and "temp_image_" in path:
                import pathlib
                pathlib.Path(path).unlink(missing_ok=True)
            return result

        elif tool_name == "query_task_status":
            from .tools import query_task_status as _qts
            return await asyncio.to_thread(_qts, args.get("keyword", ""))

        elif tool_name == "tool_search":
            # tool_search in FAST is an edge case — route to DELIBERATE
            return "Error: tool_search requires DELIBERATE mode. Escalating."

        # ── MCP tools ─────────────────────────────────────────────────────────
        elif registry is not None:
            server = registry.get_server(tool_name)
            if server and server != "native" and mcp_client is not None:
                # Resource ledger: write conflict check + exclusive lock
                if task_id and tool_name == "write_file":
                    path = args.get("path", "")
                    if path:
                        ok, reason = resource_ledger.check_write(task_id, path)
                        if not ok:
                            return f"Error: {reason}"
                        write_lock = await resource_ledger.acquire_write(path, task_id)
                        try:
                            result = await mcp_client.call(server, tool_name, args)
                        finally:
                            write_lock.release()
                    else:
                        result = await mcp_client.call(server, tool_name, args)
                else:
                    result = await mcp_client.call(server, tool_name, args)

                result_str = result if isinstance(result, str) else str(result)

                # Resource ledger: record read hash after successful read_file
                if task_id and tool_name == "read_file":
                    path = args.get("path", "")
                    if path and not result_str.lower().startswith(("error:", "tool error:")):
                        resource_ledger.record_read(task_id, path, result_str)

                _update_filesystem_map(tool_name, args, result_str)
                return result_str
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
            return await asyncio.to_thread(run_python_code, code or query)

        elif tool_name == "date_time":
            return await asyncio.to_thread(get_time_date)

        elif tool_name == "consult_archive":
            (q,) = _extract_param(query, "query")
            return await asyncio.to_thread(consult_archive, q or query)

        elif tool_name == "vision_tool":
            if _tools_module._xai_client_ref is None:
                return "Error: Vision client not yet initialized. Retry in a moment."
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
            result = await asyncio.to_thread(analyze_image_grok, _tools_module._xai_client_ref, path, q)
            if "temp_image_" in path:
                import pathlib
                pathlib.Path(path).unlink(missing_ok=True)
            return result

        elif tool_name == "query_task_status":
            from .tools import query_task_status as _qts
            (keyword,) = _extract_param(query, "keyword")
            return await asyncio.to_thread(_qts, keyword or query)

        # ── MCP tools ─────────────────────────────────────────────────────────
        elif registry is not None:
            server = registry.get_server(tool_name)
            if server and server != "native" and mcp_client is not None:
                schema = registry.get_schema(tool_name)
                mcp_args = _build_args_from_query(tool_name, query, schema)

                # Resource ledger: write conflict check + exclusive lock
                if task_id and tool_name == "write_file":
                    path = mcp_args.get("path", "")
                    if path:
                        ok, reason = resource_ledger.check_write(task_id, path)
                        if not ok:
                            return f"Error: {reason}"
                        write_lock = await resource_ledger.acquire_write(path, task_id)
                        try:
                            result = await mcp_client.call(server, tool_name, mcp_args)
                        finally:
                            write_lock.release()
                    else:
                        result = await mcp_client.call(server, tool_name, mcp_args)
                else:
                    result = await mcp_client.call(server, tool_name, mcp_args)

                result_str = result if isinstance(result, str) else str(result)

                # Resource ledger: record read hash after successful read_file
                if task_id and tool_name == "read_file":
                    path = mcp_args.get("path", "")
                    if path and not result_str.lower().startswith(("error:", "tool error:")):
                        resource_ledger.record_read(task_id, path, result_str)

                _update_filesystem_map(tool_name, mcp_args, result_str)
                return result_str
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
