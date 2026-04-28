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
from .session_logger import slog


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


async def execute_fast(tool_name: str, args: dict, registry, mcp_client) -> str:
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
                # Direct await — mcp_client.call is async, never wrap in to_thread
                result = await mcp_client.call(server, tool_name, args)
                return result if isinstance(result, str) else str(result)
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
            from .tool_registry import format_tool_schemas_for_observation
            q_emb = await encode_fn(query, convert_to_tensor=True)
            q_emb_cpu = q_emb.to("cpu")
            if q_emb_cpu.dim() == 1:
                q_emb_cpu = q_emb_cpu.unsqueeze(0)
            schemas = registry.search(q_emb_cpu, top_k=4)
            return format_tool_schemas_for_observation(schemas)

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
                # Direct await — mcp_client.call is async
                result = await mcp_client.call(server, tool_name, mcp_args)
                return result if isinstance(result, str) else str(result)
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
        "list_directory":        {"depth": 2},
        "write_file":            {"mode": "rewrite"},
    }

    stripped = query.strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            slog.debug(f"   [Executor] '{tool_name}' JSON-parsed query → {list(parsed.keys())}")
            for arg, default_val in TOOL_ARG_DEFAULTS.get(tool_name, {}).items():
                if arg not in parsed:
                    parsed[arg] = default_val
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
