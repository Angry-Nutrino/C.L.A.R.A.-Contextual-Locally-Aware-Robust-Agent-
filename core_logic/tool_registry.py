"""
Tool Registry — central store for all tool schemas across native and MCP sources.

Every tool in CLARA's ecosystem is registered here at startup.
At query time, tool_registry.search(q_emb_cpu) returns the most relevant
tool schemas via MiniLM cosine similarity. This enables dynamic tool
discovery without polluting the Interpreter context with all schemas upfront.

Thread safety: _lock guards all writes. Reads of _embeddings/_names are
safe after rebuild_embeddings() completes (single writer, multiple readers).
"""

import threading
import torch
import torch.nn.functional as F
from .session_logger import slog


# ── Native tool schemas ───────────────────────────────────────────────────────
# These mirror TOOL_ARG_SCHEMAS in interpreter.py. Defined here as full schemas
# so they are discoverable via semantic search alongside MCP tool schemas.
# Keep in sync with any changes to interpreter.py TOOL_ARG_SCHEMAS.

NATIVE_TOOL_SCHEMAS = [
    {
        "name": "web_search",
        "_server": "native",
        "description": (
            "Search the web for live or post-training information. "
            "Use ONLY for current prices, news, recent events, live data, "
            "or anything explicitly marked latest, current, today, now. "
            "Do NOT use for stable knowledge answerable from training data."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"]
        }
    },
    {
        "name": "python_repl",
        "_server": "native",
        "description": (
            "Execute Python code. Use for calculations, data transformation, "
            "structured processing, or any task requiring computation. "
            "Always print output. "
            "Do NOT use for file I/O — use read_file/write_file for reading or writing files."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to execute"}},
            "required": ["code"]
        }
    },
    {
        "name": "date_time",
        "_server": "native",
        "description": "Get the current date and time. No arguments needed.",
        "inputSchema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "vision_tool",
        "_server": "native",
        "description": (
            "Analyze images or screenshots using Grok Vision. "
            "Provide absolute file path and a specific question. "
            "Supports single image or multi-image comparison via paths list."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to image file"},
                "question": {"type": "string", "description": "What to ask about the image"},
                "paths": {"type": "array", "items": {"type": "string"},
                          "description": "Optional: multiple image paths for comparison"}
            },
            "required": ["path", "question"]
        }
    },
    {
        "name": "consult_archive",
        "_server": "native",
        "description": (
            "Search the FAISS knowledge base for information from indexed documents "
            "(CLAUDE.md, ROADMAP.md, resume, docs/). Use when the query is about "
            "CLARA architecture, project history, or indexed personal documents."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Question for the archive"}},
            "required": ["query"]
        }
    },
    {
        "name": "query_task_status",
        "_server": "native",
        "description": (
            "Look up the status of any task in the task graph by keyword. "
            "Returns pending, active, running, completed, or failed status."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"keyword": {"type": "string", "description": "Keyword from task goal"}},
            "required": ["keyword"]
        }
    },
]


class ToolRegistry:
    """
    Central registry for all tool schemas across native tools and MCP servers.

    Lifecycle:
        1. register_native_tools() at init — always available
        2. register_server_tools(server_name, schemas) after each MCP handshake
        3. rebuild_embeddings(encode_fn) after all registrations complete
        4. search(q_emb_cpu, top_k) at query time — cosine similarity lookup

    The registry is populated once at startup and read-only during request
    processing. rebuild_embeddings() may be called again if new MCP servers
    connect at runtime.
    """

    def __init__(self):
        self._tools: dict = {}                           # name → full schema dict
        self._embeddings: "torch.Tensor | None" = None  # (N, 384) CPU tensor
        self._names: list = []                           # ordered, matches _embeddings rows
        self._lock = threading.Lock()
        self._embeddings_ready = False

    # ── Registration ──────────────────────────────────────────────────────────

    def register_native_tools(self) -> None:
        """Register all built-in Python tools. Called once at startup."""
        with self._lock:
            for schema in NATIVE_TOOL_SCHEMAS:
                self._tools[schema["name"]] = schema
        slog.info(f"   [Registry] Registered {len(NATIVE_TOOL_SCHEMAS)} native tools.")

    def register_server_tools(self, server_name: str, tools: list) -> int:
        """
        Register tools returned from an MCP server handshake.
        Tags each tool with _server for dispatch routing.
        Cleans descriptions at registration time to improve embedding quality.
        Returns number of tools registered.
        """
        count = 0
        with self._lock:
            for tool in tools:
                schema = dict(tool)
                schema["_server"] = server_name
                if "description" in schema:
                    schema["description"] = self._clean_description(schema["description"])
                self._tools[schema["name"]] = schema
                count += 1
        slog.info(f"   [Registry] Registered {count} tools from '{server_name}'.")
        return count

    def _clean_description(self, desc: str) -> str:
        """
        Extract the first meaningful paragraph from an MCP tool description.
        Strips boilerplate that dominates DC tool descriptions and degrades
        embedding quality by making all DC tools semantically similar.
        Falls back to first 100 chars of raw desc if nothing survives cleaning.
        """
        if not desc:
            return ""
        desc = desc.strip()
        cutoffs = [
            "\nIMPORTANT:",
            "\nOnly works within",
            "\nThis command can be referenced",
            "\nWINDOWS-SPECIFIC",
            "\nCRITICAL RULE:",
            "\nREQUIRED WORKFLOW",
            "\nCOMMON FILE ANALYSIS",
            "\nPERFORMANCE DEBUGGING",
            "\nALWAYS USE FOR:",
        ]
        for cutoff in cutoffs:
            idx = desc.find(cutoff)
            if idx != -1:
                desc = desc[:idx]
        desc = desc.strip()
        if not desc:
            return desc[:100] if desc else ""
        if len(desc) > 200:
            last_period = desc[:200].rfind(".")
            desc = desc[:last_period + 1] if last_period > 50 else desc[:200]
        return desc.strip()

    # ── Embedding ─────────────────────────────────────────────────────────────

    async def rebuild_embeddings(self, encode_fn) -> None:
        """
        Encode all registered tool descriptions with MiniLM.
        encode_fn: agent._encode(texts, convert_to_tensor=True) — async, runs on CUDA.
        Result stored CPU-side for cosine similarity at query time.
        Must be called after all register_*() calls complete.
        """
        with self._lock:
            snapshot = dict(self._tools)

        if not snapshot:
            slog.warning("   [Registry] rebuild_embeddings called with empty registry.")
            return

        names = list(snapshot.keys())
        texts = []
        for name in names:
            schema = snapshot[name]
            desc = schema.get("description", "")
            texts.append(f"{name}: {desc[:300]}")

        slog.info(f"   [Registry] Encoding {len(texts)} tool descriptions...")
        embeddings = await encode_fn(texts, convert_to_tensor=True)
        embeddings_cpu = embeddings.to("cpu")

        with self._lock:
            self._names = names
            self._embeddings = embeddings_cpu
            self._embeddings_ready = True

        slog.info(f"   [Registry] Embeddings ready. Shape: {tuple(embeddings_cpu.shape)}")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, q_emb_cpu: "torch.Tensor", top_k: int = 5) -> list:
        """
        Return top-k most relevant tool schemas via cosine similarity.
        q_emb_cpu: CPU-side tensor from agent._encode(). Shape: (384,) or (1, 384).
        Returns list of schema dicts ordered by relevance (highest first).
        Returns empty list if embeddings not ready or registry empty.
        """
        with self._lock:
            if not self._embeddings_ready or self._embeddings is None:
                return []
            embeddings = self._embeddings
            names = list(self._names)

        if q_emb_cpu.dim() == 1:
            q_emb_cpu = q_emb_cpu.unsqueeze(0)  # (1, 384)

        sims = F.cosine_similarity(q_emb_cpu, embeddings)  # (N,)
        k = min(top_k, len(names))
        top_indices = sims.topk(k).indices.tolist()

        with self._lock:
            return [self._tools[names[i]] for i in top_indices if names[i] in self._tools]

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_schema(self, tool_name: str) -> "dict | None":
        """Get full schema for a specific tool by name."""
        with self._lock:
            return self._tools.get(tool_name)

    def get_server(self, tool_name: str) -> "str | None":
        """Get the server name for a tool ('native', 'desktop_commander', etc.)."""
        with self._lock:
            schema = self._tools.get(tool_name)
            return schema.get("_server") if schema else None

    @property
    def tool_count(self) -> int:
        with self._lock:
            return len(self._tools)

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._embeddings_ready


# ── Formatting helpers ────────────────────────────────────────────────────────

def format_tool_schemas_for_context(schemas: list, tag: str = "DISCOVERED_TOOLS") -> str:
    """
    Format a list of tool schemas into a context string for injection
    before the Interpreter call. Compact but complete.
    """
    if not schemas:
        return ""
    lines = [f"\n[{tag}]"]
    for schema in schemas:
        name = schema.get("name", "unknown")
        desc = schema.get("description", "")
        input_schema = schema.get("inputSchema", {})
        props = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        lines.append(f"\nTool: {name}")
        lines.append(f"Description: {desc[:150]}")
        if props:
            lines.append("Args:")
            for arg_name, arg_info in props.items():
                req_marker = " (required)" if arg_name in required else " (optional)"
                arg_desc = arg_info.get("description", "")
                arg_type = arg_info.get("type", "string")
                lines.append(f"  - {arg_name} [{arg_type}]{req_marker}: {arg_desc}")
    lines.append(f"[/{tag}]\n")
    return "\n".join(lines)


def format_tool_schemas_for_glint(schemas: list) -> str:
    """
    Format tool schemas as a DELIBERATE Glint string.
    Used when tool_search is called as a tool within the ReAct loop.
    Compact — includes enough for the model to immediately make a tool call.
    """
    if not schemas:
        return "No relevant tools found. Try a different search query."
    lines = ["Available tools matching your query:"]
    for schema in schemas:
        name = schema.get("name", "unknown")
        desc = schema.get("description", "")
        input_schema = schema.get("inputSchema", {})
        props = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        lines.append(f"\n{name}:")
        lines.append(f"  {desc[:200]}")
        if props:
            arg_parts = []
            for arg_name, arg_info in props.items():
                req = "*" if arg_name in required else ""
                arg_parts.append(f"{arg_name}{req}")
            lines.append(f"  Args: {', '.join(arg_parts)}  (* = required)")
    return "\n".join(lines)
