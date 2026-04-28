"""
MCP Client — manages subprocess lifecycle and JSON-RPC communication
for MCP servers (Desktop Commander and future servers).

Protocol: JSON-RPC 2.0 over stdio.
Transport: subprocess stdin/stdout pipes.

Windows note: Must use absolute path to node.exe + absolute path to cli.js.
The npx.cmd batch wrapper breaks stdio transport on Windows (known issue, 2026).

Thread safety: asyncio.Lock serializes all JSON-RPC calls per server.
"""

import asyncio
import json
from .session_logger import slog


class MCPError(Exception):
    """Raised when an MCP server returns an error response."""
    pass


class MCPClient:
    """
    Manages one or more MCP server connections.
    Each server runs as a subprocess connected via stdio JSON-RPC.

    Usage:
        client = MCPClient()
        schemas = await client.connect("desktop_commander", command, args)
        result  = await client.call("desktop_commander", "read_file", {"path": "..."})
        await client.disconnect_all()
    """

    def __init__(self):
        self._servers: dict = {}  # server_name → {process, lock, id_counter}

    async def connect(self, server_name: str, command: str, args: list) -> list:
        """
        Start a server subprocess and perform the MCP handshake.
        Returns the list of tool schemas from tools/list.
        Raises MCPError on failure.

        command: absolute path to executable (e.g. node.exe absolute path)
        args: argument list (e.g. [absolute_cli_js_path])
        """
        if server_name in self._servers:
            slog.warning(f"   [MCP] '{server_name}' already connected. Skipping.")
            return await self._list_tools(server_name)

        slog.info(f"   [MCP] Connecting to '{server_name}'...")

        try:
            process = await asyncio.create_subprocess_exec(
                command, *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise MCPError(f"Cannot start '{server_name}': {e}. Check command path.")

        self._servers[server_name] = {
            "process": process,
            "lock": asyncio.Lock(),
            "id_counter": 0,
        }

        try:
            # MCP handshake: initialize → initialized notification → tools/list
            await self._send_request(server_name, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "CLARA", "version": "2.0"}
            })
            await self._send_notification(server_name, "notifications/initialized", {})
            tools = await self._list_tools(server_name)
            slog.info(f"   [MCP] '{server_name}' connected. {len(tools)} tools available.")
            return tools

        except Exception as e:
            await self._kill_server(server_name)
            raise MCPError(f"Handshake failed for '{server_name}': {e}")

    async def call(self, server_name: str, tool_name: str, arguments: dict) -> str:
        """
        Call a tool on the named server. Returns string result.
        Raises MCPError on tool error or server failure.
        """
        if server_name not in self._servers:
            raise MCPError(f"Server '{server_name}' not connected.")

        response = await self._send_request(server_name, "tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })

        # Extract text content from MCP response blocks
        content = response.get("content", [])
        if not content:
            return ""

        parts = []
        for block in content:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "resource":
                resource = block.get("resource", {})
                if "text" in resource:
                    parts.append(resource["text"])

        result = "\n".join(parts).strip()

        if response.get("isError"):
            raise MCPError(f"Tool '{tool_name}' returned error: {result}")

        return result

    async def disconnect_all(self) -> None:
        """Terminate all server subprocesses. Called at api.py shutdown."""
        for server_name in list(self._servers.keys()):
            await self._kill_server(server_name)
        slog.info("   [MCP] All servers disconnected.")

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _list_tools(self, server_name: str) -> list:
        response = await self._send_request(server_name, "tools/list", {})
        return response.get("tools", [])

    async def _send_request(self, server_name: str, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and return the parsed result."""
        state = self._servers[server_name]
        async with state["lock"]:
            state["id_counter"] += 1
            msg_id = state["id_counter"]

            payload = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "method": method,
                "params": params,
            }
            line = json.dumps(payload) + "\n"
            process = state["process"]
            process.stdin.write(line.encode())
            await process.stdin.drain()

            # Read lines until we get a response with matching id
            while True:
                raw = await asyncio.wait_for(
                    process.stdout.readline(), timeout=30.0
                )
                if not raw:
                    raise MCPError(f"'{server_name}' closed stdout unexpectedly.")
                try:
                    data = json.loads(raw.decode().strip())
                except json.JSONDecodeError:
                    continue  # Skip non-JSON lines (server debug logs)

                if data.get("id") == msg_id:
                    if "error" in data:
                        err = data["error"]
                        raise MCPError(
                            f"JSON-RPC error from '{server_name}': "
                            f"[{err.get('code')}] {err.get('message')}"
                        )
                    return data.get("result", {})
                # Ignore messages with different ids (notifications, etc.)

    async def _send_notification(self, server_name: str, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        state = self._servers[server_name]
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        line = json.dumps(payload) + "\n"
        process = state["process"]
        process.stdin.write(line.encode())
        await process.stdin.drain()

    async def _kill_server(self, server_name: str) -> None:
        state = self._servers.pop(server_name, None)
        if state:
            process = state["process"]
            try:
                process.stdin.close()
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except Exception:
                process.kill()
            slog.info(f"   [MCP] '{server_name}' disconnected.")
