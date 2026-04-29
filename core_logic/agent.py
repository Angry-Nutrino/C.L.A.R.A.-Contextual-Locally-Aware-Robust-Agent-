import re
import json
import asyncio
import base64
import threading
import torch
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from .tools import (
    run_python_code, web_search, get_time_date, consult_archive,
    fs_read_file, fs_list_directory, fs_write_file, fs_run_command,
    query_task_status, get_archive_context,
)
from .system_prompt import SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT, PERSONA, TEMP_SYSTEM_PROMPT
from .bench_logger import Timer, log_request
from .interpreter import interpret, TOOL_ARG_SCHEMAS
from .memory_manager import free_gpu_memory
# from .ears import listen_local
# from .kokoro_mouth import speak
import os
from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import user, system, assistant
from .crud import crud
from .session_logger import slog
from .tool_executor import execute_deliberate, execute_fast
from .tool_registry import format_tool_schemas_for_context
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))


def _turn_message(current_turn: int, max_turns: int, body: str) -> str:
    """
    Prepend [Turn N/M] to every Glint message so the model knows its budget.
    On the final turn, append a forced wrap-up instruction so the model writes
    Final Answer instead of attempting another Action that will never execute.
    """
    next_turn = current_turn + 1
    prefix = f"[Turn {next_turn}/{max_turns}] "
    if next_turn == max_turns:
        suffix = (
            "\n\n[FINAL TURN — no further Actions will execute after this response. "
            "Write your Final Answer now. State: what succeeded, what failed, "
            "what you attempted for each failure, and what remains incomplete if anything.]"
        )
    else:
        suffix = ""
    return f"{prefix}{body}{suffix}"


@dataclass
class TokenUsage:
    """Accumulated token usage for a single request across all LLM calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    call_breakdown: dict = field(default_factory=dict)

    def add(self, label: str, usage_obj) -> None:
        if usage_obj is None:
            return
        p = getattr(usage_obj, 'prompt_tokens', 0) or 0
        c = getattr(usage_obj, 'completion_tokens', 0) or 0
        t = getattr(usage_obj, 'total_tokens', 0) or (p + c)
        cached = getattr(usage_obj, 'cached_prompt_text_tokens', 0) or 0
        self.prompt_tokens += p
        self.completion_tokens += c
        self.total_tokens += t
        self.cached_tokens += cached
        self.call_breakdown[label] = {"prompt": p, "completion": c}

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "breakdown": self.call_breakdown,
        }


def route(interpreted: dict) -> str:
    """
    Decide execution mode from Interpreter output.
    Returns "FAST", "CHAT", or "DELIBERATE".

    FAST:       high confidence, low uncertainty, no planning needed,
                tool is specified with complete args
    CHAT:       high confidence, low uncertainty, no planning needed,
                no tool needed — direct conversational response
    DELIBERATE: anything else
    """
    if (
        interpreted.get("requires_planning") is False
        and interpreted.get("confidence", 0) >= 0.75
        and interpreted.get("uncertainty", 1) <= 0.30
    ):
        if (
            interpreted.get("tool") is not None
            and interpreted.get("args") is not None
        ):
            return "FAST"
        if interpreted.get("tool") is None:
            return "CHAT"
    return "DELIBERATE"


class Clara_Agent:
    def __init__(self, model_name="phi3:mini"):
        self.system_prompt = TEMP_SYSTEM_PROMPT or """

### Role ###
You are CLARA — Alkama's autonomous agent.

### Objective ###
Execute multi-step tasks, retrieve information, and solve problems for Alkama without question or hesitation.

### Context ###
You work for Alkama and must obey his commands without question. You have access to powerful tools and must use them effectively to complete any task.

### Tools ###
You have access to the following tools:

1. Python Repl: Execute Python code for calculations, data analysis, and logic.
   Action: [{"tool": "python_repl", "code": "print(2 + 2)"}]

2. Web Search: Search the internet for real-time information.
   Action: [{"tool": "web_search", "query": "current Bitcoin price"}]

3. Date/Time: Get current date and time.
   Action: [{"tool": "date_time"}]

4. Vision Tool: Analyze images from local disk.
   Action: [{"tool": "vision_tool", "path": "/absolute/path/image.png", "question": "What is in this image?"}]

5. Consult Archive: Look up information from local documents.
   Action: [{"tool": "consult_archive", "query": "technical skills resume"}]

6. Read File: Read file contents from disk.
   Action: [{"tool": "read_file", "path": "E:\\path\\to\\file.txt"}]

7. List Directory: List files and folders in a directory.
   Action: [{"tool": "list_directory", "path": "E:\\path\\to\\folder", "depth": 1}]

8. Write File: Write content to a file on disk.
   Action: [{"tool": "write_file", "path": "E:\\path\\to\\file.txt", "content": "file content", "mode": "w"}]

9. Run Command: Execute shell commands.
   Action: [{"tool": "start_process", "command": "dir C:\\Users"}]

10. Query Task Status: Look up task status by keyword.
    Action: [{"tool": "query_task_status", "keyword": "task name"}]

### Execution Loop ###
Follow this format strictly:
Thought: Why you are doing what you're doing (1-2 sentences, plain English)
Action: [{"tool": "tool_name", "param1": "value1", "param2": "value2"}]
Glint: [Wait for system response]
... repeat until complete ...
Final Answer: Your response to Alkama

### Rules ###
1. ALWAYS output Thought before any Action.
2. Batch independent tool calls in one Action array.
3. Trust Glints — do not re-calculate.
4. On tool failure, diagnose and correct in the next turn.
5. Output Final Answer only when you have all needed information.
6. Never output Final Answer and Action in the same turn.
7. Use python_repl for ALL calculations, never mental math.
8. For filesystem tools, always use FULL ABSOLUTE PATHS.
9. When reading files, synthesize — don't dump raw content.
10. Use proper JSON format with named parameters for all tools.

### Examples ###

User: List files in core_logic directory.
Thought: I need to see the directory structure.
Action: [{"tool": "list_directory", "path": "E:\\ML PROJECTS\\AGENT_ZERO\\core_logic", "depth": 1}]
Glint: [FILE] agent.py [FILE] orchestrator.py ...
Final Answer: The core_logic directory contains agent.py, orchestrator.py, and others.

User: Write a Python test file.
Thought: I'll create a test file with the specified content.
Action: [{"tool": "write_file", "path": "E:\\ML PROJECTS\\AGENT_ZERO\\tests\\test_example.py", "content": "def test_example():\\n    assert True", "mode": "rewrite"}]
Glint: File written successfully.
Final Answer: Created test_example.py in the tests directory.

User: Search for Bitcoin price and today's date.
Thought: These are independent lookups so I can fetch both at once.
Action: [{"tool": "web_search", "query": "Bitcoin price USD"}, {"tool": "date_time"}]
Glint from web_search: Bitcoin is $95,000 USD.
Glint from date_time: 2026-04-27 12:53:55
Final Answer: Bitcoin is $95,000 USD. Today is April 27, 2026.

User: Calculate compound interest: ₹50,000 at 8% for 5 years.
Thought: I need exact calculation — use Python.
Action: [{"tool": "python_repl", "code": "print(round(50000 * (1 + 0.08)**5, 2))"}]
Glint: 73466.44
Final Answer: ₹50,000 at 8% compounded annually for 5 years grows to ₹73,466.44.

### Memory ###
At the start of each conversation, you receive [MEMORY_CONTEXT_BLOCK] containing your episodic history and long-term facts with Alkama.

Treat it as your memory. Use it to maintain continuity and avoid repeating known information.
"""
        self.chat_history = ""
        self.db = crud()
        slog.info(f"Initializing Clara with model : {model_name}")
        self.llm = None
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        slog.info(f"Loading MiniLM model for episodic memory on {device}...")
        self.miniLM = SentenceTransformer('all-MiniLM-L6-v2').to(device)
        self._miniLM_lock = asyncio.Lock()
        self._vault_lock = threading.Lock()  # guards vault writes against concurrent memorize_episode calls
        self._event_loop = None  # set at first async call
        self.episodic_embeddings = self._build_episodic_embeddings()
        self.load_clara(model_name)
        self.tool_registry = None   # injected from api.py after startup
        self.mcp_client = None      # injected from api.py after startup
        slog.info("Brain loaded")

    def _build_episodic_embeddings(self) -> list:
        """
        Cold-start: encode all existing episodic summaries once at startup.
        Returns a list of tensors parallel to db.memory["episodic_log"].
        """
        episodes = self.db.memory.get("episodic_log", [])
        if not episodes:
            slog.info("   [Memory] No episodic entries to embed at startup.")
            return []
        summaries = [ep.get("summary", "") for ep in episodes]
        slog.info(f"   [Memory] Encoding {len(summaries)} episodic entries at startup...")
        embs = self.miniLM.encode(summaries, convert_to_tensor=True)
        return [e.to('cpu') for e in embs]  # store on CPU — must match memorize_episode embeddings

    async def _encode(self, texts, convert_to_tensor=True):
        """
        Thread-safe MiniLM encoder. All miniLM.encode() calls must go through here.
        Acquires asyncio.Lock to prevent concurrent CUDA access from main thread
        and background memorization thread.
        """
        async with self._miniLM_lock:
            return self.miniLM.encode(texts, convert_to_tensor=convert_to_tensor)

    def _encode_sync(self, texts):
        """
        Sync wrapper for use inside background threads (e.g. memorize_episode).
        Submits encode to the event loop and blocks until complete.
        Raises concurrent.futures.TimeoutError if encode takes > 30s.
        """
        future = asyncio.run_coroutine_threadsafe(
            self._encode(texts, convert_to_tensor=True),
            self._event_loop
        )
        return future.result(timeout=30)

    def load_clara(self, model_name="phi3:mini"):
        try:
            # self.llm = Ollama(model=model_name,
            #                   num_ctx=4096,
            #                    stop=["Observation"])
            # print("Clara Brain loaded successfully.")
            self.client = Client(
                api_key=os.getenv("XAI_API_KEY"),
                timeout=120, # Override default timeout with longer timeout for reasoning models
                )
            self.llm = self.client.chat.create(model="grok-4-1-fast-reasoning")
            slog.info("Clara Brain loaded successfully.")

        except Exception as e:
            slog.error(f"Failed to load Clara Brain: {e}")
        
    def unload_clara(self):
        print("Putting Clara Brain to sleep...")
        free_gpu_memory(self.llm)
        self.llm = None
    

    def parse_actions(self, llm_output: str) -> list:
        """
        Three-layer parser for batched JSON action format.
        Always returns a list of dicts: [{"tool": str, "query": str}]
        Returns [] only if absolutely no action is found.
        Each failed extraction is represented as {"tool": None, "query": None, "error": "reason"}.
        """
        # Build VALID_TOOLS dynamically from the registry so all MCP tools
        # and tool_search are always included without manual maintenance.
        # Falls back to core native set if registry is not yet available.
        if self.tool_registry and self.tool_registry.is_ready:
            VALID_TOOLS = set(self.tool_registry._tools.keys()) | {"tool_search"}
        else:
            VALID_TOOLS = {
                "web_search", "python_repl", "date_time", "vision_tool",
                "consult_archive", "query_task_status", "tool_search",
            }

        # ── Locate "Action:" in the output ────────────────────────────────────────
        action_match = re.search(r"Action:\s*", llm_output)
        if not action_match:
            return []

        after_action = llm_output[action_match.end():]

        # ── LAYER 1 & 2: JSON array path ──────────────────────────────────────────
        bracket_start = after_action.find("[")
        if bracket_start != -1:
            # Layer 1: Direct json.loads on everything from [ onward
            candidate = after_action[bracket_start:]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    # Only treat as JSON action format if at least one item is a dict
                    # (a list of strings = misidentified old format, fall through to Layer 3)
                    if any(isinstance(item, dict) for item in parsed):
                        return self._validate_actions(parsed, VALID_TOOLS)
            except json.JSONDecodeError:
                pass

            # Layer 2: Bracket counting to find true closing ]
            depth = 0
            end_idx = -1
            in_string = False
            escape_next = False
            for i, ch in enumerate(candidate):
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\" and in_string:
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break

            if end_idx != -1:
                clean = candidate[:end_idx + 1]
                try:
                    parsed = json.loads(clean)
                    if isinstance(parsed, list):
                        # Only treat as JSON action format if at least one item is a dict
                        if any(isinstance(item, dict) for item in parsed):
                            return self._validate_actions(parsed, VALID_TOOLS)
                except json.JSONDecodeError:
                    pass

            # Both JSON layers failed (or produced no dict items) — fall through to Layer 3

        # ── LAYER 3: Old format fallback  tool_name[input] ────────────────────────
        old_match = re.search(r"(\w+)\[(.+?)\]", after_action, re.DOTALL)
        if old_match:
            tool = old_match.group(1).strip()
            query = old_match.group(2).strip().strip('"').strip("'")
            if tool in VALID_TOOLS:
                slog.warning(f"   [Parser] Fell back to old format. Tool: {tool}")
                return [{"tool": tool, "query": query}]
            else:
                return [{"tool": None, "query": None, "error": f"Unknown tool in old format: '{tool}'"}]

        return []

    def _validate_actions(self, parsed: list, valid_tools: set) -> list:
        """
        Validates each item in a parsed JSON array.
        Returns a list where invalid items are replaced with error dicts.
        Handles both old format (single "query" param) and new format (named parameters).
        """
        result = []
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                result.append({"tool": None, "query": None, "error": f"Item {i} is not a dict"})
                continue

            tool = str(item.get("tool", "")).strip()

            if tool not in valid_tools:
                result.append({"tool": None, "query": None, "error": f"Unknown tool: '{tool}'"})
                continue

            # Get schema to understand tool's required parameters
            schema = None
            required_params = []
            is_no_arg_tool = False
            if self.tool_registry and self.tool_registry.is_ready:
                schema = self.tool_registry.get_schema(tool)
                if schema:
                    required_params = schema.get("inputSchema", {}).get("required", [])
                    is_no_arg_tool = len(required_params) == 0

            # Check for required parameters
            has_required_args = all(param in item for param in required_params) if required_params else True

            # Build query string — three cases:
            # 1. Item uses named params (anything beyond "tool"/"query") → serialize full JSON
            #    so the executor can unpack the right fields regardless of param count.
            # 2. Item uses flat "query" key → use it directly (old-style).
            # 3. No-arg tool → empty string, passes validation below.
            has_named_params = any(k not in ("tool", "query") for k in item)
            if has_named_params:
                query = json.dumps(item)
            else:
                query = str(item.get("query", "")).strip()

            # Validation: all required args must be present OR it's a no-arg tool
            if not has_required_args and not is_no_arg_tool:
                result.append({
                    "tool": None,
                    "query": None,
                    "error": f"Missing required parameters for tool '{tool}': {required_params}"
                })
                continue

            # Validation: flat-query tools (no named params) must have a non-empty query
            if not has_named_params and not is_no_arg_tool and not query:
                result.append({
                    "tool": None,
                    "query": None,
                    "error": f"Empty query for tool '{tool}'"
                })
                continue

            result.append({"tool": tool, "query": query})

        return result
    
    def parse_json_safely(self, text):
        original = text

        # 1. Hard clean: remove BOM + whitespace
        text = text.strip().lstrip("\ufeff")

        # 2. Remove markdown fences properly
        text = re.sub(r"```(?:json)?|```", "", text).strip()

        # 2b. Fix invalid JSON escape sequences (e.g. \' is not valid JSON)
        text = text.replace("\\'", "'")

        # 3. Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            last_error = str(e)

        # 4. Extract smallest JSON object (non-greedy)
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0).strip())
            except json.JSONDecodeError as e:
                last_error = str(e)

        slog.error(f"JSON Parse Failed: {last_error} | Input: {original[:80]}...")
        return None

    def log_system_episode(self, summary: str) -> None:
        """
        Write an autonomous/system episodic entry with a zero-vector embedding
        to maintain episodic_embeddings sync with episodic_log.
        System entries are filtered from retrieval ([AUTONOMOUS] prefix) so
        the zero vector will never pollute semantic search results.
        """
        self.db.add_episodic_log(summary)
        zero_emb = torch.zeros(384, dtype=torch.float32)  # MiniLM output dim
        self.episodic_embeddings.append(zero_emb)

    def memorize_episode(self, chat_snapshot: str):
        """
        Dual-Layer Memory Processing:
        1. Summarizes the last session for the Episodic Stream (Always).
        2. Extracts permanent facts for the Long-Term Vault (Conditional).
        """
        if not chat_snapshot: return
        slog.info("   [Memory] Consolidating memories...")
        
        # The prompt asks for a JSON object with two keys: 'summary' and 'facts'
        memory_prompt = (
            "You are Clara's memory consolidation system. Your job is to compress a raw conversation into a clean memory entry.\n\n"
            "RULES:\n"
            "- Write the summary as a plain description of what was discussed and what happened. Example: 'Alkama asked about Galaxy S26 pricing in India. Clara searched and found base at ₹79,990.'\n"
            "- Do NOT mention internal system details: no 'CHAT mode', 'TASK mode', 'memory context block', 'gatekeeper', 'routing', or any technical pipeline terms.\n"
            "- Do NOT mention the prefix 'Now, execute this request:' — strip it and focus on what Alkama actually said.\n"
            "- Keep the summary to 1-2 sentences focused purely on the content of the exchange.\n\n"
            "- FACTS: Extract only TRULY PERMANENT facts worth remembering forever. A fact qualifies if it is:\n"
            "  * A personal attribute of Alkama (name, relationship, personality trait, confirmed preference)\n"
            "  * A stable project decision or architectural constraint\n"
            "  * A real-world fact about a person, place, or thing that won't change\n"
            "  * Something Alkama explicitly stated as a standing preference or rule\n\n"
            "- DO NOT extract as facts:\n"
            "  * File paths, file counts, file sizes, screenshot metadata, directory listings\n"
            "  * Timestamps, dates of events, or anything time-sensitive\n"
            "  * Tool outputs or Glints (web search results, command output)\n"
            "  * Anything that could be stale within days or weeks\n\n"
            "Output ONLY a JSON object, no extra text:\n"
            "{ \"summary\": \"Alkama asked X. Clara did Y.\", \"facts\": [\"Alkama likes Z\"] }\n"
            "If no permanent facts qualify, leave 'facts' as []."
        )
        
        try:
            
            temp_llm = self.client.chat.create(model="grok-4-1-fast-non-reasoning")
            
            temp_llm.append(system(memory_prompt))
            print(f"Snapshot for Memory Consolidation:\n{chat_snapshot}")  # console only — too large for log
            temp_llm.append(user(f"Interaction:\n{chat_snapshot}"))
            content = temp_llm.sample().content
            slog.info(f"   [Memory] Raw consolidation output: {content}")
            temp_llm=None
            
            # # 2. Sanitize JSON
            # if "```" in content:
            #     content = content.split("```")[1]
            #     if content.startswith("json"):
            #         content = content[4:]
            #     content = content.strip()
            # Using a different approach to extract JSON that is more robust to formatting issues:
            
            # 3. Parse
            data = self.parse_json_safely(content)
            if data is None:
                slog.error("   [Memory] parse_json_safely returned None — raw output was not valid JSON. Consolidation aborted.")
                return

            # 4. Save to The Stream (Episodic)
            summary = data.get("summary", "Interaction completed.")
            self.db.add_episodic_log(summary)
            # Encode and append the new episodic embedding incrementally
            new_emb = self._encode_sync(summary).to('cpu')
            self.episodic_embeddings.append(new_emb)
            slog.info(f"   [Memory] Episodic embedding updated ({len(self.episodic_embeddings)} total)")

            # 5. Save to The Vault (Long Term) - Only if facts exist
            facts = data.get("facts", [])
            if facts and isinstance(facts, list) and len(facts) > 0:
                slog.info(f"   [Memory] Found {len(facts)} permanent facts.")
                with self._vault_lock:
                    # Re-read inside the lock so concurrent threads see each other's writes
                    existing_facts = list(self.db.memory.get("long_term", []))
                    existing_embs = self._encode_sync(existing_facts).to('cpu') if existing_facts else None
                    for fact in facts:
                        # Fast path: exact string match (catches identical concurrent writes)
                        if fact in existing_facts:
                            slog.info(f"   [Memory] Skipping exact duplicate: {fact[:60]}")
                            continue
                        fact_emb = self._encode_sync(fact).to('cpu')
                        if existing_embs is not None:
                            sims = torch.nn.functional.cosine_similarity(fact_emb.unsqueeze(0), existing_embs)
                            if sims.max().item() >= 0.85:
                                slog.info(f"   [Memory] Skipping near-duplicate (sim={sims.max().item():.2f}): {fact[:60]}")
                                continue
                        self.db.add_long_term_fact(fact)
                        existing_facts.append(fact)
                        if existing_embs is not None:
                            existing_embs = torch.cat([existing_embs, fact_emb.unsqueeze(0)], dim=0)
                        else:
                            existing_embs = fact_emb.unsqueeze(0)

        except Exception as e:
            slog.error(f"   [Memory] Consolidation failed: {e}")

    async def process_request(self, query, image_data=None, on_step_update=None,
                               source="user", task_context=None):
        try:
            if self._event_loop is None:
                self._event_loop = asyncio.get_running_loop()
            slog.info(f"\n=== New Mission [{source}]: {query[:80]} ===")

            total_timer = Timer()   # starts now, covers full request

            final_prompt = query
            if image_data:
                try:
                    import uuid as _uuid_mod
                    if "," in image_data:
                        image_data = image_data.split(",")[1]
                    image_path = f"temp_image_{_uuid_mod.uuid4().hex[:8]}.png"
                    with open(image_path, "wb") as f:
                        f.write(base64.b64decode(image_data))
                    abs_path = os.path.abspath(image_path)
                    final_prompt = (
                        f"{query} \n\n[SYSTEM: An image has been uploaded and saved at "
                        f"'{abs_path}'. If the user asks about it, use the 'vision' tool "
                        f"to analyze this file.]"
                    )
                except Exception as e:
                    slog.error(f"   Failed to save image: {e}")

            # 1. Get memory context (always — feeds Interpreter)
            q_emb = await self._encode(final_prompt, convert_to_tensor=True)
            q_emb_cpu = q_emb.to('cpu')
            mem_context = self.db.get_smart_context(
                final_prompt, q_emb_cpu, self.episodic_embeddings
            )

            # 1b. Conditional archive context — same embedding, no extra MiniLM call
            archive_context = await asyncio.to_thread(
                get_archive_context, q_emb_cpu, final_prompt
            )
            if archive_context:
                slog.info(f"   [Archive] Context injected ({len(archive_context)} chars)")

            # Dynamic tool discovery — inject relevant schemas before Interpreter
            tool_context = ""
            if self.tool_registry and self.tool_registry.is_ready:
                discovered = self.tool_registry.search(q_emb_cpu, top_k=8)

                # Mandatory injection: cosine similarity cannot reliably detect
                # enumeration intent (query describes target, not operation).
                # Guarantee list_directory and start_search are present for
                # any query that implies finding or listing files.
                ENUMERATION_KEYWORDS = (
                    "find", "list", "all", "search", "what files", "which files",
                    "show files", "directory", "folder", "files in", "images in",
                    "locate", "where is", "enumerate"
                )
                query_lower = final_prompt.lower()
                if any(kw in query_lower for kw in ENUMERATION_KEYWORDS):
                    discovered_names = {s.get("name") for s in discovered}
                    ld_schema = self.tool_registry.get_schema("list_directory")
                    ss_schema = self.tool_registry.get_schema("start_search")
                    if ld_schema and "list_directory" not in discovered_names:
                        discovered.append(ld_schema)
                        slog.info("   [Registry] Mandatory injection: list_directory")
                    if ss_schema and "start_search" not in discovered_names:
                        discovered.append(ss_schema)
                        slog.info("   [Registry] Mandatory injection: start_search")

                if discovered:
                    tool_context = format_tool_schemas_for_context(discovered)
                    slog.info(
                        f"   [Registry] Injecting {len(discovered)} discovered tools."
                    )
                    slog.debug(
                        f">> [DISCOVERED_TOOLS] Full context injected to Interpreter:\n{tool_context}"
                    )

            full_context = mem_context
            if archive_context:
                full_context += "\n" + archive_context
            if tool_context:
                full_context += tool_context

            # 2. Interpret
            interp_timer = Timer()
            interpreted, interp_usage = await interpret(
                content=final_prompt,
                source=source,
                context=full_context,
                client=self.client,
                task_context=task_context,
            )
            interp_ms = interp_timer.elapsed_ms()

            # 3. Route
            mode = route(interpreted)
            slog.info(f">> [Router] Mode: {mode}")

            # 4. Create a fresh LLM instance per request — isolates concurrent tasks
            #    Model selection by mode:
            #    CHAT:      non-reasoning — ~0.5s TTFT vs 3-8s for reasoning; no planning needed
            #    DELIBERATE: reasoning — ReAct loop quality requires it
            #    FAST:      reasoning (consolidation-only; model matters less here)
            if mode == "CHAT":
                llm = self.client.chat.create(model="grok-4-1-fast-non-reasoning")
            else:
                llm = self.client.chat.create(model="grok-4-1-fast-reasoning")

            if mode == "DELIBERATE":
                llm.append(system(self.system_prompt))

            elif mode == "CHAT":
                llm.append(system(CHAT_SYSTEM_PROMPT))

            # FAST: no system prompt appended — llm is consolidation-only

            llm.append(assistant(
                f"[MEMORY_CONTEXT_BLOCK]\n{full_context}\n[/MEMORY_CONTEXT_BLOCK]"
            ))
            llm.append(user(f"Now, execute this request: {final_prompt}"))

            # 5. Execute
            exec_timer = Timer()
            fast_usage = None
            chat_usage = None
            deliberate_usage_list = []
            if mode == "FAST":
                final_answer, fast_usage = await self._run_fast(interpreted, on_step_update, llm)
            elif mode == "CHAT":
                final_answer, chat_usage = await self._run_chat(llm, on_step_update)
            else:
                final_answer, deliberate_usage_list = await self.run_task(on_step_update=on_step_update, llm=llm)
            exec_ms = exec_timer.elapsed_ms()

            # 6. Token aggregation
            token_usage = TokenUsage()
            if interp_usage:
                token_usage.add("interpreter", interp_usage)
            if mode == "FAST":
                if isinstance(fast_usage, list):
                    for i, u in enumerate(fast_usage):
                        token_usage.add(f"deliberate_turn_{i+1}", u)
                elif fast_usage:
                    token_usage.add("fast_execution", fast_usage)
            elif mode == "CHAT" and chat_usage:
                token_usage.add("chat", chat_usage)
            elif mode == "DELIBERATE":
                for i, u in enumerate(deliberate_usage_list or []):
                    token_usage.add(f"deliberate_turn_{i+1}", u)

            slog.info(
                f">> [Tokens] total={token_usage.total_tokens} "
                f"prompt={token_usage.prompt_tokens} "
                f"completion={token_usage.completion_tokens} "
                f"cached={token_usage.cached_tokens}"
            )

            if source == "user" and on_step_update:
                await on_step_update(
                    "",
                    type="token_usage",
                    turn_id=None,
                    extra=token_usage.to_dict()
                )

            # 7. Benchmark log (user requests only — skip system/background noise)
            if source == "user":
                log_request(
                    mode=mode,
                    tool=interpreted.get("tool"),
                    total_ms=total_timer.elapsed_ms(),
                    interp_ms=interp_ms,
                    exec_ms=exec_ms,
                    query=query,
                    token_usage=token_usage,
                )

            # 6. Memory consolidation
            chat_snapshot = "\n".join([
                f"{'User' if str(m.role) == '1' else 'Clara'}: {m.content}"
                for m in llm.messages
                if str(m.role) not in ['3', 'system']
                and "[MEMORY_CONTEXT_BLOCK]" not in m.content
            ])
            mem_task = asyncio.create_task(
                asyncio.to_thread(self.memorize_episode, chat_snapshot)
            )
            def _on_memorize_done(t):
                if not t.cancelled() and t.exception():
                    slog.error(
                        f"   [Memory] memorize_episode failed: {t.exception()}"
                    )
            mem_task.add_done_callback(_on_memorize_done)

            return final_answer

        except Exception as e:
            slog.error(f"   [process_request] Unhandled error: {e}")
            return f"I encountered an internal error: {e}"

    async def _run_fast(self, interpreted: dict, on_step_update=None, llm=None) -> str:
        """
        FAST_EXECUTION: execute the Interpreter-specified tool directly,
        or respond conversationally when tool is None.
        On any failure, escalate to DELIBERATE_EXECUTION (run_task).
        """
        tool_name = interpreted.get("tool")
        args      = interpreted.get("args", {})
        intent    = interpreted.get("intent", "")

        try:
            tool_result = None
            if tool_name is not None:
                slog.info(f">> [FAST] tool={tool_name} args={str(args)[:80]}")
                if on_step_update:
                    await on_step_update(f"Running {tool_name}...", type="status")
                tool_result = await self._execute_fast_tool(tool_name, args)
                if tool_result.startswith("Error"):
                    raise ValueError(tool_result)

            # Format response with non-reasoning model for speed
            format_llm = self.client.chat.create(model="grok-4-1-fast-non-reasoning")
            if tool_name == "vision_tool":
                # Vision responses must describe ONLY what is visually present.
                # Passing intent (derived from full_context including memory) causes
                # memory details to bleed into the image description.
                format_llm.append(system(
                    PERSONA + "\n\n---\n\n"
                    "Describe ONLY what you can see in the image analysis result. "
                    "Do not reference session history, memory, or prior conversations. "
                    "Do not mention tool names or pipeline details. "
                    "The image analysis result is the sole source of truth."
                ))
                prompt_parts = [f"Image analysis result: {tool_result}"]
            else:
                format_llm.append(system(
                    PERSONA + "\n\n---\n\n"
                    "Format the tool result into a natural response. "
                    "Do not mention tool names or pipeline details."
                ))
                prompt_parts = [f"Request: {intent}"]
                if tool_result:
                    prompt_parts.append(f"Tool result: {tool_result}")
            format_llm.append(user("\n".join(prompt_parts)))

            _fast_response = format_llm.sample()
            response = _fast_response.content
            fast_usage = getattr(_fast_response, 'usage', None)
            format_llm = None
            slog.info(f">> [FAST] Response:\n{response}")
            if on_step_update:
                await on_step_update(response, type="stream", turn_id=0)

            # Append to request-local LLM for memory consolidation
            if llm is not None:
                llm.append(assistant(response))
            return response, fast_usage

        except Exception as e:
            slog.warning(f">> [FAST] Failed ({e}). Escalating to DELIBERATE.")
            if on_step_update:
                await on_step_update("Thinking more carefully...", type="status")

            # Build failure context for DELIBERATE — tell it what was attempted,
            # what result was obtained (if any), and why it failed.
            # This prevents DELIBERATE from blindly repeating the same approach.
            failure_parts = [
                "[FAST_EXECUTION_FAILED]",
                f"Tool attempted: {tool_name}",
                f"Args: {args}",
                f"Error: {e}",
            ]
            if tool_result is not None:
                # FAST got a result but something downstream failed (e.g. format_llm)
                # Give DELIBERATE the raw data so it doesn't re-fetch
                failure_parts.append(
                    f"Partial result obtained before failure:\n{str(tool_result)[:1000]}"
                )
            failure_parts.append(
                "Reason through the failure and do not repeat the same thing. "
                "Use the partial result if available. "
                "Reason through an alternative if not."
            )
            failure_note = "\n".join(failure_parts)

            if llm is not None:
                llm.append(assistant(failure_note))

            final_answer, deliberate_usage_list = await self.run_task(on_step_update=on_step_update, llm=llm)
            return final_answer, deliberate_usage_list

    async def _run_chat(self, llm, on_step_update=None) -> tuple:
        """
        CHAT_EXECUTION: direct conversational response with no tool loop.
        Streams tokens straight to the UI. Used when Interpreter returns tool=None
        and requires_planning=False — e.g. greetings, follow-up questions, opinions.
        """
        slog.info(">> [CHAT] Direct conversational response.")
        raw = ""
        sent_len = 0
        last_response_obj = None
        for response_obj, chunk in llm.stream():
            last_response_obj = response_obj
            token = chunk.content
            if not token:
                continue
            raw += token
            if on_step_update:
                await on_step_update(token, type="stream", turn_id=1)
                sent_len += len(token)
                await asyncio.sleep(0.01)
        response = raw.strip()
        chat_usage = getattr(last_response_obj, 'usage', None) if last_response_obj else None
        slog.info(f">> [CHAT] Response:\n{response}")
        llm.append(assistant(response))
        return response, chat_usage

    async def _execute_fast_tool(self, tool_name: str, args: dict) -> str:
        """Delegates to unified tool executor."""
        return await execute_fast(
            tool_name, args, self.tool_registry, self.mcp_client
        )

    # def run(self, direct_input=None, image_data=None) -> str:
        """
        Main Loop: Now Powered by Ears 👂 (Async Wrapper Version)
        """
        # --- MODE A: DIRECT INPUT (Used by API/CLI arguments) ---
        if direct_input:
            final_prompt = direct_input
            if image_data:
                print("🖼️ Image received from Interface. Processing...")
                try:
                    import uuid as _uuid_mod
                    if "," in image_data:
                        image_data = image_data.split(",")[1]

                    image_path = f"temp_image_{_uuid_mod.uuid4().hex[:8]}.png"

                    with open(image_path, "wb") as f:
                        f.write(base64.b64decode(image_data))
                        
                    abs_path = os.path.abspath(image_path)
                    final_prompt = f"{direct_input} \n\n[SYSTEM: An image has been uploaded and saved at '{abs_path}'. If the user asks about it, use the 'vision' tool to analyze this file.]"
                    
                    print(f"   Saved to: {image_path}")
                except Exception as e:
                    print(f"   ❌ Failed to save image: {e}")
            
            # ⚠️ FIX: Wrap the async call in asyncio.run()
            try:
                # If there's already an event loop running (unlikely in simple CLI, but possible), use it
                loop = asyncio.get_event_loop()
                if loop.is_running():
                     # This happens if you call run() from inside another async function (bad practice, but safety net)
                     response = loop.run_until_complete(self.process_request(final_prompt))
                else:
                     response = asyncio.run(self.process_request(final_prompt))
            except RuntimeError:
                # Fallback for complex environments (like Jupyter or some servers)
                response = asyncio.run(self.process_request(final_prompt))

            # Optional: Speak locally
            # speak(response)
            
            return response

        # --- MODE B: CLI / VOICE LOOP (Terminal) ---
        else:
            print("🎤 Voice Mode Active. (Say 'Clara' to trigger, or Ctrl+C to type)")
            
            while True:
                try:
                    # 1. Listen (Blocking - this is fine to stay sync)
                    # user_input = listen_local()
                    
                    if user_input:
                        # 2. Wake Word Check
                        if "clara" not in user_input.lower():
                            print(f"   [Ignored] Heard: '{user_input}' (No wake word)")
                            continue
                        
                        print(f"✅ Wake Word Detected: '{user_input}'")
                        
                        # 3. Process (⚠️ FIX: Wrap in asyncio.run)
                        response = asyncio.run(self.process_request(user_input))
                        
                        # 4. Speak

                        # speak(response)
                        
                except KeyboardInterrupt:
                    print("\n\n⌨️ MANUAL OVERRIDE ENGAGED")
                    try:
                        manual_input = input("   Enter command for CLARA: ")
                        if not manual_input.strip():
                            print("   (Cancelled)")
                            continue
                            
                        # Manual Input Processing (⚠️ FIX: Wrap in asyncio.run)
                        response = asyncio.run(self.process_request(manual_input))
                        # speak(response)
                        print("🎤 Returning to Voice Mode...\n")
                        
                    except KeyboardInterrupt:
                        print("\n👋 System Shutdown.")
                        break
                
                
            
    async def run_task(self, on_step_update=None, llm=None):
        if llm is None:
            llm = self.llm
        max_turns = 8
        llm.append(user(f"[SYSTEM MODE: TASK] [Turn 1/{max_turns}] You are in agent task mode."))
        turn_count = 0
        deliberate_usage_list = []
        last_response_text = ""

        while turn_count < max_turns:
            turn_count += 1
            slog.info(f"[Loop {turn_count}] Thinking (Streaming)...")

            raw_content = ""
            answer_sent_len = 0
            last_response_obj = None

            # 1. Open the Live Pipe (Native xai_sdk syntax)
            for response_obj, chunk in llm.stream():
                last_response_obj = response_obj
                token = chunk.content
                if not token:
                    continue

                raw_content += token

                # STATE C: The user-facing answer.
                if "Final Answer:" in raw_content:
                    start_idx = raw_content.find("Final Answer:") + 13
                    current_answer = raw_content[start_idx:]

                    # Only send the NEW characters
                    new_chars = current_answer[answer_sent_len:]
                    if new_chars and on_step_update:
                        await on_step_update(new_chars, type="stream", turn_id=turn_count)
                        answer_sent_len += len(new_chars)
                        await asyncio.sleep(0.02)

                # STATE B: The code.
                elif "Action:" in raw_content:
                    pass

                # STATE A: The internal monologue.
                elif "Thought:" in raw_content:
                    start_idx = raw_content.find("Thought:") + 8
                    current_thought = raw_content[start_idx:].strip()
                    clean_thought = re.split(r'\n?(?:Action|Final Answer|Glint):?', current_thought)[0].strip()

                    if current_thought and on_step_update:
                        await on_step_update(clean_thought, type="thought", turn_id=turn_count)

            # Capture usage from this turn's stream
            if last_response_obj:
                turn_usage = getattr(last_response_obj, 'usage', None)
                if turn_usage:
                    deliberate_usage_list.append(turn_usage)

            await asyncio.sleep(0.05)  # Small delay to simulate thinking time and allow UI to update

            inline_hallucination = False
            if "Glint:" in raw_content and "Action:" not in raw_content:
                # Bare fabricated Glint — no Action at all
                slog.warning("   [System] Hallucinated Glint detected (no Action) — correcting.")
                response_text = raw_content.split("Glint:")[0].strip()
                llm.append(assistant(response_text))
                llm.append(user(
                    "System: You generated a Glint without calling a tool. "
                    "Glints can ONLY come from actual tool execution. "
                    "If you need information, call the tool using Action: [...]. "
                    "Do not simulate or assume tool results. Continue with a valid Action."
                ))
                continue
            elif "Glint:" in raw_content and "Action:" in raw_content:
                # Inline fabrication — model wrote Action then immediately invented the Glint
                # in the same token stream without waiting for system execution.
                # Strip everything from the first Glint: onward, keep only the real Action.
                slog.warning("   [System] Inline hallucination detected (Action + fabricated Glint in same turn) — stripping fabricated Glint.")
                response_text = raw_content.split("Glint:")[0].strip()
                inline_hallucination = True
            else:
                response_text = raw_content.strip()

            last_response_text = response_text
            slog.info(f"Clara (Task turn {turn_count}):\n{response_text}")
            llm.append(assistant(response_text))

            if inline_hallucination:
                llm.append(user(
                    "System: You generated a Glint before the system executed your Action. "
                    "Glints are ONLY produced by the system after real tool execution — never by you. "
                    "The fabricated Glint was discarded. Your Action is being executed now — wait for the real Glint."
                ))

            if "Final Answer:" in response_text:
                final = response_text.split("Final Answer:")[-1].strip()
                slog.info(f">> [DELIBERATE] Final Answer:\n{final}")
                return final, deliberate_usage_list

            # Safety net: detect off-format turns — no Thought, no Action, no Final Answer.
            # This happens when the model dumps prose directly after an observation
            # instead of following the ReAct format. Treat as implicit final answer
            # rather than looping on empty turns.
            has_format_markers = (
                "Thought:" in response_text
                or "Action:" in response_text
                or "Final Answer:" in response_text
            )
            if not has_format_markers and response_text.strip():
                slog.warning(
                    f"   [Loop] Off-format turn {turn_count} — no Thought/Action/Final Answer. "
                    f"Treating as implicit Final Answer."
                )
                slog.info(f">> [DELIBERATE] Final Answer (implicit):\n{response_text}")
                return response_text, deliberate_usage_list

            # Parse all actions
            actions = self.parse_actions(response_text)

            if actions:
                # Separate valid actions from failed extractions
                valid_actions = [a for a in actions if a.get("tool")]
                failed_actions = [a for a in actions if not a.get("tool")]

                # Log failed extractions to glints
                glints = []
                for f in failed_actions:
                    msg = f"System: Action extraction failed — {f.get('error', 'unknown reason')}. Skipped."
                    glints.append(msg)
                    slog.warning(f"   [Parser] Action extraction failed: {msg}")

                # Execute all valid actions in parallel
                async def execute_tool(action: dict) -> str:
                    tool_name  = action["tool"]
                    tool_input = action["query"]
                    slog.info(
                        f"   -> Tool: {tool_name} ({tool_input[:60]}...)"
                        if len(tool_input) > 60
                        else f"   -> Tool: {tool_name} ({tool_input})"
                    )
                    return await execute_deliberate(
                        tool_name,
                        tool_input,
                        self.tool_registry,
                        self.mcp_client,
                        encode_fn=self._encode,
                    )

                # Run all valid tools concurrently
                if valid_actions:
                    results = await asyncio.gather(*[execute_tool(a) for a in valid_actions])
                    for action, result in zip(valid_actions, results):
                        glint = f"Glint from {action['tool']}[{action['query']}]: {result}"
                        glints.append(glint)
                        slog.info(f"   -> Glint: {glint[:120]}...")
                        slog.debug(f">> [Glint] {action['tool']}:\n{result}")

                # Feed all Glints back as a single combined message
                combined_glints = "\n".join(glints)
                llm.append(user(_turn_message(turn_count, max_turns, combined_glints)))

            else:
                llm.append(user(_turn_message(turn_count, max_turns, "System: No valid Action found. Please continue.")))

        # Fallback: exhausted all turns without a Final Answer.
        # Should be rare after final-turn wrap-up — model was told to conclude on last turn.
        slog.warning("[DELIBERATE] Turn budget exhausted without Final Answer.")
        if last_response_text:
            slog.info(f">> [DELIBERATE] Final Answer (turn limit — last content):\n{last_response_text}")
            return last_response_text, deliberate_usage_list
        return "Task incomplete — turn limit reached.", deliberate_usage_list