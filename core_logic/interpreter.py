import json
from xai_sdk.chat import user, system
from .session_logger import slog

# Tool argument schemas — tells the Interpreter what args each tool needs.
# Add new tools here as they are added to the system.
TOOL_ARG_SCHEMAS = {
    "web_search":        {"query": "string — search query"},
    "python_repl":       {"code": "string — python code to execute"},
    "date_time":         {},
    "vision_tool":       {"path": "string — absolute path to image file (jpg, png, gif, webp)",
                          "question": "string — what to ask about the image",
                          "paths": "list[string] — optional: multiple image paths for multi-image analysis"},
    "consult_archive":   {"query": "string — question for the archive"},
    "fs_read_file":      {"path": "string — absolute file path"},
    "fs_list_directory": {"path": "string — absolute directory path"},
    "fs_write_file":     {"path": "string — absolute file path",
                          "content": "string — content to write"},
    "fs_run_command":    {"command": "string — shell command to run"},
    "query_task_status": {"keyword": "string — keyword from task goal"},
}

INTERPRETER_SYSTEM_PROMPT = """
You are CLARA's Interpreter. Your job is to parse any input and produce
a structured intent JSON object. You output ONLY valid JSON, no other text.

Given an input (user message, system trigger description, or task goal),
output this exact schema:

{
  "intent": "brief description of what needs to be done",
  "tool": "tool_name or null",
  "args": {},
  "confidence": 0.0-1.0,
  "uncertainty": 0.0-1.0,
  "requires_planning": true/false
}

Rules:
- tool: name of the single best tool if one tool clearly suffices, else null
- args: tool-specific args matching the schema for the chosen tool. Empty if no tool.
- confidence: how confident you are in this interpretation (1.0 = certain)
- uncertainty: how ambiguous the input is (0.0 = crystal clear)
- requires_planning: true if multi-step reasoning is needed, false if one step suffices

Routing guidance:
- Single clear tool + clear args + no dependency chain → tool set, requires_planning=false
- Vague request, multiple steps, or tool outputs feed into next step → requires_planning=true
- Greetings, opinions, conversation → tool=null, requires_planning=false
- System triggers (health_check, memory_maintenance, etc.) → tool=null, requires_planning=false
  (these are handled by existing background workers, do not assign tools to them)
"""


async def interpret(
    content: str,
    source: str,
    context: str,
    client,
    task_context: dict = None,
) -> dict:
    """
    Interpret any input and return a structured intent dict.

    Args:
        content:      The raw input text (user message, task goal, trigger description)
        source:       "user" | "system" | "worker"
        context:      Relevant memory context string (from get_smart_context)
        client:       xai_sdk.Client instance
        task_context: Optional task context dict — may contain failure_summary for retries

    Returns dict with keys: intent, tool, args, confidence, uncertainty,
    requires_planning. Returns safe fallback on any failure.
    """
    FALLBACK = {
        "intent": content[:100],
        "tool": None,
        "args": {},
        "confidence": 0.5,
        "uncertainty": 0.5,
        "requires_planning": True,
    }

    try:
        llm = client.chat.create(model="grok-4-1-fast-non-reasoning")
        llm.append(system(INTERPRETER_SYSTEM_PROMPT))

        tool_schema_str = json.dumps(TOOL_ARG_SCHEMAS, indent=2)

        failure_note = ""
        if task_context and "failure_summary" in task_context:
            fs = task_context["failure_summary"]
            failure_note = (
                f"\nPREVIOUS ATTEMPT FAILED:\n"
                f"Reason: {fs.get('reason', '')}\n"
                f"Attempt: {fs.get('attempt', 1)}\n"
                f"Adjust your approach — avoid the previous failure pattern.\n"
            )

        prompt = (
            f"Source: {source}\n"
            f"Input: {content}\n"
            + failure_note +
            f"\nAvailable tools and their arg schemas:\n{tool_schema_str}\n\n"
            f"Relevant context:\n{context}\n\n"
            "Output ONLY the JSON object."
        )
        llm.append(user(prompt))
        raw = llm.sample().content
        slog.info(f">> [Interpreter] Raw output:\n{raw}")

        # Strip markdown fences if present
        clean = raw.strip().lstrip("\ufeff")
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
            clean = clean.strip()

        result = json.loads(clean)

        # Validate required keys
        for key in ("intent", "tool", "args", "confidence",
                    "uncertainty", "requires_planning"):
            if key not in result:
                slog.warning(
                    f">> [Interpreter] Missing key '{key}' — using fallback"
                )
                return FALLBACK

        slog.info(
            f">> [Interpreter] Parsed → tool={result['tool']} | "
            f"confidence={result['confidence']:.2f} | "
            f"uncertainty={result['uncertainty']:.2f} | "
            f"requires_planning={result['requires_planning']} | "
            f"intent={result['intent'][:80]}"
        )
        return result

    except Exception as e:
        slog.error(f">> [Interpreter] Failed: {e}. Using fallback.")
        return FALLBACK
