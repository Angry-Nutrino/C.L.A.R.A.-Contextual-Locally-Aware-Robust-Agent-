import json
from xai_sdk.chat import user, system
from .session_logger import slog

# Tool argument schemas — tells the Interpreter what args each tool needs.
# Filesystem tools are NO LONGER listed here. They are discovered via tool_search
# or appear in [DISCOVERED_TOOLS] context block injected before this call.
TOOL_ARG_SCHEMAS = {
    # Core always-available native tools
    "web_search":        {"query": "string — search query"},
    "python_repl":       {"code": "string — python code to execute"},
    "date_time":         {},
    "vision_tool":       {"path": "string — absolute path to image file",
                          "question": "string — what to ask about the image",
                          "paths": "list[string] — optional: multiple image paths"},
    "consult_archive":   {"query": "string — question for the archive"},
    "query_task_status": {"keyword": "string — keyword from task goal"},

    # Dynamic tool discovery
    "tool_search":       {"query": "string — semantic description of capability needed"},
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

web_search — use ONLY when the answer requires live or post-training data:
- Current prices, rates, scores, weather, news, stock values
- Events or releases after mid-2025 (training cutoff)
- Anything explicitly marked "latest", "current", "today", "now"

Do NOT use web_search for:
- Stable factual knowledge (capitals, historical facts, scientific concepts, definitions)
- Well-established technical knowledge (language features, algorithms, best practices)
- Questions that can be answered from training data with high confidence
- Explanations, opinions, creative tasks, reasoning, analysis

When in doubt: if the answer could have been in a textbook 5 years ago, do not search.

Filesystem path rules:
- If Alkama explicitly provides the full path → use it as-is, confidence can be high
- If you are inferring or constructing a path → set requires_planning=true, confidence ≤ 0.70
- NEVER assume directory names, filenames, or casing

[DISCOVERED_TOOLS] block:
- When a [DISCOVERED_TOOLS] block is present in context, use those tool names
  and schemas EXACTLY as provided — including arg names and types
- Discovered tools take precedence for filesystem, process, and search operations
- If a filesystem/process/search task is needed and NO [DISCOVERED_TOOLS] block
  is present, assign tool="tool_search" with a semantic query describing what
  capability is needed (e.g. "read file from disk", "list directory", "run shell command")

Personal memory rules:
- For questions about people Alkama has mentioned, past conversations, things
  you discussed previously, or anything phrased as "do you remember X" or
  "did I tell you about X" → answer from [MEMORY_CONTEXT_BLOCK] directly.
  Set tool=null, requires_planning=false.
- Do NOT use consult_archive for personal memory lookups.
  consult_archive searches indexed documentation (CLAUDE.md, ROADMAP.md,
  resume) — it does not contain conversation history.
- If the memory context has no relevant information, say so directly.
  Do not search for it — it either exists in memory or it doesn't.

Follow-up resolution:
If the query is short (under 6 words), uses demonstrative references without
specifying what they refer to ("these", "that", "it", "the same", "in india",
"over there"), or clearly lacks a subject noun — treat it as a continuation of
the most recent exchange in [RELEVANT PAST INTERACTIONS] before interpreting
as a new topic.
Examples:
- "In india clara" after a watch price query → interpret as asking for India
  prices of the watches just discussed, not a new topic about India
- "What's the major difference in these versions?" with no prior context about
  versions → check recent interactions first; if Porsche was just discussed,
  "these versions" means Porsche variants
- "How much is it there?" → refer to the most recent item and location discussed
Do NOT default to CLARA architecture or project topics when the recent
conversation was about something else entirely.
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
        _interp_response = llm.sample()
        raw = _interp_response.content
        interp_usage = getattr(_interp_response, 'usage', None)
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
                return FALLBACK, None

        slog.info(
            f">> [Interpreter] Parsed → tool={result['tool']} | "
            f"confidence={result['confidence']:.2f} | "
            f"uncertainty={result['uncertainty']:.2f} | "
            f"requires_planning={result['requires_planning']} | "
            f"intent={result['intent'][:80]}"
        )
        return result, interp_usage

    except Exception as e:
        slog.error(f">> [Interpreter] Failed: {e}. Using fallback.")
        return FALLBACK, None
