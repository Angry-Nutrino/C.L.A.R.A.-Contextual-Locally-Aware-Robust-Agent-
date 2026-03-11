"""
CLARA Gatekeeper Eval Script
Isolated simulation of MiniLM + Phi-3 Mini pipeline only.
Stops early on 100% pass rate. Hard cap: 20 iterations.
"""

# ── Unicode fix ───────────────────────────────────────────────────────────────
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── Standard library ──────────────────────────────────────────────────────────
import re
import json
import heapq
from datetime import datetime

# ── ML / embeddings ───────────────────────────────────────────────────────────
import torch
from sentence_transformers import SentenceTransformer

# ── Phi-3 Mini via native Ollama client (chat API — applies proper chat template) ─
import ollama as ollama_client

# ── (Optional) .env loading if model names / paths are stored there ───────────
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — mirror exactly what Clara_Agent uses
# ─────────────────────────────────────────────────────────────────────────────
MINIML_MODEL   = "all-MiniLM-L6-v2"
PHI3_MODEL     = "phi3:mini"
MAX_ITERATIONS = 1

# Tool metadata — loaded from tool_descriptions.json exactly as agent.py uses it
TOOL_NAMES = ["web_search", "python_repl", "consult_archive", "date_time", "vision_tool"]
TOOL_META  = {
    "web_search":      "find prices news and information on the internet",
    "python_repl":     "calculate math and run code",
    "consult_archive": "look up resume CV and knowledge base",
    "date_time":       "get current date and time right now",
    "vision_tool":     "analyze uploaded image or picture",
}

# Sub-descriptions used to build tool embeddings (must match tool_descriptions.json)
TOOL_SUB_DESCRIPTIONS = {
    "web_search": [
        "what is the price of gold",
        "stock price of Apple AAPL or Samsung",
        "latest news about OpenAI or any topic",
        "what is bitcoin trading at",
        "who won the cricket match or game"
    ],
    "python_repl": [
        "calculate the square root of 1445",
        "convert 45 USD to INR",
        "plot a bar chart of numbers",
        "what is 25 percent of 89000",
        "solve math equation"
    ],
    "consult_archive": [
        "find the phone number in my resume",
        "what skills are listed in my CV",
        "look up project details from the knowledge base",
        "retrieve contact details from documents",
        "search archived records"
    ],
    "date_time": [
        "what is today's date and time",
        "what time is it right now"
    ],
    "vision_tool": [
        "analyze the image I just uploaded",
        "what is in this picture",
        "describe the contents of this photo",
        "identify objects in this image",
        "tell me what you see in this screenshot"
    ],
}

VALID_TOOLS   = set(TOOL_NAMES) | {"NONE"}
VALID_INTENTS = {"TASK", "CHAT"}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL INIT
# ─────────────────────────────────────────────────────────────────────────────
print("Loading MiniLM...")
miniLM = SentenceTransformer(MINIML_MODEL)

print("Pre-computing tool embeddings...")
tool_emb = []
for name in TOOL_NAMES:
    descs = TOOL_SUB_DESCRIPTIONS[name]
    embs  = miniLM.encode(descs, convert_to_tensor=True)
    tool_emb.append(embs)

print(f"Phi-3 Mini will be called via ollama.chat() ({PHI3_MODEL})...")
print("Models ready.\n")

# ─────────────────────────────────────────────────────────────────────────────
# GATEKEEPER SIMULATION  (MiniLM + Phi-3 only — no Grok, no memory, no boost)
# ─────────────────────────────────────────────────────────────────────────────
def run_gatekeeper(query: str) -> tuple:
    """
    Mirrors agent.py gatekeeper() steps 1-5 exactly.
    Returns (raw_phi3_response, gatekeeper_prompt_used).
    """
    # Step 1 — MiniLM cosine similarity
    q_emb = miniLM.encode(query, convert_to_tensor=True)
    tool_scores = {}
    for i, embs in enumerate(tool_emb):
        cos_sims = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), embs)
        tool_scores[TOOL_NAMES[i]] = cos_sims.max().item()

    # Step 2 — Top-2 via heapq
    top2        = heapq.nlargest(2, tool_scores.items(), key=lambda x: x[1])
    tool1_name, tool1_score = top2[0]
    tool2_name, tool2_score = top2[1] if len(top2) > 1 else ("NONE", 0.0)
    tool1_micro_desc = TOOL_META[tool1_name]
    tool2_micro_desc = TOOL_META.get(tool2_name, "")

    # Step 3 — Build prompt (EXACT copy from agent.py)
    gatekeeper_prompt = (
        f"User Query: '{query}'\n\n"
        "Available Tools & Confidence Scores:\n"
        f"1. {tool1_name} ({tool1_micro_desc}): {tool1_score:.3f}\n"
        f"2. {tool2_name} ({tool2_micro_desc}): {tool2_score:.3f}\n\n"
        "ROUTING RULES:\n"
        "1. High Confidence (Score > 0.70): Select Tool 1. Set intent to TASK.\n"
        "2. Mid Confidence (Score 0.36 - 0.70): Select the most relevant tool. Set intent to TASK.\n"
        "3. Low Confidence (Score < 0.36): You MUST select NONE and set intent to CHAT. Do not select any tool.\n"
        "4. context_needed: TRUE if the query refers to personal facts, previous tasks, or past conversation. Otherwise FALSE.\n\n"
        "TOOL QUERY RULES:\n"
        "- tool_query MUST contain a specific query string when a tool is selected.\n"
        "- For python_repl: tool_query is the exact code or math expression to evaluate.\n"
        "- For web_search: tool_query is the specific search string.\n"
        "- For date_time: tool_query is 'now'.\n"
        "- For vision_tool or consult_archive: tool_query is the specific question.\n"
        "- For NONE: tool_query is empty.\n\n"
        "OUTPUT FORMAT:\n"
        "Output ONLY the XML block below:\n"
        "<analysis>\n"
        "  <tool>Selected tool name or NONE</tool>\n"
        "  <tool_query>Specific query for the tool. Empty only if NONE.</tool_query>\n"
        "  <intent>TASK or CHAT</intent>\n"
        "  <context_needed>TRUE or FALSE</context_needed>\n"
        "</analysis>"
    )

    # Step 4 — Invoke Phi-3 via chat API with stop sequence to prevent runaway generation
    response = ollama_client.chat(
        model=PHI3_MODEL,
        messages=[
            {"role": "system", "content": "You are an XML routing assistant. Output ONLY the XML block. No code fences, no explanations, no extra text."},
            {"role": "user", "content": gatekeeper_prompt},
        ],
        options={"temperature": 0, "num_ctx": 2048, "stop": ["</analysis>"]},
    )
    # Append the stop token since Ollama excludes it from the output
    raw_response = response["message"]["content"] + "</analysis>"
    return raw_response, gatekeeper_prompt


# ─────────────────────────────────────────────────────────────────────────────
# PARSER  (mirrors agent.py step 5 exactly)
# ─────────────────────────────────────────────────────────────────────────────
def parse_gatekeeper_output(raw: str) -> dict | None:
    """
    Returns parsed dict on success, None on failure.
    Mirrors the regex parser in agent.py step 5.
    """
    try:
        match = re.search(r"<analysis>.*?</analysis>", raw, re.DOTALL)
        if not match:
            return None
        xml = match.group(0)

        t_match  = re.search(r"<tool>(.*?)</tool>",                     xml)
        tq_match = re.search(r"<tool_query>(.*?)</tool_query>",         xml)
        i_match  = re.search(r"<intent>(.*?)</intent>",                 xml)
        c_match  = re.search(r"<context_needed>(TRUE|FALSE)", xml)

        if not all([t_match, tq_match, i_match, c_match]):
            return None

        return {
            "tool":           t_match.group(1).strip(),
            "tool_query":     tq_match.group(1).strip(),
            "intent":         i_match.group(1).strip().upper(),
            "context_needed": c_match.group(1).strip().upper() == "TRUE",
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def validate(parsed: dict | None, raw: str | None) -> tuple[bool, str]:
    """Returns (passed, failure_reason)."""
    if raw is None:
        return False, "No output from Phi-3"
    if parsed is None:
        return False, "XML parse failed — <analysis> block missing or malformed tags"
    if parsed["tool"] not in VALID_TOOLS:
        return False, f"Invalid <tool> value: '{parsed['tool']}'"
    if parsed["intent"] not in VALID_INTENTS:
        return False, f"Invalid <intent> value: '{parsed['intent']}'"
    if parsed["tool"] not in ("NONE", "date_time") and not parsed["tool_query"]:
        return False, "<tool_query> is empty but tool is not NONE or date_time"
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# TEST QUERIES  (covers all 5 tools + NONE/CHAT + edge cases)
# ─────────────────────────────────────────────────────────────────────────────
TEST_QUERIES = [
    # web_search
    "What is the current price of Bitcoin?",
    "Search for the latest AI news.",
    "Search for RTX 4090 price in India.",
    "What's the latest news about GPT-5?",
    "Find me the weather in Mumbai today.",
    # python_repl
    "Calculate the compound interest on 50000 at 8% for 5 years.",
    "Write Python code to reverse a string.",
    "Run a Python script to sort this list: [3,1,4,1,5,9].",
    "What is 2 + 2?",
    "Convert 100 USD to INR using current rates.",
    # date_time
    "What time is it right now?",
    "What is today's date?",
    "Tell me the current time in Tokyo.",
    # vision_tool
    "Look at this image and tell me what you see.",
    "Analyze the image I uploaded.",
    "Look at this screenshot and describe the UI.",
    # consult_archive
    "What did I tell you about my project last time?",
    "Check the archive for my previous notes on CLARA.",
    "What were we discussing in our last session?",
    # CHAT / NONE edge case
    "Hello, how are you today?",
]


# ─────────────────────────────────────────────────────────────────────────────
# EVAL LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_eval_iteration(iteration: int) -> dict:
    results = []
    for i, query in enumerate(TEST_QUERIES):
        raw, parsed, passed, reason = None, None, False, ""
        try:
            raw, _         = run_gatekeeper(query)
            parsed         = parse_gatekeeper_output(raw)
            passed, reason = validate(parsed, raw)
        except Exception as e:
            reason = f"Exception: {e}"

        results.append({
            "query_id":       i,
            "query":          query,
            "raw_output":     raw,
            "parsed":         parsed,
            "passed":         passed,
            "failure_reason": reason if not passed else None,
        })

    pass_count = sum(1 for r in results if r["passed"])
    return {
        "iteration":  iteration,
        "timestamp":  datetime.now().isoformat(),
        "pass_count": pass_count,
        "total":      len(TEST_QUERIES),
        "pass_rate":  pass_count / len(TEST_QUERIES),
        "results":    results,
    }


def print_iteration_summary(ev: dict):
    print(f"\n{'='*55}")
    print(f"ITERATION {ev['iteration']} | {ev['pass_rate']*100:.1f}%  ({ev['pass_count']}/{ev['total']})")
    print(f"{'='*55}")
    for r in ev["results"]:
        if not r["passed"]:
            print(f"  FAIL [{r['query_id']:02d}] {r['query'][:55]}")
            print(f"         > {r['failure_reason']}")
            if r["raw_output"]:
                safe_output = r['raw_output'][:120].encode('utf-8', errors='replace').decode('utf-8')
                print(f"         > Raw: {safe_output}...")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    all_results = []
    print(f"CLARA Gatekeeper Eval | {len(TEST_QUERIES)} queries | max {MAX_ITERATIONS} iterations\n")

    for iteration in range(1, MAX_ITERATIONS + 1):
        result = run_eval_iteration(iteration)
        all_results.append(result)
        print_iteration_summary(result)

        if result["pass_rate"] == 1.0:
            print(f"\n✅ 100% pass rate on iteration {iteration}. Done.")
            break

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("FINAL REPORT")
    print(f"{'='*55}")
    print(f"Iterations run: {len(all_results)} / {MAX_ITERATIONS}")
    for r in all_results:
        print(f"  Iter {r['iteration']:02d}: {r['pass_rate']*100:.1f}%  ({r['pass_count']}/{r['total']})")

    best = max(all_results, key=lambda x: x["pass_rate"])
    print(f"\nBest: {best['pass_rate']*100:.1f}% on iteration {best['iteration']}")

    with open("gatekeeper_eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Results saved → gatekeeper_eval_results.json")


if __name__ == "__main__":
    main()
