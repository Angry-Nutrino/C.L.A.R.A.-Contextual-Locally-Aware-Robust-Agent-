# GATEKEEPER FIX — Implementation Brief
**Project:** AGENT_ZERO / CLARA  
**Target file:** `core_logic/agent.py`  
**Scope:** Two isolated fixes to the `gatekeeper()` method  
**Rule:** Do NOT touch `agent.py` until both test phases pass 100%. All work happens in test files first.

---

## PROBLEM SUMMARY

### Problem 1 — False tool triggers on conversational queries
MiniLM never has NONE as a candidate, so Phi3 is always forced to choose between two real tools even on greetings or casual chat. It tries to reason its way to `intent=CHAT` but gets confused and selects a tool anyway, causing a spurious boost call that wastes tokens and latency.

**Fix:** Add a `NONE` tool entry to `tool_descriptions.json` with strong conversational sub_descriptions. MiniLM will naturally surface NONE in top-2 for conversational queries. Phi3 then has a real candidate to select.

### Problem 2 — `context_needed` unreliable from Phi3
Phi3 Mini (3.8B) cannot reliably infer whether a query references past sessions or personal data from a vague prompt rule. It misses TRUE cases too often.

**Fix:** Remove `context_needed` from Phi3's responsibility entirely. Implement it as a deterministic Python keyword scan before the Phi3 call. Also update the Phi3 prompt to remove this field from output format.

---

## PHASE 1 — MiniLM NONE Tool (Problem 1)

### Step 1.1 — Update `tool_descriptions.json`

Add the following entry to the JSON array in `core_logic/tool_descriptions.json`:

```json
{
  "name": "NONE",
  "description": "general conversation chat opinion greeting or explanation requiring no tool",
  "sub_descriptions": [
    "hello how are you doing today",
    "what do you think about this idea",
    "tell me a joke",
    "just talk to me",
    "what is your opinion on artificial intelligence",
    "can you explain what this concept means",
    "I wanted to say thank you",
    "you did a great job earlier",
    "what would you recommend I do",
    "that is interesting tell me more",
    "nice work on that",
    "I disagree with that approach",
    "what are your thoughts on this",
    "explain this to me simply",
    "can you help me understand something"
  ]
}
```

### Step 1.2 — Create `core_logic/miniLM_test.py`

Create a NEW file. Do NOT modify `core_logic/test.py`.

This file tests MiniLM scoring only — no Phi3, no Grok, no memory. It loads `tool_descriptions.json` directly (same as `agent.py` does) so any change to that file is instantly reflected.

**Structure:**
- Load MiniLM + build embeddings from `tool_descriptions.json` (NONE included)
- Define 60 test queries with expected tool labels (see below)
- For each query: compute max-similarity scores across all tools including NONE, take the top scorer
- Print pass/fail per query, overall pass rate
- Hard exit at 100% pass rate, hard cap at 20 iterations of description tuning

**60 Test Queries with Expected Labels:**

```python
TEST_QUERIES = [
    # ── 20 MAINSTREAM (clear, direct intent) ─────────────────────────────────
    # Expected: the obvious tool
    ("What is the current price of Bitcoin?",                   "web_search"),
    ("Search for the latest AI news today.",                    "web_search"),
    ("What is the stock price of Tesla right now?",             "web_search"),
    ("Who won the IPL match yesterday?",                        "web_search"),
    ("Find the latest news about GPT-5.",                       "web_search"),
    ("Calculate the square root of 1764.",                      "python_repl"),
    ("What is 18 percent of 75000?",                            "python_repl"),
    ("Convert 200 USD to INR.",                                 "python_repl"),
    ("Write Python code to reverse a string.",                  "python_repl"),
    ("Solve: what is 345 multiplied by 78?",                    "python_repl"),
    ("What time is it right now?",                              "date_time"),
    ("What is today's date?",                                   "date_time"),
    ("Analyze the image I just uploaded.",                      "vision_tool"),
    ("What is in this picture?",                                "vision_tool"),
    ("Describe the contents of this screenshot.",               "vision_tool"),
    ("What skills are listed in my CV?",                        "consult_archive"),
    ("Find the phone number in my resume.",                     "consult_archive"),
    ("Look up project details from the knowledge base.",        "consult_archive"),
    ("Hello, how are you today?",                               "NONE"),
    ("What do you think about this idea?",                      "NONE"),

    # ── 20 INDIRECT (intent is implied, not stated directly) ──────────────────
    ("I need to know the going rate for gold.",                 "web_search"),
    ("Is Ethereum doing well today?",                           "web_search"),
    ("Any updates on the OpenAI drama?",                        "web_search"),
    ("How is the Sensex performing?",                           "web_search"),
    ("Did India win the cricket test?",                         "web_search"),
    ("I have a list [5, 3, 8, 1] and need it sorted.",          "python_repl"),
    ("Help me figure out the compound interest on 10000 at 7%.", "python_repl"),
    ("Can you run some math for me — 456 divided by 12?",       "python_repl"),
    ("I need a script to count words in a string.",             "python_repl"),
    ("Figure out what percentage 340 is of 2000.",              "python_repl"),
    ("What does the clock say?",                                "date_time"),
    ("Is it morning or evening right now?",                     "date_time"),
    ("Take a look at this photo and describe it.",              "vision_tool"),
    ("I uploaded something, can you check it out?",             "vision_tool"),
    ("Pull up my resume and see what languages I know.",        "consult_archive"),
    ("Is there anything in the archive about my contact info?", "consult_archive"),
    ("What were my project notes about?",                       "consult_archive"),
    ("Can you just chat with me for a bit?",                    "NONE"),
    ("What are your thoughts on machine learning?",             "NONE"),
    ("That was a really helpful answer, thank you.",            "NONE"),

    # ── 20 TRICKY (ambiguous, misleading, or adversarial) ────────────────────
    # These are designed to confuse MiniLM — verify NONE doesn't bleed into tools
    # and tools don't bleed into NONE
    ("Tell me a joke about Python.",                            "NONE"),   # Python = language, not code
    ("What do you think my height is?",                         "NONE"),   # question but no tool can answer
    ("Can you search your memory for what I told you?",         "consult_archive"),  # 'search' but about memory
    ("I want to know the time, is it late?",                    "date_time"),        # indirect date_time
    ("Run through the image with me.",                          "vision_tool"),      # 'run' != python_repl
    ("Calculate my mood today.",                                "NONE"),   # 'calculate' but not math
    ("What is the meaning of life?",                            "NONE"),   # philosophical, no tool
    ("Search deep inside yourself.",                            "NONE"),   # 'search' but not web_search
    ("Is the archive of human knowledge vast?",                 "NONE"),   # 'archive' but philosophical
    ("Show me how to calculate BMI.",                           "python_repl"),  # 'show me how' = still code
    ("I uploaded an image of my resume, what skills do I have?","vision_tool"),  # resume + image = vision
    ("What's 2 + 2? I'm curious.",                              "python_repl"),  # casual framing, still math
    ("Any news on whether the market opened today?",            "web_search"),   # indirect market query
    ("I just wanted to say you explained that really well.",    "NONE"),
    ("What would happen if I ran this Python code mentally?",   "NONE"),   # 'Python code' but no exec needed
    ("Tell me the date in your own words.",                     "date_time"),    # unusual framing
    ("Look at what I've been working on — here's the image.",   "vision_tool"),
    ("Find me, philosophically speaking.",                      "NONE"),   # 'find' but not a search
    ("My CV is in the archive, right?",                         "consult_archive"),
    ("Nice chat. What's Bitcoin at though?",                    "web_search"),   # starts conversational, ends task
]
```

**Pass Criteria:** Top-scoring tool (by max cosine similarity) must match expected label. 100% required before proceeding to Phase 2.

**Iteration Guidance:**
- If NONE bleeds into tool queries: tighten the NONE sub_descriptions (remove any that share vocabulary with tool queries)
- If tool queries score NONE as top: add more specific sub_descriptions to the correct tool entry
- Only modify `tool_descriptions.json` between iterations, never the test file's expected labels
- Print full score breakdown on failures so it's clear which tool is competing

---

## ⛔ CHECKPOINT 1 — PAUSE AFTER PHASE 1

After Phase 1 reaches 100% pass rate on `miniLM_test.py`:
1. Print the final passing `tool_descriptions.json` NONE entry to console in full
2. Print the full score table for all 60 queries showing top-2 tools and scores
3. **STOP. Do not proceed to Phase 2.**
4. Output this exact message: `"PHASE 1 COMPLETE — Waiting for Alkama's review and approval to proceed."`
5. Wait for explicit instruction before continuing.

---

## PHASE 2 — Gatekeeper Full Fix (Problem 2)

Only begin this phase after Phase 1 is at 100%.

### Step 2.1 — Update `gatekeeper_test.py`

Modify the existing `gatekeeper_test.py` in place. Changes required:

**A) Add NONE to all inline tool registries:**

Add NONE to `TOOL_NAMES`, `TOOL_META`, and `TOOL_SUB_DESCRIPTIONS` — copy the exact description and sub_descriptions from the final working `tool_descriptions.json` after Phase 1 iteration.

```python
TOOL_NAMES = ["web_search", "python_repl", "consult_archive", "date_time", "vision_tool", "NONE"]

TOOL_META["NONE"] = "general conversation chat opinion greeting or explanation requiring no tool"

TOOL_SUB_DESCRIPTIONS["NONE"] = [
    # copy final working sub_descriptions from tool_descriptions.json here
]
```

**B) Add Python keyword scan for `context_needed` BEFORE the Phi3 call:**

```python
CONTEXT_TRIGGERS = [
    "my project", "my name", "my preference", "my resume", "my file",
    "last time", "previously", "you said", "remember when", "as we discussed",
    "earlier you", "before you", "you mentioned", "we talked", "you told me",
    "do you remember", "from before", "last session", "you remember",
    "what did we", "what did you", "our conversation", "you suggested",
]

def needs_context(query: str) -> bool:
    q = query.lower()
    return any(trigger in q for trigger in CONTEXT_TRIGGERS)
```

Call this before invoking Phi3. Store result as `context_needed_result = needs_context(query)`.

**C) Update the Phi3 gatekeeper prompt — remove `context_needed` entirely:**

Replace the existing `gatekeeper_prompt` string. Remove:
- Rule 4 about `context_needed`
- `<context_needed>` from the OUTPUT FORMAT XML block

Updated prompt should output only:
```xml
<analysis>
  <tool>Selected tool name or NONE</tool>
  <tool_query>Specific query for the tool. Empty only if NONE.</tool_query>
  <intent>TASK or CHAT</intent>
</analysis>
```

Also update the ROUTING RULES section. Replace the current 4 rules with these cleaner versions:

```
ROUTING RULES:
1. High Confidence (Score > 0.70): Select Tool 1. Set intent to TASK.
2. Mid Confidence (Score 0.36 - 0.70): Select the most relevant tool. Set intent to TASK.
3. Low Confidence (Score < 0.36): Select NONE. Set intent to CHAT.
4. If Tool 1 is NONE: Always select NONE. Set intent to CHAT. Do not override.
```

Rule 4 is critical — it prevents Phi3 from ignoring NONE even when MiniLM correctly surfaces it as the top candidate.

**D) Update `parse_gatekeeper_output()` — remove `context_needed` parsing:**

Remove the `c_match` line. Return dict with only `tool`, `tool_query`, `intent`.

**E) Update `validate()` — remove `context_needed` validation.**

**F) Expand TEST_QUERIES to 60 with expected outputs:**

Each entry: `(query, expected_tool, expected_intent, expected_context_needed)`

```python
TEST_QUERIES = [
    # ── 20 MAINSTREAM ────────────────────────────────────────────────────────
    ("What is the current price of Bitcoin?",                   "web_search",     "TASK",  False),
    ("Search for the latest AI news today.",                    "web_search",     "TASK",  False),
    ("What is the stock price of Tesla right now?",             "web_search",     "TASK",  False),
    ("Who won the IPL match yesterday?",                        "web_search",     "TASK",  False),
    ("Find the latest news about GPT-5.",                       "web_search",     "TASK",  False),
    ("Calculate the square root of 1764.",                      "python_repl",    "TASK",  False),
    ("What is 18 percent of 75000?",                            "python_repl",    "TASK",  False),
    ("Convert 200 USD to INR.",                                 "python_repl",    "TASK",  False),
    ("Write Python code to reverse a string.",                  "python_repl",    "TASK",  False),
    ("What is 345 multiplied by 78?",                           "python_repl",    "TASK",  False),
    ("What time is it right now?",                              "date_time",      "TASK",  False),
    ("What is today's date?",                                   "date_time",      "TASK",  False),
    ("Analyze the image I just uploaded.",                      "vision_tool",    "TASK",  False),
    ("What is in this picture?",                                "vision_tool",    "TASK",  False),
    ("What skills are listed in my CV?",                        "consult_archive","TASK",  False),
    ("Find the phone number in my resume.",                     "consult_archive","TASK",  False),
    ("Look up project details from the knowledge base.",        "consult_archive","TASK",  False),
    ("Hello, how are you today?",                               "NONE",           "CHAT",  False),
    ("What do you think about this idea?",                      "NONE",           "CHAT",  False),
    ("Tell me a joke.",                                         "NONE",           "CHAT",  False),

    # ── 20 INDIRECT ──────────────────────────────────────────────────────────
    ("I need to know the going rate for gold.",                 "web_search",     "TASK",  False),
    ("Is Ethereum doing well today?",                           "web_search",     "TASK",  False),
    ("Any updates on the OpenAI drama?",                        "web_search",     "TASK",  False),
    ("How is the Sensex performing?",                           "web_search",     "TASK",  False),
    ("Did India win the cricket test?",                         "web_search",     "TASK",  False),
    ("I have a list [5,3,8,1] and need it sorted.",             "python_repl",    "TASK",  False),
    ("Help me figure out compound interest on 10000 at 7%.",    "python_repl",    "TASK",  False),
    ("Can you run some math — 456 divided by 12?",              "python_repl",    "TASK",  False),
    ("I need a script to count words in a string.",             "python_repl",    "TASK",  False),
    ("What does the clock say?",                                "date_time",      "TASK",  False),
    ("Take a look at this photo and describe it.",              "vision_tool",    "TASK",  False),
    ("Pull up my resume and see what languages I know.",        "consult_archive","TASK",  False),
    ("Is there anything in the archive about my contact info?", "consult_archive","TASK",  False),
    ("Can you just chat with me for a bit?",                    "NONE",           "CHAT",  False),
    ("What are your thoughts on machine learning?",             "NONE",           "CHAT",  False),
    ("That was really helpful, thank you.",                     "NONE",           "CHAT",  False),
    ("What did we discuss last session about CLARA?",           "NONE",           "CHAT",  True),   # context_needed
    ("Do you remember what I told you about my project?",       "NONE",           "CHAT",  True),   # context_needed
    ("Previously you suggested a fix, what was it?",            "NONE",           "CHAT",  True),   # context_needed
    ("As we discussed before, what was my preference?",         "NONE",           "CHAT",  True),   # context_needed

    # ── 20 TRICKY ─────────────────────────────────────────────────────────────
    ("Tell me a joke about Python.",                            "NONE",           "CHAT",  False),
    ("What do you think my height is?",                         "NONE",           "CHAT",  False),
    ("Can you search your memory for what I told you?",         "consult_archive","TASK",  True),
    ("I want to know the time, is it late?",                    "date_time",      "TASK",  False),
    ("Run through the image with me.",                          "vision_tool",    "TASK",  False),
    ("Calculate my mood today.",                                "NONE",           "CHAT",  False),
    ("What is the meaning of life?",                            "NONE",           "CHAT",  False),
    ("Search deep inside yourself.",                            "NONE",           "CHAT",  False),
    ("Is the archive of human knowledge vast?",                 "NONE",           "CHAT",  False),
    ("Show me how to calculate BMI.",                           "python_repl",    "TASK",  False),
    ("I uploaded an image of my resume, what skills do I have?","vision_tool",    "TASK",  False),
    ("What's 2 + 2? I'm curious.",                              "python_repl",    "TASK",  False),
    ("Any news on whether the market opened today?",            "web_search",     "TASK",  False),
    ("I just wanted to say you explained that really well.",    "NONE",           "CHAT",  False),
    ("What would happen if I ran this Python code mentally?",   "NONE",           "CHAT",  False),
    ("Tell me the date in your own words.",                     "date_time",      "TASK",  False),
    ("Look at what I've been working on — here's the image.",   "vision_tool",    "TASK",  False),
    ("Find me, philosophically speaking.",                      "NONE",           "CHAT",  False),
    ("My CV is in the archive, right?",                         "consult_archive","TASK",  False),
    ("Nice chat. What's Bitcoin at though?",                    "web_search",     "TASK",  False),
]
```

**G) Update `validate()` to check all three fields: `tool`, `intent`, `context_needed`:**

- `tool` must match expected
- `intent` must match expected
- `context_needed` (from Python keyword scan) must match expected boolean

Since `context_needed` is now computed in Python (not from Phi3), validation of it does not require a Phi3 call — validate it directly from `needs_context(query)` result.

**Pass Criteria:** 100% on all 60 queries across tool, intent, and context_needed fields.

**Iteration Guidance:**
- Tool/intent failures: adjust the Phi3 prompt routing rules or the NONE description
- context_needed failures: add or remove triggers from `CONTEXT_TRIGGERS` list
- MAX_ITERATIONS = 20, stop early at 100%
- Print per-failure details: which field failed and what was returned vs expected

---

## ⛔ CHECKPOINT 2 — PAUSE AFTER PHASE 2

After Phase 2 reaches 100% pass rate on `gatekeeper_test.py`:
1. Print the final Phi3 prompt used (the full `gatekeeper_prompt` string)
2. Print the final `CONTEXT_TRIGGERS` list
3. Print the full 60-query results table showing tool / intent / context_needed pass/fail
4. **STOP. Do not proceed to Phase 3.**
5. Output this exact message: `"PHASE 2 COMPLETE — Waiting for Alkama's review and approval to proceed."`
6. Wait for explicit instruction before continuing.

---

## PHASE 3 — Apply to `agent.py`

Only begin after Phase 2 is at 100%.

### Changes to `agent.py` `gatekeeper()` method:

**1. Add CONTEXT_TRIGGERS constant** at class or module level (same list from Phase 2).

**2. Add `_needs_context()` method** to `Clara_Agent`:
```python
def _needs_context(self, query: str) -> bool:
    q = query.lower()
    return any(trigger in q for trigger in CONTEXT_TRIGGERS)
```

**3. In `gatekeeper()`**, before the Phi3 call, compute:
```python
need_context = self._needs_context(final_prompt)
```

**4. Update the `gatekeeper_prompt` string** — remove `context_needed` rule and XML field (use exact prompt from Phase 2 that achieved 100%).

**5. Update `parse_gatekeeper_output`-equivalent inline parsing** — remove `c_match` and `need_context` assignment from the XML parser block. `need_context` is already set from step 3.

**6. `tool_descriptions.json`** is already updated from Phase 1 — `_build_tool_embeddings()` loads it dynamically, so NONE is automatically included. No code change needed there.

**7. `TOOL_NAMES` in `_build_tool_embeddings()`** — verify `"NONE"` appears in `self.tool_names` after loading (it will, since it's in the JSON). No change needed.

**8. In the boost block** — the existing guard `if tool_name != "NONE"` already handles this correctly. No change needed.

---

## FILE SUMMARY

| File | Action |
|---|---|
| `core_logic/tool_descriptions.json` | Add NONE entry (Phase 1) |
| `core_logic/miniLM_test.py` | Create new (Phase 1 testing) |
| `gatekeeper_test.py` | Modify in place (Phase 2 testing) |
| `core_logic/agent.py` | Modify after Phase 2 passes (Phase 3) |
| `core_logic/test.py` | Do NOT touch |

---

## COMMIT PROTOCOL
Do NOT commit anything. Alkama handles all git operations manually.
