# SMART CONTEXT RETRIEVAL — Implementation Brief
**Project:** AGENT_ZERO / CLARA
**Files touched:** `core_logic/crud.py`, `core_logic/agent.py`
**Rule:** Build and test in isolation first (`core_logic/smart_context_test.py`), then apply to agent. No commits. Alkama reviews manually.

---

## WHAT THIS DOES

Replaces the current flat last-10 episodic dump with a targeted retrieval:
- **Last 3** episodic entries (recency guarantee)
- **Top 2** episodic entries by semantic similarity to current query (MiniLM cosine sim)
- Merged and deduplicated — max 5 entries injected, potentially fewer after dedup
- Vault (permanent facts) always included regardless

This keeps context lean, relevant, and bounded.

---

## ARCHITECTURE

### Embedding management
`self.episodic_embeddings` lives on `Clara_Agent` as a list of tensors, parallel to `self.db.memory["episodic_log"]`.

- **Cold start (startup):** In `__init__`, after MiniLM loads, encode all existing episodic summaries in one pass to populate `self.episodic_embeddings`
- **Incremental (per session):** After `memorize_episode()` adds a new entry, encode only that new summary and append its tensor to `self.episodic_embeddings`
- MiniLM is already on `self.miniLM` — no new model needed

### Retrieval (per request)
In `gatekeeper()`, replace the `self.db.get_full_context()` call with `self.db.get_smart_context(query, self.miniLM, self.episodic_embeddings)`.

---

## PHASE 1 — Changes to `crud.py`

### Add `get_smart_context()` method

```python
def get_smart_context(self, query: str, miniLM, episodic_embeddings: list) -> str:
    """
    Smart retrieval: last 3 episodic entries + top 2 semantic hits.
    Vault always included. Deduplicates overlaps.
    """
    import torch

    profile   = self.memory.get("user_profile", {})
    project   = self.memory.get("project_state", {})
    long_term = self.memory.get("long_term", [])
    episodes  = self.memory.get("episodic_log", [])

    selected_indices = set()

    # 1. Last 3 by recency
    if episodes:
        last3 = list(range(max(0, len(episodes) - 3), len(episodes)))
        for idx in last3:
            selected_indices.add(idx)

    # 2. Top 2 semantic hits via MiniLM
    if episodes and episodic_embeddings and len(episodic_embeddings) == len(episodes):
        q_emb  = miniLM.encode(query, convert_to_tensor=True)
        all_embs = torch.stack(episodic_embeddings)  # (N, 384)
        cos_sims = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), all_embs)
        top2_indices = cos_sims.topk(min(2, len(episodes))).indices.tolist()
        for idx in top2_indices:
            selected_indices.add(idx)

    # 3. Build context string
    context = "--- MEMORY CONTEXT ---\n"

    # Identity
    context += f"USER: {profile.get('name', 'Unknown')} | ROLE: {profile.get('role', 'User')}\n"
    context += f"TECH STACK: {', '.join(profile.get('preferences', {}).get('tools', []))}\n"
    context += f"CURRENT PHASE: {project.get('current_phase', 'Unknown')}\n"

    # Vault
    if long_term:
        context += "\n[PERMANENT KNOWLEDGE VAULT]:\n"
        for fact in long_term:
            context += f"- {fact}\n"

    # Selected episodic entries (sorted chronologically)
    if selected_indices:
        context += "\n[RELEVANT PAST INTERACTIONS]:\n"
        for idx in sorted(selected_indices):
            ep = episodes[idx]
            context += f"- [{ep.get('timestamp', '')[:16]}] {ep.get('summary', '')}\n"

    context += "----------------------"
    return context
```

---

## PHASE 2 — Changes to `agent.py`

### 2A — Add `episodic_embeddings` cold start in `__init__`

After the line `self.tool_emb = self._build_tool_embeddings()`, add:

```python
self.episodic_embeddings = self._build_episodic_embeddings()
```

Add this method to `Clara_Agent`:

```python
def _build_episodic_embeddings(self) -> list:
    """
    Cold-start: encode all existing episodic summaries once at startup.
    Returns a list of tensors parallel to db.memory["episodic_log"].
    """
    episodes = self.db.memory.get("episodic_log", [])
    if not episodes:
        print("   [Memory] No episodic entries to embed at startup.")
        return []
    summaries = [ep.get("summary", "") for ep in episodes]
    print(f"   [Memory] Encoding {len(summaries)} episodic entries at startup...")
    embs = self.miniLM.encode(summaries, convert_to_tensor=True)
    return list(embs)  # list of (384,) tensors
```

### 2B — Incremental embedding update in `memorize_episode()`

After `self.db.add_episodic_log(summary)`, add:

```python
# Encode and append the new episodic embedding incrementally
new_emb = self.miniLM.encode(summary, convert_to_tensor=True)
self.episodic_embeddings.append(new_emb)
print(f"   [Memory] 📎 Episodic embedding updated ({len(self.episodic_embeddings)} total)")
```

### 2C — Replace context injection in `gatekeeper()`

Find:
```python
mem_context = self.db.get_full_context()
```

Replace with:
```python
mem_context = self.db.get_smart_context(final_prompt, self.miniLM, self.episodic_embeddings)
```

No other changes to `gatekeeper()`.

---

## PHASE 3 — Isolation Test (`core_logic/smart_context_test.py`)

Create this file. It does NOT run the agent. It loads the real `memory.json`, builds embeddings, runs test queries, and prints what gets retrieved so you can inspect it.

**Structure:**

```python
"""
Smart Context Retrieval — Isolation Test
Loads real memory.json, builds episodic embeddings, runs queries, prints retrieved context.
No agent, no Grok, no Phi3. Pure retrieval verification.
"""
import json
import torch
from sentence_transformers import SentenceTransformer

MEMORY_PATH = "core_logic/memory.json"
MINIML_MODEL = "all-MiniLM-L6-v2"

# Load memory
with open(MEMORY_PATH, "r") as f:
    memory = json.load(f)

episodes  = memory.get("episodic_log", [])
long_term = memory.get("long_term", [])

print(f"Loaded {len(episodes)} episodic entries, {len(long_term)} vault facts.\n")

# Load MiniLM and encode all episodes
miniLM = SentenceTransformer(MINIML_MODEL)
if episodes:
    summaries = [ep.get("summary", "") for ep in episodes]
    episodic_embeddings = list(miniLM.encode(summaries, convert_to_tensor=True))
    print(f"Encoded {len(episodic_embeddings)} episodic embeddings.\n")
else:
    episodic_embeddings = []
    print("No episodic entries to embed.\n")

# Import the new method directly from crud for testing
import sys, os
sys.path.insert(0, os.path.abspath("."))
from core_logic.crud import crud
db = crud()
db.memory = memory  # inject real memory directly

# Test queries
TEST_QUERIES = [
    "What did we work on last time?",
    "What is the current price of Bitcoin?",
    "Tell me about CLARA's architecture",
    "Hello, how are you?",
    "What was my preference for the gatekeeper fix?",
]

print("=" * 60)
print("SMART CONTEXT RETRIEVAL TEST")
print("=" * 60)

for query in TEST_QUERIES:
    print(f"\nQUERY: {query}")
    print("-" * 40)
    context = db.get_smart_context(query, miniLM, episodic_embeddings)
    print(context)
    print()
```

**What to check when reviewing output:**
- Vault facts are always present
- Last 3 entries appear for every query
- Semantic hits are plausibly relevant to the query
- No duplicate entries appear
- If episodic log is empty, it degrades gracefully (vault only)

**Pass criteria:** Output looks sensible for all 5 queries. This is a human review step, not automated pass/fail.

---

## ⛔ CHECKPOINT — PAUSE AFTER PHASE 3 TEST

After running `smart_context_test.py`:
1. Print the full retrieved context for all 5 test queries
2. **STOP. Do not apply changes to `agent.py`.**
3. Output: `"SMART CONTEXT TEST COMPLETE — Waiting for Alkama's review and approval to proceed."`
4. Wait for explicit instruction.

---

## PHASE 4 — Apply to `agent.py`

Only after checkpoint approval. Make the following changes:

**4A — Apply Phase 2A, 2B, 2C** as described above (cold start embeddings, incremental update, smart context call).

**4B — Remove `need_context` entirely from `gatekeeper()`:**
- Delete the line `need_context = self._needs_context(final_prompt)`
- Change the condition `if intent == "CHAT" or need_context:` to unconditional:
  ```python
  print(">> [Memory] Loading Soul from disk...")
  mem_context = self.db.get_smart_context(final_prompt, self.miniLM, self.episodic_embeddings)
  self.llm.append(assistant(f"[MEMORY_CONTEXT_BLOCK]\n{mem_context}\n[/MEMORY_CONTEXT_BLOCK]"))
  ```
- Remove `"need_context"` from the returned dict at the end of `gatekeeper()`

**4C — Remove dead code from `agent.py`:**
- Delete the `CONTEXT_TRIGGERS` list at the top of the file (module level)
- Delete the `_needs_context()` method from `Clara_Agent`

**4D — Cleanup: delete the test file:**
After all agent changes are verified to be correctly written, delete `core_logic/smart_context_test.py`. It has served its purpose and should not persist in the directory.

---

## FILE SUMMARY

| File | Action |
|---|---|
| `core_logic/crud.py` | Add `get_smart_context()` method |
| `core_logic/smart_context_test.py` | Create for testing, DELETE after Phase 4 |
| `core_logic/agent.py` | Apply Phase 2 + remove need_context + remove dead code |

---

## COMMIT PROTOCOL
Do NOT commit. Alkama handles all git operations manually.
