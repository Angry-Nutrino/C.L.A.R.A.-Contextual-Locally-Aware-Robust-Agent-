# Changelog

## [Unreleased] — 2026-03-11

### Gatekeeper — Phi-3 Mini Reliability Fix (`core_logic/agent.py`)

**Problem:** Phi-3 Mini was producing 0-15% valid XML output via the LangChain `Ollama`
wrapper, which uses Ollama's `/api/generate` (text-completion) endpoint. Phi-3 Mini is a
chat-tuned model and requires the chat template to behave correctly.

**Root causes identified via iterative eval (`gatekeeper_test.py`, 8 iterations):**
1. LangChain `Ollama` wrapper bypasses the model's chat template — model drifts into hallucination
2. No stop sequence — model continues generating past `</analysis>` into irrelevant text
3. `<context_needed>` closing tag mangled by Phi-3 into prose echoed from the prompt

**Changes:**

| Location | Before | After |
|----------|--------|-------|
| Import | `from langchain_community.llms import Ollama` | `import ollama as ollama_client` |
| Cold-start | `Ollama(...); helper_llm.invoke("hi")` | `ollama_client.chat(model=..., messages=[{"role":"user","content":"hi"}])` |
| Invoke | `helper_llm.invoke(gatekeeper_prompt)` | `ollama_client.chat(..., options={"stop": ["</analysis>"]})` |
| System msg | None | `{"role": "system", "content": "Output ONLY the XML block..."}` |
| Prompt | Basic routing rules | Added TOOL QUERY RULES section; strengthened Low Confidence rule |
| Parser `context_needed` | `(.*?)</context_needed>` | `(TRUE|FALSE)` — value-anchored, no closing tag needed |

**Result:** 15% to 100% pass rate across all 20 eval queries (8 outer iterations).

**Eval artefacts:** `gatekeeper_test.py`, `gatekeeper_eval_results.json`
