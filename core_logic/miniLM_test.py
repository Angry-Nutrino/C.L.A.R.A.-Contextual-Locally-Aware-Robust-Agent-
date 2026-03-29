"""
Phase 1 — MiniLM NONE Tool Test
Tests MiniLM scoring only (no Phi3, no Grok, no memory).
Loads tool_descriptions.json directly so any changes are instantly reflected.
Hard exit at 100% pass rate. Hard cap: 20 iterations.
"""

# ── Unicode fix ────────────────────────────────────────────────────────────────
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── Standard library ───────────────────────────────────────────────────────────
import json
import os

# ── ML / embeddings ────────────────────────────────────────────────────────────
import torch
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MINIML_MODEL       = "all-MiniLM-L6-v2"
MAX_ITERATIONS     = 20
TOOL_DESC_PATH     = os.path.join(os.path.dirname(__file__), "tool_descriptions.json")

# ─────────────────────────────────────────────────────────────────────────────
# TEST QUERIES  (60 total: 20 mainstream, 20 indirect, 20 tricky)
# ─────────────────────────────────────────────────────────────────────────────
TEST_QUERIES = [
    # ── 20 MAINSTREAM (clear, direct intent) ─────────────────────────────────
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
    ("Tell me a joke about Python.",                            "NONE"),
    ("What do you think my height is?",                         "NONE"),
    ("Can you search your memory for what I told you?",         "consult_archive"),
    ("I want to know the time, is it late?",                    "date_time"),
    ("Run through the image with me.",                          "vision_tool"),
    ("Calculate my mood today.",                                "NONE"),
    ("What is the meaning of life?",                            "NONE"),
    ("Search deep inside yourself.",                            "NONE"),
    ("Is the archive of human knowledge vast?",                 "NONE"),
    ("Show me how to calculate BMI.",                           "python_repl"),
    ("I uploaded an image of my resume, what skills do I have?","vision_tool"),
    ("What's 2 + 2? I'm curious.",                              "python_repl"),
    ("Any news on whether the market opened today?",            "web_search"),
    ("I just wanted to say you explained that really well.",    "NONE"),
    ("What would happen if I ran this Python code mentally?",   "NONE"),
    ("Tell me the date in your own words.",                     "date_time"),
    ("Look at what I've been working on — here's the image.",   "vision_tool"),
    ("Find me, philosophically speaking.",                      "NONE"),
    ("My CV is in the archive, right?",                         "consult_archive"),
    ("Nice chat. What's Bitcoin at though?",                    "web_search"),
]

# ─────────────────────────────────────────────────────────────────────────────
# LOAD TOOL DESCRIPTIONS
# ─────────────────────────────────────────────────────────────────────────────
def load_tool_descriptions():
    with open(TOOL_DESC_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# BUILD EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────
def build_embeddings(miniLM, tools):
    tool_names = [t["name"] for t in tools]
    tool_emb   = []
    for t in tools:
        embs = miniLM.encode(t["sub_descriptions"], convert_to_tensor=True)
        tool_emb.append(embs)
    return tool_names, tool_emb

# ─────────────────────────────────────────────────────────────────────────────
# SCORE QUERY
# ─────────────────────────────────────────────────────────────────────────────
def score_query(query, miniLM, tool_names, tool_emb):
    q_emb = miniLM.encode(query, convert_to_tensor=True)
    scores = {}
    for i, embs in enumerate(tool_emb):
        cos_sims = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), embs)
        scores[tool_names[i]] = cos_sims.max().item()
    top_tool = max(scores, key=scores.get)
    return top_tool, scores

# ─────────────────────────────────────────────────────────────────────────────
# RUN EVAL
# ─────────────────────────────────────────────────────────────────────────────
def run_eval(miniLM, tool_names, tool_emb):
    results = []
    for query, expected in TEST_QUERIES:
        top_tool, scores = score_query(query, miniLM, tool_names, tool_emb)
        passed = (top_tool == expected)
        results.append({
            "query":    query,
            "expected": expected,
            "got":      top_tool,
            "scores":   scores,
            "passed":   passed,
        })
    return results

def print_results(results, iteration):
    pass_count = sum(1 for r in results if r["passed"])
    total      = len(results)
    print(f"\n{'='*65}")
    print(f"ITERATION {iteration} | {pass_count}/{total} ({pass_count/total*100:.1f}%)")
    print(f"{'='*65}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            sorted_scores = sorted(r["scores"].items(), key=lambda x: x[1], reverse=True)
            score_str = "  |  ".join(f"{k}: {v:.3f}" for k, v in sorted_scores[:3])
            print(f"  {status} | expected={r['expected']:<16} got={r['got']:<16} | {r['query'][:50]}")
            print(f"         Scores: {score_str}")
    return pass_count, total

# ─────────────────────────────────────────────────────────────────────────────
# PRINT FINAL SCORE TABLE
# ─────────────────────────────────────────────────────────────────────────────
def print_full_score_table(results):
    print(f"\n{'='*65}")
    print("FULL SCORE TABLE (top-2 tools per query)")
    print(f"{'='*65}")
    print(f"{'#':<4} {'Expected':<18} {'Top-1':<18} {'Score':<8} {'Top-2':<18} {'Score':<8} Query")
    print("-"*120)
    for i, r in enumerate(results):
        sorted_scores = sorted(r["scores"].items(), key=lambda x: x[1], reverse=True)
        t1, s1 = sorted_scores[0]
        t2, s2 = sorted_scores[1] if len(sorted_scores) > 1 else ("—", 0.0)
        flag = "" if r["passed"] else " <<FAIL"
        print(f"{i:<4} {r['expected']:<18} {t1:<18} {s1:<8.3f} {t2:<18} {s2:<8.3f} {r['query'][:55]}{flag}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading MiniLM...")
    miniLM = SentenceTransformer(MINIML_MODEL)
    print("MiniLM loaded.\n")

    for iteration in range(1, MAX_ITERATIONS + 1):
        tools = load_tool_descriptions()  # reload each iteration so JSON changes are picked up
        tool_names, tool_emb = build_embeddings(miniLM, tools)

        results = run_eval(miniLM, tool_names, tool_emb)
        pass_count, total = print_results(results, iteration)

        if pass_count == total:
            print(f"\n100% pass rate achieved on iteration {iteration}.")

            # Print final NONE entry
            none_entry = next((t for t in tools if t["name"] == "NONE"), None)
            print("\n--- Final NONE entry from tool_descriptions.json ---")
            print(json.dumps(none_entry, indent=2))

            # Print full score table
            print_full_score_table(results)

            print("\nPHASE 1 COMPLETE — Waiting for Alkama's review and approval to proceed.")
            return

        print(f"\nNot at 100% yet. Iteration {iteration} complete. Edit tool_descriptions.json and re-run.")
        break  # Single run — user edits JSON between runs as per iteration guidance

    print(f"\nMax iterations ({MAX_ITERATIONS}) reached or stopped early.")


if __name__ == "__main__":
    main()
