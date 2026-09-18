import sys
from io import StringIO
from datetime import datetime
from tavily import TavilyClient
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from .session_logger import slog
import os
import re

# Explicit path, not a bare load_dotenv(). agent.py:9 imports this module ~174 lines BEFORE its own
# load_dotenv at agent.py:183, and a bare call resolves nothing from the repo root (find_dotenv() -> '').
# So every module-level os.getenv below — _COMPUTE_ONLY among them — read the DEFAULT, and setting the
# flag in core_logic/.env silently did nothing. Found 2026-09-07 while arming PYTHON_REPL_COMPUTE_ONLY.
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


RAG_ENGINE= None
current_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH=os.path.join(current_dir, "knowledge_base")

# Pre-loading rag for faster inference.
slog.info("   [Archive] Pre-loading RAG Engine for instant access...")
_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # RAG on CPU — VRAM reserved for agent MiniLM + Phi3
)

if os.path.exists(DB_PATH):
    RAG_ENGINE = FAISS.load_local(
        DB_PATH,
        _embeddings,
        allow_dangerous_deserialization=True
    )
    slog.info("   [Archive] RAG Engine is Hot.")
else:
    RAG_ENGINE = None
    slog.info("   [Archive] DB Not found. RAG will be disabled.")


def reload_rag_engine() -> bool:
    """
    Reload the FAISS index from disk into the global RAG_ENGINE.
    Called after a successful rag_rebuild background task.
    Returns True on success, False on failure.
    """
    global RAG_ENGINE
    try:
        if os.path.exists(DB_PATH):
            RAG_ENGINE = FAISS.load_local(
                DB_PATH,
                _embeddings,
                allow_dangerous_deserialization=True,
            )
            slog.info("   [Archive] RAG Engine reloaded from disk.")
            return True
        else:
            slog.warning("   [Archive] DB not found during reload.")
            return False
    except Exception as e:
        slog.error(f"   [Archive] Reload failed: {e}")
        return False


def get_archive_context(q_emb_cpu, query: str, threshold: float = 0.35) -> str:
    """
    Relevance-gated archive lookup using a pre-computed MiniLM embedding.
    q_emb_cpu: CPU-side torch tensor (384-dim) from agent._encode().
    threshold: minimum cosine similarity to inject (0.35 = reasonably relevant).
    Returns a formatted [ARCHIVE CONTEXT] string, or "" if nothing relevant found.

    Uses the FAISS index directly with a numpy vector — avoids the langchain
    similarity_search() path which requires re-encoding the query string.
    """
    global RAG_ENGINE
    if RAG_ENGINE is None:
        return ""

    try:
        import numpy as np
        # Convert torch tensor → numpy float32 array for FAISS
        q_np = q_emb_cpu.numpy().astype("float32").reshape(1, -1)

        # FAISS inner product search (index is L2-normalised → equivalent to cosine)
        scores, indices = RAG_ENGINE.index.search(q_np, k=3)

        # scores[0] are inner-product scores; for normalised vectors, range is [-1, 1]
        best_score = float(scores[0][0]) if len(scores[0]) > 0 else 0.0
        if best_score < threshold:
            return ""

        # Retrieve the actual documents for the top indices
        chunks = []
        id_map = RAG_ENGINE.index_to_docstore_id
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1 or float(score) < threshold:
                continue
            doc_id = id_map.get(int(idx))
            if doc_id is None:
                continue
            doc = RAG_ENGINE.docstore._dict.get(doc_id)
            if doc:
                chunks.append(doc.page_content.strip())

        if not chunks:
            return ""

        block = "\n[ARCHIVE CONTEXT]:\n"
        for i, chunk in enumerate(chunks):
            block += f"[{i+1}] {chunk}\n"
        block += "[END ARCHIVE]\n"
        return block

    except Exception as e:
        from .session_logger import slog
        slog.warning(f"   [Archive] Context injection failed: {e}")
        return ""


# ── G38 OPTION (b) — COMPUTE-ONLY python_repl (BRIEF_60). DORMANT BY DEFAULT. ────────
# The structural fix for the gate's largest hole. `python_repl` is not in MUTATING_TOOLS, so
# admissibility.gate() short-circuits to ALLOW with no envelope and no ledger entry, and the code
# then runs with FULL builtins (exec() auto-injects them when the globals dict has no
# __builtins__). It can therefore write, delete and execute around the gate. Proven live
# 2026-08-10: with write_file unregistered the agent FELL BACK to python_repl and completed a
# write, leaving no receipt — the bypass is the automatic degradation path, not just an
# adversary's route.
#
# Option (a) was to classify the code body and hold the risky ones. `code_intent` measured that:
# false positives fell from 56% to roughly 21% after the 08-18 receiver fix, so (a) is viable.
# (b) is still better and it is what this implements. If the namespace cannot reach the
# filesystem, there is nothing to classify and nothing to falsely hold. A control that cannot be
# bypassed beats a control that usually catches the bypass.
#
# OFF BY DEFAULT (`PYTHON_REPL_COMPUTE_ONLY=0`). Flipping it changes how Clara answers a large
# share of drill questions — she currently reaches for os.walk here rather than the DC tools, even
# though Rule 17 already tells her not to. So while the flag is off this still LOGS what it would
# have blocked, and that shadow signal is what should decide the flip. Same doctrine as the gate
# itself: measure before arming.
# BRIEF 62, 2026-09-09: the single owner of security-relevant config. Importing it also performs THE
# load of core_logic/.env by explicit path, which is what makes these reads correct regardless of
# import order. Dual import because this module is also executed directly for its self-test.
try:
    from . import policy_config as _policy
except ImportError:
    import policy_config as _policy
# Resolved through policy_config so an unrecognised value is a recorded fault instead of a silent
# fall back to OFF.
_COMPUTE_ONLY = _policy.resolve("PYTHON_REPL_COMPUTE_ONLY") == "on"

# ── G38 / BRIEF_60 — OS-ENFORCED ISOLATION (added 2026-09-18). DORMANT: default "off". ──
# `_COMPUTE_ONLY` above rebuilds the exec namespace. That is useful and it is NOT a boundary, which
# BRIEF_60 section 5.2 retracted in writing and tests/test_compute_only_containment.py re-measures
# every run: the real open() stays reachable through __subclasses__(). This flag runs the code in a
# separate process under a Windows restricted token instead, so the escape can succeed and the
# kernel refuses the write anyway. Measured 2026-09-18: gadget reaches open(), write -> PermissionError.
#   off    - in-process, unchanged
#   shadow - in-process result is returned, the sandbox ALSO runs, divergence is logged. Costs a
#            process spawn per call; that is the price of deciding the flip on evidence.
#   on     - sandbox only
_ISOLATION = _policy.resolve("PYTHON_REPL_ISOLATION")

# Pure-computation modules. Anything that can touch the filesystem, spawn a process, open a
# socket, or import arbitrary code is absent by construction rather than blacklisted, so a module
# nobody thought of is denied rather than allowed.
_ALLOWED_MODULES = frozenset({
    "math", "cmath", "statistics", "decimal", "fractions", "numbers", "random",
    "json", "re", "string", "textwrap", "unicodedata", "difflib",
    "datetime", "calendar", "time", "zoneinfo",
    "itertools", "functools", "operator", "collections", "heapq", "bisect", "array",
    "copy", "enum", "dataclasses", "typing", "abc",
    "hashlib", "hmac", "base64", "binascii", "struct", "uuid", "secrets",
})

# Builtins that reach the filesystem, execute code, or hand back an escape route to the real
# builtins. `open` is handled separately so it can carry a redirect message.
_DENIED_BUILTINS = frozenset({
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "exit", "quit", "help", "globals", "vars", "memoryview",
})


class _ComputeOnlyViolation(Exception):
    """Raised inside the sandbox so the message reaches the model as tool output."""


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in _ALLOWED_MODULES:
        raise _ComputeOnlyViolation(
            f"python_repl is compute-only, so the '{root}' module is not available here. "
            f"For files use read_file, write_file, list_directory or start_search. "
            f"For processes use start_process. Those tools are recorded; this one is not."
        )
    return __import__(name, globals, locals, fromlist, level)


def _denied_open(*a, **kw):
    raise _ComputeOnlyViolation(
        "python_repl is compute-only, so it cannot open files. "
        "Use read_file to read one and write_file to write one, so the action leaves a record."
    )


def _build_exec_namespace(capture_print, utf8_open):
    """The exec globals. Restricted only when the flag is on; otherwise byte-for-byte the
    previous behaviour, so a dormant flag cannot change how anything runs."""
    if not _COMPUTE_ONLY:
        return {"open": utf8_open, "print": capture_print}
    import builtins as _bi
    safe = {n: getattr(_bi, n) for n in dir(_bi)
            if not n.startswith("_") and n not in _DENIED_BUILTINS}
    safe["__import__"] = _restricted_import
    safe["__name__"] = "__main__"
    return {"__builtins__": safe, "open": _denied_open, "print": capture_print}


def _would_compute_only_block(code: str) -> str:
    """Shadow signal for the flag itself: what WOULD have been refused if it were on.
    Deliberately crude and string-based — it never runs and never blocks, it only tells us how
    often flipping the flag would bite before anyone flips it."""
    hits = []
    for mod in ("os", "sys", "subprocess", "shutil", "pathlib", "socket", "importlib",
                "ctypes", "glob", "tempfile", "io", "pickle", "sqlite3", "requests",
                "urllib", "http", "multiprocessing", "threading", "signal", "platform"):
        if re.search(rf"\b(?:import\s+{mod}\b|from\s+{mod}[\s.])", code):
            hits.append(f"import:{mod}")
    # 🔴 READ vs WRITE, added 2026-09-18. The old line recorded only that open() appeared.
    # Reviewing 206 logged calls on 09-18: 164 (79.6%) reached for the filesystem and open() alone
    # appeared 131 times, and NOT ONE of those records told us whether it was a read or a write.
    # That matters because compute-only blocks BOTH and the OS sandbox blocks only WRITES, so the
    # accumulated shadow data could only bracket the answer (0 to 164) instead of giving it.
    # Recording the mode turns the next day of real traffic into the actual number.
    for m in re.finditer(r"\bopen\s*\(([^)]*)\)", code):
        args = m.group(1)
        mode = re.search(r"""['"]([rwaxb+t]{1,4})['"]""", args)
        mode = mode.group(1) if mode else "r"        # open() defaults to read
        hits.append("open:w" if any(c in mode for c in "wax+") else "open:r")
    if "open:w" not in hits and "open:r" not in hits and re.search(r"\bopen\s*\(", code):
        hits.append("open:?")                        # multi-line call the single-line regex missed
    for b in ("exec", "eval", "compile", "__import__"):
        if re.search(rf"\b{b}\s*\(", code):
            hits.append(f"builtin:{b}")
    return ",".join(sorted(set(hits)))


def _isolated_result_to_output(r: dict) -> str:
    """Flatten a sandbox result into the single string the tool contract returns."""
    if r.get("timed_out"):
        return "Error: %s" % r.get("error", "timed out")
    if r.get("kind") == "sandbox":
        # The boundary itself failed. Say so as a sandbox fault rather than letting it read as a
        # failure of the user's code, which would send the model off rewriting correct code.
        return "Error: %s" % r.get("error", "sandbox failure")
    out = r.get("output") or ""
    if r.get("error"):
        return (out + ("\n" if out and not out.endswith("\n") else "") +
                "Error: %s" % r["error"]).strip()
    if not out.strip():
        return "Code executed successfully with no output. Check your format and checkcode for return values."
    return out


def _run_isolated(code: str):
    """Run in the restricted-token sandbox. Returns the output string, or None if the boundary
    could not be established. None is never silently converted into a pass: the caller logs a
    DOWNGRADE, because a sandbox that failed to start is not a sandbox that allowed the action."""
    try:
        try:
            from . import repl_sandbox as _sbx
        except ImportError:
            import repl_sandbox as _sbx
        r = _sbx.run(code)
        slog.info("   [ReplIsolation] ran in sandbox exit=%s timed_out=%s kind=%s"
                  % (r.get("exit_code"), r.get("timed_out"), r.get("kind") or "ok"))
        return _isolated_result_to_output(r)
    except Exception as e:
        slog.info("   [ReplIsolation] ⚠️ DOWNGRADE - sandbox unavailable (%s: %s). "
                  "Execution fell back IN-PROCESS and this call had no OS boundary."
                  % (type(e).__name__, e))
        return None


def run_python_code(code: str, use_case: str = "read") -> str:
    # ── G38 / BRIEF_60 — SHADOW OBSERVATION ONLY. Changes NOTHING about execution. ──
    # python_repl is not in MUTATING_TOOLS, so admissibility.gate() short-circuits and arbitrary
    # Python runs with no envelope, no verdict and no ledger entry. Proven live 2026-08-10: with
    # write_file unregistered, the agent completed a write through this tool and left no receipt.
    #
    # Before restricting anything we need the FALSE-POSITIVE RATE on real traffic — over-blocking is
    # what gets a control switched off, which costs more than the hole it closes. So this block only
    # derives and logs. Grep `[CodeIntent]` in logs/ to read the accumulated evidence.
    #
    # `use_case` is a CLAIM, never a grant (the agent must not self-authorize). Its value is that a
    # declared/derived disagreement becomes a DETECTABLE finding rather than a silent bypass.
    # Review date: 2026-08-18 (see My_Schedule/REMINDERS.md).
    try:
        from .code_intent import derive as _derive, agrees as _agrees
        _op, _ev = _derive(code)
        _ok = _agrees(use_case, _op)
        slog.info(f"   [CodeIntent] declared={use_case} derived={_op} agrees={_ok} "
                  f"imports={_ev.get('imports')} calls={_ev.get('calls')} "
                  f"modes={_ev.get('open_modes')} opaque={_ev.get('opaque')}")
        if not _ok:
            slog.info(f"   [CodeIntent] ⚠️ MISMATCH would be REVIEW under enforce — "
                      f"declared {use_case!r} but code derives {_op!r}")
        # Shadow signal for G38 option (b): what a compute-only namespace would have refused.
        _blocked_by = _would_compute_only_block(code)
        if _blocked_by:
            slog.info(f"   [ComputeOnly] would_block={_blocked_by} "
                      f"(flag={'ON' if _COMPUTE_ONLY else 'OFF'})")
    except Exception as _e:                      # observation must never break execution
        slog.info(f"   [CodeIntent] derive failed (non-fatal): {type(_e).__name__}: {_e}")

    # ── G38 — OS-ENFORCED ISOLATION. The only branch here that changes execution. ──
    if _ISOLATION == "on":
        _iso = _run_isolated(code)
        if _iso is not None:
            return _iso
        # Falling through means the boundary could not be established. _run_isolated has already
        # logged it as a DOWNGRADE. Continuing in-process is the deliberate choice: this flag is
        # about closing an unrecorded write path, not about refusing to work when pywin32 is
        # missing. The log line is what makes the downgrade auditable instead of invisible.

    # Output capture is SCOPED via a print-override in the exec namespace — the old
    # implementation swapped the process-global sys.stdout, so two concurrent
    # python_repl calls (parallel Actions run via asyncio.gather in worker threads)
    # could steal/interleave each other's output (Brief 36 D-15). Code that writes
    # via sys.stdout.write directly bypasses capture — a known, accepted trade-off
    # (model code overwhelmingly uses print()).
    buf = StringIO()

    def _capture_print(*args, sep=" ", end="\n", **kwargs):
        buf.write(sep.join(str(a) for a in args) + end)

    try:
        # Inject a UTF-8-defaulting open() so model code that omits encoding= does
        # not hit Windows cp1252 charmap errors on UTF-8 files (e.g. memory.json
        # contains multibyte chars). This removes the TRIGGER of the Q08-class
        # cascade: on Windows bare open() defaults to cp1252 → 'charmap' decode
        # error → FAST escalates → the model embeds fragile multi-line code in a
        # JSON Action → malformed JSON. A working open() short-circuits all of it.
        import builtins as _bi

        def _utf8_open(*a, **kw):
            mode = a[1] if len(a) > 1 else kw.get("mode", "r")
            if "b" not in mode and "encoding" not in kw:
                kw["encoding"] = "utf-8"
            return _bi.open(*a, **kw)

        # A single namespace dict also serves as BOTH globals and locals. With bare
        # exec(code), top-level assignments land in the function's locals while
        # comprehensions resolve free variables against globals — so multiline code
        # like `content = open(...).read(); [x for x in content]` fails with
        # "name 'content' is not defined". A shared module-level namespace fixes that.
        exec_ns: dict = _build_exec_namespace(_capture_print, _utf8_open)
        exec(code, exec_ns)
        output = buf.getvalue()

        if not output.strip():
            output = "Code executed successfully with no output. Check your format and checkcode for return values."

    except _ComputeOnlyViolation as e:
        # A refusal, not a crash. The message names the gated tool to use instead, so the model
        # re-routes to a recorded path rather than treating this as a failure to work around.
        slog.info(f"   [ComputeOnly] REFUSED: {e}")
        output = f"Error: {e}"
    except Exception as e:
        output = f"Error: {str(e)}"

    # ── SHADOW: run the sandbox too, return the in-process answer, log where they disagree. ──
    # This is the evidence that should decide the flip, and it is the same doctrine used for the
    # gate and for compute-only: measure on real traffic before arming. It costs a process spawn
    # per call, which is why it is a mode and not the default.
    if _ISOLATION == "shadow":
        try:
            _iso = _run_isolated(code)
            if _iso is None:
                slog.info("   [ReplIsolation] shadow: sandbox unavailable, nothing compared")
            else:
                _a, _b = (output or "").strip(), _iso.strip()
                if _a == _b:
                    slog.info("   [ReplIsolation] shadow: AGREE")
                else:
                    # The interesting case, and the one worth reading: in-process succeeded and the
                    # sandbox refused. That is a real mutation this tool is performing today with
                    # no envelope and no ledger row.
                    _refused = "PermissionError" in _b and "PermissionError" not in _a
                    slog.info("   [ReplIsolation] shadow: DIVERGE%s\n"
                              "        in-process: %s\n"
                              "        sandboxed : %s"
                              % ("  ⚠️ SANDBOX REFUSED WHAT IN-PROCESS ALLOWED" if _refused else "",
                                 _a[:300], _b[:300]))
        except Exception as _e:                  # observation must never break execution
            slog.info(f"   [ReplIsolation] shadow compare failed (non-fatal): {type(_e).__name__}: {_e}")

    return output

def web_search(query: str) -> dict:
    try:
        ap = os.getenv("tavily_api")
        client = TavilyClient(ap)
        response = client.search(
            query=query,
            include_answer="advanced",
            search_depth="advanced",
            max_results=2
        )
        return response
    except Exception as e:
        return {"answer": f"Error doing web_search: {e}", "results": []}
    
def get_time_date(offset_days: int = 0, offset_minutes: int = 0) -> str:
    """Rich temporal grounding (upgraded 2026-06-12 — the old version returned a bare
    datetime repr). Gives Clara everything needed to resolve relative time: weekday,
    both clock formats, timezone, part of day, and yesterday/tomorrow anchors.

    `offset_days` (Brief 50): when non-zero, appends a DETERMINISTICALLY-computed target line for
    'N days from now / N days ago' questions — so Clara never hand-computes a calendar date/weekday
    (she reliably errs on month-boundary rollovers; +10d failed 06-24m/06-25m). Sign: future +, past −.

    `offset_minutes` (G25, 2026-08-01): the time-delta analogue — when non-zero, appends a
    deterministically-computed target CLOCK TIME (12h+24h) for 'what time in N hours/minutes' questions,
    so Clara never hand-adds minutes (2026-07-31e Q22: a CHAT turn emitted a bogus tool-call for exactly
    this). Wraps across midnight and notes the date when it does. Sign: future +, past −."""
    now = datetime.now().astimezone()
    from datetime import timedelta
    yest, tom = now - timedelta(days=1), now + timedelta(days=1)
    hour = now.hour
    part = ("early morning" if hour < 6 else "morning" if hour < 12
            else "afternoon" if hour < 17 else "evening" if hour < 21 else "night")
    tz = now.strftime("%z")
    tz_fmt = f"UTC{tz[:3]}:{tz[3:]}" if tz else "local"
    out = (
        f"Date: {now.strftime('%A, %d %B %Y')} ({now.strftime('%Y-%m-%d')})\n"
        f"Time: {now.strftime('%H:%M:%S')} (24h) / {now.strftime('%I:%M:%S %p').lstrip('0')} (12h) — {part}\n"
        f"Timezone: {tz_fmt} (IST)\n"
        f"Week {now.isocalendar()[1]} of {now.year}, day {now.timetuple().tm_yday} of the year\n"
        f"Yesterday was {yest.strftime('%A, %Y-%m-%d')}; tomorrow is {tom.strftime('%A, %Y-%m-%d')}"
    )
    try:
        n = int(offset_days)
    except (TypeError, ValueError):
        n = 0
    if n:
        tgt = now + timedelta(days=n)
        rel = f"{abs(n)} day(s) {'from today' if n > 0 else 'ago'}"
        out += (f"\n{rel.capitalize()}: {tgt.strftime('%A, %d %B %Y')} "
                f"({tgt.strftime('%Y-%m-%d')})  (computed -- do not recompute by hand)")
    try:
        mo = int(offset_minutes)
    except (TypeError, ValueError):
        mo = 0
    if mo:
        tgt_t = now + timedelta(minutes=mo)
        rel_t = f"{abs(mo)} minute(s) {'from now' if mo > 0 else 'ago'}"
        daynote = f" on {tgt_t.strftime('%A, %Y-%m-%d')}" if tgt_t.date() != now.date() else ""
        out += (f"\n{rel_t.capitalize()}: {tgt_t.strftime('%H:%M')} (24h) / "
                f"{tgt_t.strftime('%I:%M %p').lstrip('0')} (12h){daynote}  (computed -- do not recompute by hand)")
    return out

def consult_archive(query: str) -> str:
    global RAG_ENGINE
    global DB_PATH
    
    if RAG_ENGINE is None:
        if os.path.exists(DB_PATH):
            slog.info("   [Archive] Loading Vector Database into RAM...")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            RAG_ENGINE = FAISS.load_local(
                DB_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True 
            )
        else:
            return "Error: Knowledge base not found. Run core_logic/rag_db_builder.py (or restart the backend) to build it."
    
    slog.debug(f"   [Archive] Searching for: '{query}'")
    results = RAG_ENGINE.similarity_search(query, k=4)
    
    # Pro-tip: Join with a separator so the LLM knows where one chunk ends and another begins
    return "\n---\n".join([doc.page_content for doc in results])


# response_web_search = web_search("What is the price of iphone 15 pro max in INR?")
# print("Web Search Result:", response_web_search)


# NOTE (Brief 36 D-9/D-12, removed 2026-06-10): the dead fs_* quartet
# (fs_read_file / fs_list_directory / fs_write_file / fs_run_command — replaced by
# Desktop Commander tools long ago, nothing imported them) and the dead pre-Gemini
# vision helpers (_pick_detail / _compress_image — the Gemini path below does its own
# inline thumbnail+JPEG) were deleted. Git history preserves them.

import pathlib


# ── Vision Tool (Gemini 2.5 Flash; function name is legacy from the Grok era) ──

import base64


def analyze_image_grok(
    client,
    path: str,
    question: str = "Describe what you see in this image in detail.",
    paths=None,
    max_side: int = 1280,
) -> str:
    """
    Analyze image(s) using Gemini 2.5 Flash (google-genai SDK).
    'client' parameter kept for API compatibility but unused.

    max_side: images larger than this on either axis are downscaled before sending. 1280 is the
    right default for screenshots/photos (payload size), but it DESTROYS dense-figure OCR — a
    diagram crop rendered at high DPI gets flattened back to ~2.4 px/pt and the model confabulates
    labels (BRIEF_58 D2, root-caused 2026-07-23). PDF figure reading passes a higher cap.
    """
    import io
    from google import genai
    from google.genai import types
    from PIL import Image

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        return "Error: GEMINI_API_KEY not set in .env"

    image_paths = []
    if path:
        image_paths.append(path)
    if paths:
        image_paths += [p for p in paths if p and p != path]

    parts = []
    for img_path in image_paths:
        try:
            p = pathlib.Path(img_path.strip().strip('"').strip("'"))
            if not p.exists():
                return f"Error: Image not found at path: {img_path}"
            with Image.open(p) as img:
                cap = int(max_side) if max_side else 0
                if cap and (img.width > cap or img.height > cap):
                    img.thumbnail((cap, cap), Image.LANCZOS)
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=85)
                parts.append(types.Part.from_bytes(
                    data=buf.getvalue(),
                    mime_type="image/jpeg",
                ))
        except Exception as e:
            return f"Error loading image {img_path}: {e}"

    if not parts:
        return "Error: No valid images provided"

    parts.append(question)

    # Free-tier gemini-2.5-flash throws transient 503 UNAVAILABLE ("high demand")
    # fairly often — observed on the very first wired-up call (2026-06-11). Retry
    # a couple of times with backoff before reporting failure; a 503 surfaced to
    # the ReAct loop reads like a broken tool when it's a 15-second blip.
    import time as _time
    gc = genai.Client(api_key=gemini_key)
    last_err = None
    for attempt in range(3):
        try:
            response = gc.models.generate_content(
                model="gemini-2.5-flash",
                contents=parts,
            )
            return response.text
        except Exception as e:
            last_err = e
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                _time.sleep(8 * (attempt + 1))
                continue
            return f"Vision error: {e}"
    return f"Vision error after retries: {last_err}"


def analyze_images_grok(
    client,
    paths: list,
    question: str = "Describe what you see in these images.",
    detail: str = "high",
) -> str:
    """Wrapper — delegates to analyze_image_grok with multiple paths."""
    return analyze_image_grok(client, path=paths[0] if paths else "", question=question, paths=paths[1:] if len(paths) > 1 else None)


_OCR_PROMPT = (
    "Transcribe ALL text visible in this image VERBATIM, preserving reading order and line breaks. "
    "Output only the transcribed text — no commentary, no description. If there is no text, output '(no text)'."
)


_FIGURE_PROMPT = (
    "This is a figure cropped from a document page. Transcribe ALL text visible in it VERBATIM, "
    "preserving its structure (box labels, numbered lists, arrows, axes). If it is a diagram or chart, "
    "also state briefly what it depicts and how the parts connect. Do not infer or invent anything that "
    "is not legible — if part of it is unreadable, say so."
)

# Resolution constants (BRIEF_58 D2, recalibrated 2026-07-23 after root-causing the 1280px
# downscale inside analyze_image_grok): dense diagram labels read reliably at ~12 px/pt and
# confabulate below ~3 px/pt (measured on the CLARA architecture map — 4/11 labels correct at
# ~2.4 px/pt, 11/11 at 12.5 px/pt). One sent image is capped at _TILE_MAX_PX so Gemini doesn't
# downsample it internally; figures needing more resolution than one image can carry are split
# into overlapping tiles, ALL sent in ONE multi-image call so the model keeps global context.
_PX_PER_PT = 12            # target effective resolution on figure content
_TILE_MAX_PX = 2600        # max long side of any single image sent to the vision model
_TILE_GRID_MAX = (3, 2)    # max cols x rows -> at most 6 tiles + 1 overview per figure
_TILE_OVERLAP = 0.15       # fractional overlap between adjacent tiles — 0.06 left boundary words
                           # cut ("Planning Req~", run 3); 15% ≈ 26pt covers a full word either side
_IMG_MIN_PT = 40           # smaller than this is decoration (logo, rule, bullet) — don't spend a call
_IMG_MAX_PER_DOC = 20      # cost ceiling: vision calls per document (a tiled figure = ONE call)


def _describe_pdf_image(page, bbox, question: str, idx: int) -> str:
    """Render ONE image block at readable resolution — tiling large figures — and read it.

    ALWAYS returns a string: a description, or an explicit bracketed note saying why it wasn't read.
    Never silence — a silently dropped figure is the failure class BRIEF_58 exists to kill. Never
    raises. One vision call per figure regardless of tile count (tiles ride a single request).
    """
    import fitz
    import tempfile
    rect = fitz.Rect(bbox)
    w_pt, h_pt = rect.width, rect.height
    if min(w_pt, h_pt) < _IMG_MIN_PT:
        return f"[image {w_pt:.0f}x{h_pt:.0f}pt — decorative size, not read]"

    tmps = []

    def _render(clip, dpi, tag):
        pix = page.get_pixmap(dpi=dpi, clip=clip)
        fd, t = tempfile.mkstemp(prefix=f"pdffig{idx}_{tag}_", suffix=".png")
        os.close(fd)
        pix.save(t)
        tmps.append(t)
        return t

    try:
        long_pt = max(w_pt, h_pt, 1)
        if long_pt * _PX_PER_PT <= _TILE_MAX_PX:
            # Small figure: one crop carries the full target resolution.
            dpi = int(min(_PX_PER_PT * 72, _TILE_MAX_PX * 72 / long_pt))
            main = _render(rect, dpi, "full")
            desc = str(analyze_image_grok(None, path=main,
                                          question=question or _FIGURE_PROMPT,
                                          max_side=_TILE_MAX_PX)).strip()
            label = f"[FIGURE {w_pt:.0f}x{h_pt:.0f}pt @ {dpi}dpi]"
        else:
            # Large/dense figure: overlapping high-res tiles + a low-res overview, one call.
            cols = min(_TILE_GRID_MAX[0], max(1, -(-int(w_pt * _PX_PER_PT) // _TILE_MAX_PX)))
            rows = min(_TILE_GRID_MAX[1], max(1, -(-int(h_pt * _PX_PER_PT) // _TILE_MAX_PX)))
            tile_w, tile_h = w_pt / cols, h_pt / rows
            dpi = int(min(_PX_PER_PT * 72, _TILE_MAX_PX * 72 / max(tile_w, tile_h)))
            ov_w, ov_h = tile_w * _TILE_OVERLAP, tile_h * _TILE_OVERLAP
            tiles = []
            for r in range(rows):
                for c in range(cols):
                    clip = fitz.Rect(
                        rect.x0 + c * tile_w - (ov_w if c else 0),
                        rect.y0 + r * tile_h - (ov_h if r else 0),
                        rect.x0 + (c + 1) * tile_w + (ov_w if c < cols - 1 else 0),
                        rect.y0 + (r + 1) * tile_h + (ov_h if r < rows - 1 else 0),
                    )
                    tiles.append(_render(clip, dpi, f"r{r}c{c}"))
            # ONE image per call — the empirically proven configuration. A single multi-image
            # request dilutes per-tile resolution (Gemini budgets tokens across images: measured
            # 2026-07-23, tiny labels misread in a 7-image call but read perfectly one-at-a-time).
            overview = _render(rect, max(72, int(_TILE_MAX_PX * 72 / long_pt)), "ovw")
            ov_desc = str(analyze_image_grok(
                None, path=overview, max_side=_TILE_MAX_PX,
                question="This is a figure from a document page. State briefly what it depicts and "
                         "how its parts connect. Do NOT try to transcribe small text — a separate "
                         "high-resolution pass handles that.")).strip()
            tile_txts = []
            for k, t in enumerate(tiles):
                r, c = divmod(k, cols)
                tt = str(analyze_image_grok(
                    None, path=t, max_side=_TILE_MAX_PX,
                    question="This image is one high-resolution tile of a larger figure. Transcribe "
                             "ALL text visible in it VERBATIM, preserving structure (box titles, "
                             "numbered lists, labels). Output only the transcription. If a word is "
                             "cut off at an edge, transcribe the visible part followed by '~'. Do "
                             "not guess at anything illegible — mark it '(illegible)'.")).strip()
                tile_txts.append(f"[tile row {r + 1}/{rows}, col {c + 1}/{cols}]\n{tt}")
            desc = (ov_desc + "\n\nHigh-resolution text by tile (tiles overlap slightly, so some "
                    "lines repeat across adjacent tiles):\n" + "\n".join(tile_txts))
            label = f"[FIGURE {w_pt:.0f}x{h_pt:.0f}pt @ {dpi}dpi, {rows}x{cols} tiles]"
        low = desc.lower()
        if not desc or low.startswith("error") or low.startswith("vision error"):
            return f"[image {w_pt:.0f}x{h_pt:.0f}pt — could not be read: {desc or 'empty response'}]"
        return f"{label}\n{desc}\n[/FIGURE]"
    except Exception as e:
        return f"[image {w_pt:.0f}x{h_pt:.0f}pt — could not be read: {e}]"
    finally:
        for t in tmps:
            try:
                os.remove(t)
            except Exception:
                pass


def ocr_pdf(path: str, max_pages: int = 10, question: str = "") -> str:
    """Read a PDF COMPLETELY, in READING ORDER — page text and embedded figures interleaved exactly
    as they appear on the page (BRIEF_58, 2026-07-21).

    Use this for ANY PDF you need to actually understand: normal text PDFs, scans, and (the case that
    matters) mixed documents where a diagram, chart or screenshot carries information the text does not.

    How it works: for each page, PyMuPDF returns text blocks (type 0) and image blocks (type 1), each
    with a bounding box; sorting by (y, x) gives true reading order. Text is emitted as-is; each image
    is cropped to its own box, rendered at a resolution scaled to its size, and read by the vision model
    inline where it sits. There is deliberately NO "does this document have a text layer" decision — the
    old per-document short-circuit is what silently dropped every image in any PDF that had text
    anywhere (BRIEF_58 D1).

    Every image block ALWAYS appears in the output, described or explicitly noted as unread, so the
    caller can never be unaware that a figure was there (BRIEF_58 D3).

    Known limit: a diagram drawn as native PDF vectors carries no image block, so it is read only as its
    stray text labels. Embedded rasters (scans, pasted screenshots, exported diagrams) are covered.

    Read-only; never raises (returns an error string). Pages bounded by max_pages, figures by
    _IMG_MAX_PER_DOC. `question` overrides the default figure prompt if you want something specific.
    """
    try:
        import fitz  # PyMuPDF — self-contained wheel, no onnxruntime/magika (avoids the markitdown footgun)
    except Exception:
        return "Error: PyMuPDF (fitz) is not installed — cannot read PDFs."
    p = pathlib.Path(str(path).strip().strip('"').strip("'"))
    if not p.exists():
        return f"Error: PDF not found at path: {path}"
    try:
        max_pages = max(1, min(int(max_pages or 10), 25))
    except (TypeError, ValueError):
        max_pages = 10
    try:
        doc = fitz.open(str(p))
    except Exception as e:
        return f"Error: could not open PDF: {e}"
    try:
        n_pages = doc.page_count
        if n_pages == 0:
            return "Error: PDF has no pages."
        pages_to_do = min(n_pages, max_pages)
        out, n_fig, n_unread = [], 0, 0
        for i in range(pages_to_do):
            page = doc[i]
            try:
                blocks = page.get_text("dict").get("blocks", [])
            except Exception as e:
                out.append(f"--- Page {i + 1} ---\n(page could not be parsed: {e})")
                continue
            # Reading order: top-to-bottom, then left-to-right.
            blocks.sort(key=lambda b: (round((b.get("bbox") or [0, 0, 0, 0])[1], 1),
                                       round((b.get("bbox") or [0, 0, 0, 0])[0], 1)))
            parts = []
            for b in blocks:
                if b.get("type") == 0:
                    txt = "\n".join(
                        "".join(s.get("text", "") for s in (ln.get("spans") or []))
                        for ln in (b.get("lines") or [])
                    ).strip()
                    if txt:
                        parts.append(txt)
                elif b.get("type") == 1:
                    if n_fig >= _IMG_MAX_PER_DOC:
                        n_unread += 1
                        parts.append("[image not read — per-document figure budget reached]")
                        continue
                    desc = _describe_pdf_image(page, b.get("bbox"), question, n_fig + 1)
                    if desc.startswith("[FIGURE"):
                        n_fig += 1
                    else:
                        n_unread += 1
                    parts.append(desc)
            out.append(f"--- Page {i + 1} ---\n" + "\n".join(parts))
        header = (f"[Read {pages_to_do} of {n_pages} page(s) in reading order; "
                  f"{n_fig} figure(s) read, {n_unread} not read]")
        return header + "\n\n" + "\n\n".join(out)
    except Exception as e:
        # Honor the "never raises" contract: get_text()/page_count can raise on an encrypted or
        # corrupt PDF (a realistic weird-scan input) — return an error string, don't propagate.
        return f"Error: failed while reading the PDF: {e}"
    finally:
        doc.close()


# ── Ambient Recall (BRIEF_39 A1) ────────────────────────────────────────────

def ambient_recall(window: str = "24", query: str = "", date: str = "") -> str:
    """Grounded recall over the A0 watcher's observation store (read-only).
    date:   a specific day to recall — '2026-06-11', 'June 11', 'yesterday'. PREFERRED
            for "what was I doing on <day>" (no error-prone hours-back math). Overrides window.
    window: hours to look back ('24', '2', '48') when no date is given.
    query:  optional keyword filter (an explicit app/site name only)."""
    from .ambient import recall as _recall, _parse_date_anchor
    d = str(date or "").strip()
    w = str(window or "24").lower().strip()
    # If no explicit date but `window` is actually a date phrase ('june 11', 'yesterday',
    # an ISO date), route it to the date anchor instead of misparsing it as hours.
    if not d and _parse_date_anchor(w):
        d = w
    if d:
        return _recall(query=query or "", date=d)
    import re as _re
    mnum = _re.search(r"[\d.]+", w)
    hours = float(mnum.group(0)) if mnum else 24.0
    if "day" in w:
        hours *= 24
    return _recall(window_hours=hours, query=query or "")


# ── Task Status Tool ───────────────────────────────────────────────────────

# Injected at startup by api.py — allows query_task_status to access
# the live TaskGraph without circular imports.
_task_graph_ref = None

def set_task_graph(tg) -> None:
    """Called once by api.py after TaskGraph is created."""
    global _task_graph_ref
    _task_graph_ref = tg


# Injected at startup by api.py — the live Clara_Agent, so episodic_search can read
# the in-RAM episodic_log + episodic_embeddings and reuse the agent's encoder.
_agent_ref = None

def set_agent_ref(agent) -> None:
    """Called once by api.py after Clara_Agent is created."""
    global _agent_ref
    _agent_ref = agent


def whatsapp_missed(query: str = "", limit: int = 20, status: str = "", mark_read=None) -> str:
    """
    Read-only retrieval over the HELD WhatsApp archive (non-priority senders; Shobha is surfaced
    straight into the chat and is never held). TWO behaviours, by whether you name a sender:
      • No `query` → a DIGEST of UNREAD held messages ("what did I miss on WhatsApp"): grouped by
        sender, counts + a short preview. Marks NOTHING read (a glance is not engagement).
      • A sender/text in `query` → that sender's messages VERBATIM (the exact-fetch drill-down,
        "what did Yash say"). This IS engagement, so the shown messages flip unread→read
        (engage-to-read). They are never removed — ask again any time (they return under status='all').
    `status`: 'unread' | 'read' | 'all' (default: unread for the digest, all for a drill-down so a
    re-ask still returns already-read messages). `mark_read`: override the auto engage-to-read
    (True = mark shown messages read even on a digest; False = peek without marking). Read is a LABEL,
    never a delete. Brief 49.
    """
    try:
        from .conversations import read_whatsapp_held, mark_whatsapp_read
    except ImportError:
        from conversations import read_whatsapp_held, mark_whatsapp_read
    try:
        limit = max(1, min(int(limit or 20), 100))
    except (TypeError, ValueError):
        limit = 20

    q = (query or "").lower().strip()
    drilldown = bool(q)
    # status default depends on intent: a drill-down wants ALL of that sender's messages (so a re-ask
    # after engage-to-read still returns them); the digest wants only what's NEW (unread).
    st = (status or "").lower().strip()
    if st not in ("unread", "read", "all"):
        st = "all" if drilldown else "unread"
    status_filter = None if st == "all" else st

    rows = read_whatsapp_held(limit=0, status=status_filter)   # whole filtered archive, not the tail
    if drilldown:
        rows = [r for r in rows
                if q in str(r.get("sender", "")).lower() or q in str(r.get("text", "")).lower()]
    rows = rows[-limit:]

    if not rows:
        if drilldown:
            return f"No held WhatsApp messages match {query!r}."
        return ("Nothing new on WhatsApp — you're caught up. (Priority senders like Shobha come straight "
                "to the chat; ask for any sender by name to see their earlier messages, read or not.)")

    # engage-to-read: a drill-down marks EXACTLY the shown rows read (precise, by id), unless overridden.
    do_mark = mark_read if mark_read is not None else drilldown
    marked = 0
    if do_mark:
        ids = [r.get("id") for r in rows if r.get("id") and r.get("status") != "read"]
        if ids:
            marked = mark_whatsapp_read(ids=ids)

    from collections import OrderedDict
    by_sender = OrderedDict()
    for r in rows:
        by_sender.setdefault(str(r.get("sender", "unknown")), []).append(r)

    if drilldown:
        # VERBATIM: every matched message, full text, no truncation.
        head = f"{len(rows)} WhatsApp message(s) from {len(by_sender)} sender(s)"
        head += " (now marked read)" if marked else ""
        lines = [head + ":"]
        for sender, msgs in by_sender.items():
            lines.append(f"\n**{sender}** ({len(msgs)}):")
            for m in msgs:
                ts = str(m.get("ts", ""))[:16].replace("T", " ") or "undated"
                body = " ".join(str(m.get("text", "")).split())
                lines.append(f"  [{ts}] {body}")
        return "\n".join(lines)

    # DIGEST: unread overview, grouped, short preview, last 5 per sender.
    lines = [f"{len(rows)} unread held WhatsApp message(s) from {len(by_sender)} sender(s) "
             f"(priority senders go straight to the chat). Ask for a sender by name to read theirs:"]
    for sender, msgs in by_sender.items():
        lines.append(f"\n**{sender}** ({len(msgs)}):")
        for m in msgs[-5:]:
            ts = str(m.get("ts", ""))[:16].replace("T", " ") or "undated"
            body = " ".join(str(m.get("text", "")).split())
            if len(body) > 160:
                body = body[:160] + "…"
            lines.append(f"  [{ts}] {body}")
    return "\n".join(lines)


def episodic_search(query: str, k: int = 5) -> str:
    """
    Semantic search over the FULL user-facing episodic log, returning the top-k most
    relevant past interactions WITH their timestamps (best match first).

    Why this exists (Brief 47): get_smart_context injects only the top-2 semantic hits +
    last-3 by recency. That passive window structurally misses (a) older interactions, and
    (b) the two-part temporal follow-up — "have I said X?" matches on content, but the bare
    follow-up "when?" embeds to nothing near the original episode, so the timestamp is
    unreachable. A deliberate tool call reaches the whole log and returns the timestamp
    inline, so the FIRST answer can already say "yes — on June 11 ~21:30".
    """
    if _agent_ref is None or getattr(_agent_ref, "db", None) is None:
        return "Error: episodic memory not available."
    try:
        k = max(1, min(int(k or 5), 10))
    except (TypeError, ValueError):
        k = 5

    # Memory dict lives on the agent's CRUD instance (clara.db.memory), not the agent itself.
    episodes = (_agent_ref.db.memory or {}).get("episodic_log", [])
    if not episodes:
        return "No episodic memories recorded yet."

    try:
        from .crud import SYSTEM_PREFIXES
    except ImportError:
        SYSTEM_PREFIXES = ("[AUTONOMOUS]", "[TASK")
    user_idx = [i for i, ep in enumerate(episodes)
                if not ep.get("summary", "").startswith(SYSTEM_PREFIXES)]
    if not user_idx:
        return "No user-facing episodic memories recorded yet."

    def _fmt(idx, score=None):
        ep = episodes[idx]
        ts = ep.get("timestamp", "")[:16].replace("T", " ") or "undated"
        rel = f" (relevance {score:.2f})" if score is not None else ""
        return f"- [{ts}]{rel} {ep.get('summary', '')}"

    embs = getattr(_agent_ref, "episodic_embeddings", None)
    # Semantic path — only when embeddings are present AND index-aligned with the log.
    if embs and len(embs) == len(episodes):
        try:
            import torch
            q_emb = _agent_ref._encode_sync(query).to("cpu")
            user_embs = torch.stack([embs[i] for i in user_idx])
            sims = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), user_embs)
            kk = min(k, len(user_idx))
            top = sims.topk(kk).indices.tolist()
            lines = [_fmt(user_idx[li], float(sims[li])) for li in top]
            header = "Most relevant past interactions (best match first):"
            if float(sims[top[0]]) < 0.30:
                header = ("No strongly matching memory — closest interactions below "
                          "(low relevance; treat as weak matches):")
            return header + "\n" + "\n".join(lines)
        except Exception:
            pass  # fall through to keyword scan

    # Keyword fallback — embeddings missing or drifted out of alignment.
    q_words = [w for w in query.lower().split() if len(w) > 2]
    scored = []
    for i in user_idx:
        s = episodes[i].get("summary", "").lower()
        hits = sum(1 for w in q_words if w in s)
        if hits:
            scored.append((hits, i))
    if not scored:
        return f"No episodic memory mentions: {query!r}."
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return "Past interactions mentioning that (keyword match):\n" + "\n".join(
        _fmt(i) for _, i in scored[:k])


def query_task_status(keyword: str) -> str:
    """
    Search the TaskGraph for tasks whose goal contains the keyword.
    Returns a formatted status report for all matching tasks.
    Used by CLARA to answer questions like "why hasn't X finished yet?"
    """
    if _task_graph_ref is None:
        return "Error: Task graph not available."

    keyword_lower = keyword.lower().strip()

    # Search non-terminal tasks in memory
    all_tasks = list(_task_graph_ref._tasks.values())

    # Also search terminal tasks in SQLite for recent history
    try:
        import sqlite3
        conn = sqlite3.connect(_task_graph_ref._db_path)
        rows = conn.execute(
            "SELECT id, goal, state, priority, origin, context, last_updated "
            "FROM tasks ORDER BY last_updated DESC LIMIT 50"
        ).fetchall()
        conn.close()
    except Exception:
        rows = []

    # Build a unified search set — in-memory tasks take precedence
    seen_ids = {t.id for t in all_tasks}
    import json as _json

    for row in rows:
        if row[0] not in seen_ids:
            try:
                all_tasks.append(type('T', (), {
                    'id': row[0],
                    'goal': row[1],
                    'state': row[2],
                    'priority': row[3],
                    'origin': row[4],
                    'context': _json.loads(row[5]) if row[5] else {},
                    'last_updated': row[6],
                    'dependencies': [],
                })())
            except Exception:
                pass

    # Filter by keyword
    matches = [
        t for t in all_tasks
        if keyword_lower in t.goal.lower()
    ]

    if not matches:
        return f"No tasks found matching '{keyword}'."

    lines = [f"Task status report for '{keyword}':\n"]
    for t in matches[:5]:  # cap at 5 results
        checkpoint = t.context.get("checkpoint", {})
        reason     = checkpoint.get("reason", "")
        paused_at  = checkpoint.get("interrupted_at", "")

        status_line = f"• [{t.state.upper()}] {t.goal[:80]}"
        if t.state == "paused" and reason:
            status_line += f"\n  ↳ Paused: {reason}"
            if paused_at:
                status_line += f" at {paused_at[:19]}"
        elif t.state == "pending" and t.dependencies:
            status_line += f"\n  ↳ Waiting on: {len(t.dependencies)} dependency/ies"
        elif t.state == "failed":
            err = t.context.get("error", "unknown error")
            status_line += f"\n  ↳ Failed: {err[:100]}"
        elif t.state == "invalidated":
            status_line += "\n  ↳ Invalidated (resource conflict or superseded)"

        status_line += f"\n  ↳ Priority: {t.priority:.1f} | Origin: {t.origin}"
        lines.append(status_line)

    return "\n".join(lines)