import sys
from io import StringIO
from datetime import datetime
from tavily import TavilyClient
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from .session_logger import slog
import os

load_dotenv()  # Load once at module level


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


def run_python_code(code: str) -> str:
    redirected_output = StringIO()
    old_stdout = sys.stdout
    sys.stdout = redirected_output

    try:
        exec(code)
        output = redirected_output.getvalue()
        
        if not output.strip():
            output = "Code executed successfully with no output. Check your format and checkcode for return values."
    
    except Exception as e:
        output = f"Error: {str(e)}"
    finally:
        sys.stdout = old_stdout

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
    
def get_time_date() -> str:
    return str(datetime.now())

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
            return "Error: Knowledge base not found. Please run 'rag.py' first."
    
    slog.debug(f"   [Archive] Searching for: '{query}'")
    results = RAG_ENGINE.similarity_search(query, k=3)
    
    # Pro-tip: Join with a separator so the LLM knows where one chunk ends and another begins
    return "\n---\n".join([doc.page_content for doc in results])


# response_web_search = web_search("What is the price of iphone 15 pro max in INR?")
# print("Web Search Result:", response_web_search)


# ── File System Tools ──────────────────────────────────────────────────────

import pathlib
import subprocess

def fs_read_file(path: str) -> str:
    """
    Read the contents of a file at the given path.
    Returns file contents as string, or an error message on failure.
    Enforces a 10,000 character read limit to protect context window.
    """
    try:
        p = pathlib.Path(path).expanduser()
        if not p.exists():
            return f"Error: File not found: {path}"
        if not p.is_file():
            return f"Error: Path is not a file: {path}"
        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > 10_000:
            content = content[:10_000] + f"\n\n[... truncated — file is {len(content)} chars total]"
        return content
    except PermissionError:
        return f"Error: Permission denied reading {path}"
    except Exception as e:
        return f"Error reading file: {e}"


def fs_list_directory(path: str) -> str:
    """
    List files and directories at the given path.
    Returns a formatted string listing, or an error message on failure.
    """
    try:
        p = pathlib.Path(path).expanduser()
        if not p.exists():
            return f"Error: Path not found: {path}"
        if not p.is_dir():
            return f"Error: Path is not a directory: {path}"
        items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        if not items:
            return f"Directory is empty: {path}"
        lines = []
        for item in items:
            prefix = "[FILE]" if item.is_file() else "[DIR] "
            size   = f" ({item.stat().st_size:,} bytes)" if item.is_file() else ""
            lines.append(f"{prefix} {item.name}{size}")
        return f"Contents of {path}:\n" + "\n".join(lines)
    except PermissionError:
        return f"Error: Permission denied listing {path}"
    except Exception as e:
        return f"Error listing directory: {e}"


def fs_write_file(path: str, content: str) -> str:
    """
    Write content to a file at the given path.
    Creates parent directories if they do not exist.
    Returns confirmation or error message.
    """
    try:
        p = pathlib.Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"File written successfully: {path} ({len(content):,} chars)"
    except PermissionError:
        return f"Error: Permission denied writing to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def fs_run_command(command: str) -> str:
    """
    Execute a shell command and return its output.
    Timeout: 30 seconds. Returns stdout + stderr combined.
    Uses PowerShell on Windows.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace",
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        if not output.strip():
            output = f"Command completed with exit code {result.returncode} (no output)"
        if len(output) > 5_000:
            output = output[:5_000] + "\n[... output truncated]"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        return f"Error running command: {e}"


# ── Vision Tool (Grok Vision API) ─────────────────────────────────────────

import base64

# Injected at startup by api.py — allows vision functions to access the
# live xAI client without circular imports.
_xai_client_ref = None

def set_xai_client(client) -> None:
    """Called once by api.py after Client is created."""
    global _xai_client_ref
    _xai_client_ref = client


_VISION_TEXT_KEYWORDS = (
    "read", "text", "code", "number", "exact", "ocr",
    "written", "says", "write", "characters", "digits",
)

def _pick_detail(question: str) -> str:
    """
    Auto-select vision detail level based on question intent.
    Text/code reading → high (needs tile-level resolution).
    Layout/visual description → low (3-4× faster, saves 2-5s).
    """
    q = question.lower()
    if any(kw in q for kw in _VISION_TEXT_KEYWORDS):
        return "high"
    return "low"


def _compress_image(path: pathlib.Path) -> tuple[str, str]:
    """
    Compress image to JPEG at 85% quality, resize to ≤1280px wide.
    Returns (base64_string, media_type).
    Falls back to raw PNG encoding if Pillow is not available.
    """
    try:
        from PIL import Image as PILImage
        import io
        img = PILImage.open(path).convert("RGB")
        # Resize if wider than 1280px
        if img.width > 1280:
            ratio = 1280 / img.width
            img = img.resize(
                (1280, int(img.height * ratio)),
                PILImage.LANCZOS,
            )
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"
    except ImportError:
        # Pillow not installed — fall back to raw bytes
        return base64.b64encode(path.read_bytes()).decode("utf-8"), "image/jpeg"
    except Exception:
        return base64.b64encode(path.read_bytes()).decode("utf-8"), "image/jpeg"


def analyze_image_grok(
    client,
    path: str,
    question: str = "Describe what you see in this image in detail.",
    detail: str = "auto",
) -> str:
    """
    Analyze an image using Grok Vision API.
    Reads image from local path, compresses to JPEG (≤1280px, 85% quality),
    auto-selects detail level based on question intent.
    detail="auto" → _pick_detail() selects "high" for text/code, "low" otherwise.
    detail="high"/"low" → forces that level explicitly.
    Returns description string or error message.
    """
    if client is None:
        return "Error: Vision client not initialized. Try again in a moment."
    try:
        p = pathlib.Path(path.strip().strip('"').strip("'"))
        if not p.exists():
            return f"Error: Image not found at path: {path}"
        if not p.is_file():
            return f"Error: Path is not a file: {path}"

        resolved_detail = _pick_detail(question) if detail == "auto" else detail

        b64, media_type = _compress_image(p)
        data_url = f"data:{media_type};base64,{b64}"

        from xai_sdk.chat import user, image as sdk_image
        llm = client.chat.create(model="grok-4-1-fast-non-reasoning")
        llm.append(user(sdk_image(data_url, detail=resolved_detail), question))
        response = llm.sample()
        return response.content.strip()

    except Exception as e:
        return f"Error analyzing image: {e}"


def analyze_images_grok(
    client,
    paths: list,
    question: str = "Describe what you see in these images.",
    detail: str = "high",
) -> str:
    """
    Analyze multiple images in a single Grok Vision API call.
    paths: list of absolute path strings.
    """
    try:
        from xai_sdk.chat import user, image as sdk_image
        content_parts = []
        for path in paths:
            p = pathlib.Path(path.strip().strip('"').strip("'"))
            if not p.exists():
                content_parts.append(f"[Image not found: {path}]")
                continue
            ext = p.suffix.lower()
            media_types = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".gif": "image/gif",
                ".webp": "image/webp",
            }
            media_type = media_types.get(ext, "image/jpeg")
            b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
            data_url = f"data:{media_type};base64,{b64}"
            content_parts.append(sdk_image(data_url, detail=detail))

        content_parts.append(question)
        llm = client.chat.create(model="grok-4-1-fast-non-reasoning")
        llm.append(user(*content_parts))
        response = llm.sample()
        return response.content.strip()

    except Exception as e:
        return f"Error analyzing images: {e}"


# ── Task Status Tool ───────────────────────────────────────────────────────

# Injected at startup by api.py — allows query_task_status to access
# the live TaskGraph without circular imports.
_task_graph_ref = None

def set_task_graph(tg) -> None:
    """Called once by api.py after TaskGraph is created."""
    global _task_graph_ref
    _task_graph_ref = tg


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