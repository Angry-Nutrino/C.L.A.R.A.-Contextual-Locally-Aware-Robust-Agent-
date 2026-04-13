import sys
from io import StringIO
from datetime import datetime
from tavily import TavilyClient
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# from rag import DB_PATH
import os

load_dotenv()  # Load once at module level


RAG_ENGINE= None
current_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH=os.path.join(current_dir, "knowledge_base")

# Pre-loading rag for faster inference. 
print("   [Archive] 🔌 Pre-loading RAG Engine for instant access...")
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
    print("   [Archive] ✅ RAG Engine is Hot.")
else:
    RAG_ENGINE = None
    print("   [Archive] ⚠️ DB Not found. RAG will be disabled.")



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
            print("   [Archive] 🔌 Loading Vector Database into RAM...")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            RAG_ENGINE = FAISS.load_local(
                DB_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True 
            )
        else:
            return "Error: Knowledge base not found. Please run 'rag.py' first."
    
    print(f"   [Archive] 🧠 Searching for: '{query}'")
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
        p = pathlib.Path(path)
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
        p = pathlib.Path(path)
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
        p = pathlib.Path(path)
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