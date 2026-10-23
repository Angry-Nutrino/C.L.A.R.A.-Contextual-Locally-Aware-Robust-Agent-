"""CLARA's pre-execution admissibility gate, served over MCP.

Any MCP client (an agent, an IDE, a desktop app) can ask this server before it runs an action that changes
something, and get back a verdict and a receipt. Right before executing, the client sends the arguments it is
about to run, and the server confirms they are byte-for-byte the ones the verdict was given for. Every decision
goes to a ledger, and with a signing key configured both the request and the decision are Ed25519-signed, so a
third party can check a receipt later with nothing but the public key.

The adjudication is CLARA's own gate (core_logic/admissibility.py) running its LOCAL written policy. This server
never uses a remote adapter: no envelope ever leaves the machine. Envelopes carry hashes and argument lengths,
never argument values.

What it does NOT do: it cannot stop a client that ignores the verdict. The gate decides and records; the client
has to call `check_action` first, honour DENY and REVIEW, and call `confirm_args` right before it executes.

Configuration (environment variables, all optional):
    CLARA_GATE_LEDGER        ledger file (default: ~/.clara-gate/gate_ledger.json). Never CLARA's own ledger.
    CLARA_GATE_POLICY        policy JSON (default: core_logic/admissibility_policy.json)
    CLARA_GATE_SIGNING_KEY   base64 Ed25519 private seed; without it receipts are recorded unsigned
    CLARA_GATE_ARGS_KEY      optional HMAC key for the argument digest (blocks confirming guessed values)
    CLARA_GATE_AGENT_ID      agent_id written into envelopes (default: mcp-client)
    CLARA_GATE_ADAPTER       "policy" (default) or "noop"; remote adapters are refused

Run:    python mcp_servers/clara_gate/server.py            (stdio MCP server)
Keys:   python mcp_servers/clara_gate/server.py --new-key  (prints a fresh key pair; nothing is stored)
"""
import base64
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

# The gate module loads core_logic/.env at import. Every setting it reads is fixed HERE first, because
# load_dotenv never overrides a variable that is already set: so CLARA's own signing key, adapter and
# modes can never leak into this server, and this server's settings can never leak into CLARA.
_ADAPTER = os.environ.get("CLARA_GATE_ADAPTER", "policy").strip().lower() or "policy"
if _ADAPTER not in ("policy", "noop"):
    sys.stderr.write("clara-gate: CLARA_GATE_ADAPTER must be 'policy' or 'noop'; remote adapters are refused\n")
    sys.exit(2)
os.environ["ADMISSIBILITY_GATE"] = "on"
os.environ["ADMISSIBILITY_ADAPTER"] = _ADAPTER
os.environ["ADMISSIBILITY_MODE"] = "enforce"
os.environ["ADMISSIBILITY_INTERPRETER_MODE"] = "enforce"
os.environ["ADMISSIBILITY_FAIL"] = "closed"
os.environ["ADMISSIBILITY_SIGNING_KEY"] = os.environ.get("CLARA_GATE_SIGNING_KEY", "")
os.environ["ADMISSIBILITY_ARGS_KEY"] = os.environ.get("CLARA_GATE_ARGS_KEY", "")
os.environ["ADMISSIBILITY_AGENT_ID"] = os.environ.get("CLARA_GATE_AGENT_ID", "mcp-client")
os.environ["POLICY_VERSION"] = ""        # always the content hash of the policy actually in use

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from core_logic import admissibility as adm  # noqa: E402

LEDGER_PATH = os.path.expanduser(os.environ.get(
    "CLARA_GATE_LEDGER", os.path.join("~", ".clara-gate", "gate_ledger.json")))
adm._LEDGER_PATH = LEDGER_PATH
if os.environ.get("CLARA_GATE_POLICY", "").strip():
    adm._POLICY_PATH = os.path.abspath(os.path.expanduser(os.environ["CLARA_GATE_POLICY"]))
# The gate's loader treats an unreadable policy as {} and then defaults to ALLOW, which is right for CLARA's
# own startup (startup_check covers it there) and silently fail-open here. So a policy this server cannot
# read stops it from starting at all.
try:
    with open(adm._POLICY_PATH, encoding="utf-8") as _fh:
        _policy_doc = json.load(_fh)
    if not isinstance(_policy_doc, dict) or "default" not in _policy_doc:
        raise ValueError("policy must be a JSON object with a 'default' key")
except (OSError, ValueError) as _e:
    sys.stderr.write("clara-gate: cannot use policy %s (%s); refusing to start\n" % (adm._POLICY_PATH, _e))
    sys.exit(2)
os.makedirs(os.path.dirname(LEDGER_PATH) or ".", exist_ok=True)


def _ledger_rows() -> list:
    try:
        with open(LEDGER_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        rows = data.get("decisions") if isinstance(data, dict) else None
        return rows if isinstance(rows, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _find(receipt_id: str):
    rid = (receipt_id or "").strip()
    if not rid:
        return None
    for row in reversed(_ledger_rows()):
        if row.get("receipt_id") == rid:
            return row
    return None


def _public_key_b64() -> str:
    key = adm._signing_key()
    if key is None:
        return ""
    from cryptography.hazmat.primitives import serialization
    raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


# ── The tools, as plain functions (tested directly in test_server.py) ─────────────

def check_action(tool: str, args: dict) -> dict:
    """Adjudicate one proposed action before it runs. Returns verdict, reason, receipt_id, nonce, args_digest."""
    if not isinstance(args, dict):
        return {"error": "args must be an object"}
    d = adm.gate(str(tool or ""), args)
    d["signed"] = bool(_public_key_b64()) and bool(d.get("receipt_id"))
    if not d.get("receipt_id"):
        d["note"] = ("no receipt: this tool name is not one the gate treats as state-changing. Known names: "
                     + ", ".join(sorted(adm.MUTATING_TOOLS)) + "; names containing "
                     + ", ".join(adm._MUTATING_HINTS) + " are also gated.")
    return d


def confirm_args(receipt_id: str, args: dict) -> dict:
    """Right before executing, confirm `args` are exactly the arguments the verdict was given for."""
    row = _find(receipt_id)
    if row is None:
        return {"receipt_id": receipt_id, "found": False, "proceed": False,
                "reason": "no decision with this receipt_id in the ledger"}
    env = row.get("envelope") or {}
    match = adm.args_match({"nonce": env.get("nonce", ""), "args_digest": env.get("args_digest", "")},
                           args if isinstance(args, dict) else {})
    proceed = match and row.get("verdict") == adm.ALLOW
    return {"receipt_id": receipt_id, "found": True, "args_match": match, "verdict": row.get("verdict"),
            "proceed": proceed,
            "reason": ("arguments match an ALLOW decision" if proceed else
                       "arguments differ from the ones decided on" if not match else
                       "arguments match, but the verdict was " + str(row.get("verdict")))}


def verify_receipt(receipt_id: str, public_key_b64: str = "") -> dict:
    """Check a receipt's two signatures: the request (envelope) and the decision made about it."""
    row = _find(receipt_id)
    if row is None:
        return {"receipt_id": receipt_id, "found": False}
    pub = (public_key_b64 or "").strip() or _public_key_b64()
    signed = bool((row.get("envelope") or {}).get("signature")) or bool(row.get("decision_signature"))
    if not signed:
        return {"receipt_id": receipt_id, "found": True, "signed": False,
                "reason": "recorded without a signing key; nothing to verify"}
    if not pub:
        return {"receipt_id": receipt_id, "found": True, "signed": True,
                "reason": "signed, but no public key was given and none is configured"}
    env_ok = adm.verify_envelope(row.get("envelope") or {}, pub)
    dec_ok = adm.verify_decision(row, pub)
    return {"receipt_id": receipt_id, "found": True, "signed": True, "envelope_valid": env_ok,
            "decision_valid": dec_ok, "valid": env_ok and dec_ok, "verdict": row.get("verdict")}


def recent_decisions(limit: int = 10) -> list:
    """The newest decisions in the ledger: receipt, tool, verdict, reason, time. Never argument values."""
    try:
        n = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        n = 10
    out = []
    for row in reversed(_ledger_rows()[-n:]):
        env = row.get("envelope") or {}
        out.append({"receipt_id": row.get("receipt_id"), "tool": env.get("tool"), "verdict": row.get("verdict"),
                    "reason": row.get("reason"), "risk_class": env.get("risk_class"),
                    "irreversible": env.get("irreversible"), "timestamp": env.get("timestamp"),
                    "signed": bool(env.get("signature"))})
    return out


def gate_info() -> dict:
    """What this server is running: adapter, policy version, ledger path, and the public key if signing."""
    return {"adapter": _ADAPTER, "policy_version": adm._policy_version(), "ledger": LEDGER_PATH,
            "signing": bool(_public_key_b64()), "public_key_b64": _public_key_b64(),
            "args_digest_alg": adm.ARGS_DIGEST_ALG_KEYED if adm._args_key() else adm.ARGS_DIGEST_ALG}


TOOLS = (check_action, confirm_args, verify_receipt, recent_decisions, gate_info)


def _new_key() -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    key = Ed25519PrivateKey.generate()
    seed = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                             serialization.NoEncryption())
    pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    print("CLARA_GATE_SIGNING_KEY=" + base64.b64encode(seed).decode())
    print("public key (share this): " + base64.b64encode(pub).decode())


def main() -> None:
    if "--new-key" in sys.argv[1:]:
        _new_key()
        return
    from mcp.server.fastmcp import FastMCP
    server = FastMCP("clara-gate")
    for fn in TOOLS:
        server.tool()(fn)
    server.run()


if __name__ == "__main__":
    main()
