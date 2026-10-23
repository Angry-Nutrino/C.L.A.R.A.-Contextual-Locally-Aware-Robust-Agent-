"""Tests for the clara-gate MCP server, both directions: what must pass, and attacks that must fail.

    python mcp_servers/clara_gate/test_server.py      (exit 0 = all pass)

Runs against a throwaway ledger and a throwaway signing key. CLARA's own ledger must be untouched afterwards,
and the last check drives the real server over MCP stdio, which also proves nothing else writes to stdout.
"""
import asyncio
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SERVER = os.path.join(HERE, "server.py")
CLARA_LEDGER = os.path.join(ROOT, "core_logic", "admissibility_ledger.json")

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402


def _fresh_key():
    k = Ed25519PrivateKey.generate()
    seed = k.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                           serialization.NoEncryption())
    pub = k.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(seed).decode(), base64.b64encode(pub).decode()


def _digest(path):
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return "absent"


TMP = tempfile.mkdtemp(prefix="clara_gate_test_")
SEED, PUB = _fresh_key()
os.environ["CLARA_GATE_LEDGER"] = os.path.join(TMP, "ledger.json")
os.environ["CLARA_GATE_SIGNING_KEY"] = SEED
os.environ.pop("CLARA_GATE_ADAPTER", None)
os.environ.pop("CLARA_GATE_ARGS_KEY", None)
CLARA_BEFORE = _digest(CLARA_LEDGER)

sys.path.insert(0, HERE)
import server as gate  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else "   " + str(detail)))
    if not cond:
        FAILS.append(name)


def rewrite_row(receipt_id, mutate):
    with open(gate.LEDGER_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    for row in data["decisions"]:
        if row.get("receipt_id") == receipt_id:
            mutate(row)
    with open(gate.LEDGER_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


note_path = os.path.join(TMP, "notes", "draft.txt")
secret_value = "TOP-SECRET-VALUE-4471"

# 1. a tool the gate does not treat as state-changing: allowed, no receipt, and told why
r = gate.check_action("read_file", {"path": note_path})
check("non-mutating tool is allowed without a receipt", r["verdict"] == "ALLOW" and not r["receipt_id"], r)
check("non-mutating tool gets the explanatory note", "no receipt" in r.get("note", ""), r)

# 2. an ordinary write: allowed, receipted, signed
ok_args = {"path": note_path, "content": secret_value}
ok = gate.check_action("write_file", ok_args)
check("ordinary write is ALLOW with a receipt", ok["verdict"] == "ALLOW" and len(ok["receipt_id"]) == 12, ok)
check("the decision is signed", ok.get("signed") is True, ok)
check("nonce and args_digest come back to the caller", ok.get("nonce") and ok.get("args_digest"), ok)

# 3. the written policy denies a secrets path and a denied tool
deny_path = gate.check_action("write_file", {"path": os.path.join(TMP, ".env"), "content": "x"})
check("write to a .env path is DENY", deny_path["verdict"] == "DENY" and deny_path["enforced"], deny_path)
deny_tool = gate.check_action("kill_process", {"pid": 4242})
check("kill_process is DENY by tool", deny_tool["verdict"] == "DENY", deny_tool)
check("bad args type is refused", "error" in gate.check_action("write_file", "not-a-dict"))

# 4. the exact-argument binding
c = gate.confirm_args(ok["receipt_id"], ok_args)
check("same arguments confirm and may proceed", c["args_match"] and c["proceed"], c)
same_len = dict(ok_args, content=secret_value[:-1] + "2")        # same length, one character changed
c = gate.confirm_args(ok["receipt_id"], same_len)
check("ATTACK same-length substitution is caught", not c["args_match"] and not c["proceed"], c)
c = gate.confirm_args(ok["receipt_id"], dict(ok_args, path=os.path.join(TMP, "elsewhere.txt")))
check("ATTACK a different path is caught", not c["args_match"] and not c["proceed"], c)
c = gate.confirm_args(ok["receipt_id"], dict(ok_args, extra=True))
check("ATTACK an added argument is caught", not c["args_match"], c)
c = gate.confirm_args(deny_path["receipt_id"], {"path": os.path.join(TMP, ".env"), "content": "x"})
check("matching args on a DENY still may not proceed", c["args_match"] and not c["proceed"], c)
c = gate.confirm_args("000000000000", ok_args)
check("ATTACK an unknown receipt may not proceed", not c["found"] and not c["proceed"], c)

# 5. receipts verify, and tampering shows
v = gate.verify_receipt(ok["receipt_id"])
check("a real receipt verifies with the configured key", v.get("valid") is True, v)
check("a real receipt verifies with the shared public key", gate.verify_receipt(ok["receipt_id"], PUB).get("valid"))
_, other_pub = _fresh_key()
check("ATTACK the wrong public key fails", gate.verify_receipt(ok["receipt_id"], other_pub).get("valid") is False)
rewrite_row(deny_path["receipt_id"], lambda row: row.update(verdict="ALLOW"))
v = gate.verify_receipt(deny_path["receipt_id"])
check("ATTACK a DENY edited to ALLOW fails the decision signature", v.get("decision_valid") is False, v)
rewrite_row(deny_tool["receipt_id"], lambda row: row["envelope"].update(tool="write_file"))
v = gate.verify_receipt(deny_tool["receipt_id"])
check("ATTACK an edited request fails the envelope signature", v.get("envelope_valid") is False, v)
check("ATTACK an edited request fails the receipt as a whole", v.get("valid") is False, v)

# 6. an unsigned row is reported honestly
with open(gate.LEDGER_PATH, encoding="utf-8") as fh:
    data = json.load(fh)
data["decisions"].append({"receipt_id": "unsigned0001", "verdict": "ALLOW", "envelope": {"tool": "write_file"}})
with open(gate.LEDGER_PATH, "w", encoding="utf-8") as fh:
    json.dump(data, fh)
v = gate.verify_receipt("unsigned0001")
check("an unsigned receipt says so instead of passing", v.get("signed") is False and "valid" not in v, v)

# 7. privacy, isolation, listing, info
with open(gate.LEDGER_PATH, encoding="utf-8") as fh:
    ledger_text = fh.read()
check("no argument value is ever written to the ledger", secret_value not in ledger_text)
check("no raw path is written to the ledger", note_path not in ledger_text and "draft.txt" not in ledger_text)
check("CLARA's own ledger is untouched", _digest(CLARA_LEDGER) == CLARA_BEFORE)
rd = gate.recent_decisions(3)
check("recent_decisions returns the newest first, capped", len(rd) == 3 and rd[0]["receipt_id"] == "unsigned0001", rd)
check("recent_decisions survives a junk limit", isinstance(gate.recent_decisions("junk"), list))
info = gate.gate_info()
check("gate_info reports the local policy adapter and the key", info["adapter"] == "policy" and info["public_key_b64"] == PUB, info)

# 8. a remote adapter is refused at startup
env = dict(os.environ, CLARA_GATE_ADAPTER="partner_a")
p = subprocess.run([sys.executable, SERVER], env=env, capture_output=True, text=True, timeout=60)
check("ATTACK a remote adapter is refused at startup", p.returncode == 2 and "refused" in p.stderr, p.stderr)
env = dict(os.environ, CLARA_GATE_POLICY=os.path.join(TMP, "no_such_policy.json"))
p = subprocess.run([sys.executable, SERVER], env=env, capture_output=True, text=True, timeout=60)
check("ATTACK a missing policy file stops startup instead of allowing everything",
      p.returncode == 2 and "refusing to start" in p.stderr, p.stderr)


# 9. the real thing: an MCP client over stdio
async def _mcp_roundtrip():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    env2 = dict(os.environ, CLARA_GATE_LEDGER=os.path.join(TMP, "stdio_ledger.json"))
    params = StdioServerParameters(command=sys.executable, args=[SERVER], env=env2)
    async with stdio_client(params) as (rd_, wr_):
        async with ClientSession(rd_, wr_) as s:
            await s.initialize()
            names = sorted(t.name for t in (await s.list_tools()).tools)
            res = await s.call_tool("check_action", {"tool": "write_file",
                                                     "args": {"path": note_path, "content": "hi"}})
            payload = json.loads(res.content[0].text)
            conf = await s.call_tool("confirm_args", {"receipt_id": payload["receipt_id"],
                                                      "args": {"path": note_path, "content": "hi"}})
            return names, payload, json.loads(conf.content[0].text)


try:
    names, payload, conf = asyncio.run(_mcp_roundtrip())
    check("MCP lists the five tools", names == sorted(f.__name__ for f in gate.TOOLS), names)
    check("MCP check_action returns a receipted ALLOW", payload.get("verdict") == "ALLOW" and payload.get("receipt_id"), payload)
    check("MCP confirm_args proceeds on the same args", conf.get("proceed") is True, conf)
except Exception as e:  # a protocol failure is a test failure, never a skip
    check("MCP stdio round trip", False, repr(e))

print("\n%d failed" % len(FAILS) if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
