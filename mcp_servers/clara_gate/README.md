# clara-gate: a pre-execution gate as an MCP server

Ask before you act, and get a receipt you can check later. This server exposes CLARA's admissibility gate to any
MCP client. A client asks it about an action that changes something (writing a file, starting a process, running
code) before running it, gets a verdict and a receipt, and right before executing confirms that the arguments it
is about to run are exactly the ones the verdict was given for.

It runs CLARA's own gate (`core_logic/admissibility.py`) with its local written policy. Nothing leaves the
machine: there is no remote adapter, and the recorded request carries hashes and argument lengths, never the
argument values or raw paths.

## Tools

| tool | what it does |
|---|---|
| `check_action(tool, args)` | adjudicates the action against the policy: `ALLOW`, `REVIEW` or `DENY`, with a reason, a `receipt_id`, and the `nonce` and `args_digest` it was decided on |
| `confirm_args(receipt_id, args)` | right before executing: `proceed` is true only when `args` match the decided ones exactly and the verdict was `ALLOW` |
| `verify_receipt(receipt_id, public_key_b64?)` | checks both Ed25519 signatures, the request and the decision about it |
| `recent_decisions(limit?)` | the newest decisions: tool, verdict, reason, risk class, time |
| `gate_info()` | the adapter, the policy version (a hash of the policy file in use), the ledger path, the public key |

The digest is `sha256(nonce + ":" + canonical JSON of the args)`, so anyone can recompute it with nothing but a
hash function. Set `CLARA_GATE_ARGS_KEY` and it becomes an HMAC instead, which stops a reader of the ledger from
confirming a guessed short value, at the cost of public recomputation.

## Run it

```
python mcp_servers/clara_gate/server.py --new-key     # prints a signing key and its public key; stores nothing
python mcp_servers/clara_gate/server.py               # stdio MCP server
python mcp_servers/clara_gate/test_server.py          # the test suite
```

Client config (Claude Desktop, Claude Code and other MCP clients use this shape):

```json
{
  "mcpServers": {
    "clara-gate": {
      "command": "python",
      "args": ["<repo>/mcp_servers/clara_gate/server.py"],
      "env": { "CLARA_GATE_SIGNING_KEY": "<from --new-key>" }
    }
  }
}
```

| variable | default |
|---|---|
| `CLARA_GATE_LEDGER` | `~/.clara-gate/gate_ledger.json`; CLARA's own ledger is never used |
| `CLARA_GATE_POLICY` | `core_logic/admissibility_policy.json`; a missing or unreadable policy stops the server from starting |
| `CLARA_GATE_SIGNING_KEY` | unset: decisions are recorded unsigned, and `verify_receipt` says so |
| `CLARA_GATE_ARGS_KEY` | unset: plain SHA-256 digest |
| `CLARA_GATE_AGENT_ID` | `mcp-client` |
| `CLARA_GATE_ADAPTER` | `policy`; `noop` allowed; any remote adapter is refused at startup |

## Tests

`test_server.py` runs 32 checks against a throwaway ledger and key. The ones that matter most are attacks: a
same-length substitution of one character in the arguments, a different path, an added argument, an unknown
receipt, a DENY edited to ALLOW in the ledger, an edited request, the wrong public key, a remote adapter at
startup, and a missing policy file. Each one must be caught. It also checks that no argument value or raw path reaches the ledger, that
CLARA's own ledger is untouched, and it finishes with a real MCP client calling the server over stdio.

## Limits

- **It decides; it can't enforce.** A client that never calls `check_action`, ignores a `DENY`, or skips
  `confirm_args` is not stopped. Enforcement belongs in the client's own executor, the way CLARA's executor
  re-checks the arguments before every gated call.
- **The policy is the one in the file.** Out of the box it denies the kill and terminate tools and paths matching
  `C:\Windows`, `System32`, `.env` and `memory.json`, and allows the rest. Write your own and point
  `CLARA_GATE_POLICY` at it.
- **Tool names are CLARA's vocabulary.** `write_file`, `edit_block`, `move_file`, `create_directory`,
  `start_process`, `interact_with_process`, `kill_process`, `force_terminate`, `python_repl` and
  `run_python_code` are gated, as is any name containing `write`, `create`, `move`, `delete`, `remove`, `kill`,
  `terminate` or `edit_`. Other names are allowed with no receipt, and the response says so.
- **A signature proves the record wasn't changed.** It does not prove the policy was right.
