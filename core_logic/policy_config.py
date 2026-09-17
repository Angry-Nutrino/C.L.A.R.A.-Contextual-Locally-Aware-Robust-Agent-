# -*- coding: utf-8 -*-
"""BRIEF 62 — the single owner of security-relevant configuration.

WHY THIS MODULE EXISTS
======================
Three defects of one class were found on 2026-09-07, and a fourth family on 2026-09-09.

1. `PYTHON_REPL_COMPUTE_ONLY` was set to 1 in `core_logic/.env` and the module reading it never
   loaded that file. A bare `load_dotenv()` resolves nothing from the repo root because there is no
   `.env` there, and `agent.py` imports `tools` 174 lines before it loads the env file by path. So the
   flag read its code default and arming it was a silent no-op. Same shape in two more modules.

2. Worse, and measured on 2026-09-09: an INVALID value did not error either. It silently degraded
   toward the permissive option.

       ADMISSIBILITY_MODE=enforced   ->  "shadow"      (one letter. enforcement never armed)
       ADMISSIBILITY_FAIL=close      ->  fail-OPEN     (missing 'd'. a broken adapter now ALLOWS)
       ADMISSIBILITY_GATE=yes please ->  off           (gate silently disabled)

   The first defect was "the file was never read". The second is "the file was read and the value was
   silently discarded". Both end in the same place: **the operator believes a control is armed and it
   is not.**

THE RULE THIS IMPLEMENTS
========================
A permissive default turns missing configuration into an authority grant. So the line is drawn at
AUTHORITY, not at whether a default happens to be declared:

    if a missing or invalid field gives the system MORE authority  -> refuse to start
    if the default keeps authority the same or reduces it          -> start, and record what won

The authority direction is not written down as prose that someone could get wrong. Each value
compiles into an explicit CAPABILITY SET, and the direction is DERIVED by set inclusion: if the
fallback's capability set is a strict superset of some other allowed value's, the fallback grants
authority. Nobody annotates "this one is dangerous", so nobody can annotate it wrongly.

WHAT THIS MODULE DOES NOT DO, STATED SO THE CLAIM STAYS HONEST
==============================================================
It does not enforce anything. It resolves, validates, and reports. A capability set here is a
DESCRIPTION of the authority a configuration is intended to carry, and every entry therefore also
records WHAT ENFORCES IT and HOW STRONG that primitive is. One of them is `behavioural`, because it
is a language-level namespace inside the same process and an escape has been demonstrated against
it (see `tests/test_compute_only_containment.py`). A manifest that hides that would be exactly the
"commentary beside enforcement" this design exists to remove.

    python core_logic/policy_config.py          # print the effective policy and exit non-zero on a fault
"""
import hashlib
import json
import os
import sys

from dotenv import load_dotenv

# The one load. Explicit path, so it cannot depend on the working directory or on import order,
# which is the whole point of this module.
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_PATH)

# ── Capability vocabulary ────────────────────────────────────────────────────────────────────────
# Coarse on purpose. These are the effects worth reasoning about at the configuration layer.
# IMPORTANT: A CAPABILITY HERE IS ALWAYS "AUTHORITY THE SYSTEM HOLDS", never "what the control does".
# The first draft of this file mixed the two: `shadow` was given {gate.evaluates, without_verdict} and
# `enforce` {gate.evaluates, can_refuse}, which are INCOMPARABLE sets, so set inclusion could not derive
# that shadow is the permissive one and ADMISSIBILITY_MODE=enforced typo-ed straight past the check.
# Modelling only held authority makes the pairs properly nested. Caught 2026-09-09 by the fixture run,
# which is the argument for deriving the direction instead of annotating it: a wrong SET fails a test,
# a wrong ANNOTATION fails nothing.
CAP_FS_WRITE = "filesystem.mutate"
CAP_PROC = "process.execute"
CAP_NET = "network.egress"
CAP_UNADJUDICATED = "action.without_verdict"   # may act with no binding verdict available

# Enforcement primitives, with an honest strength. `structural` means something outside the process
# maintains it. `behavioural` means the process maintains it against itself.
STRENGTH = {
    "namespace": ("language-level namespace reconstruction, same process", "behavioural"),
    "branch": ("an if-statement in the gate", "behavioural"),
    "none": ("nothing; this value is descriptive only", "none"),
}


class Field(object):
    """One security-relevant setting, its allowed values, and the capability set each value implies.

    `caps` maps an allowed value -> frozenset of capabilities the system HOLDS at that value.
    The authority direction of the fallback is derived from these sets, never declared.
    """

    def __init__(self, name, default, caps, enforced_by, note, aliases=None, open_valued=False):
        self.name = name
        self.default = default
        self.caps = {k: frozenset(v) for k, v in caps.items()}
        self.enforced_by = enforced_by
        self.note = note
        self.aliases = aliases or {}          # raw value -> canonical value
        self.open_valued = open_valued        # free-form string, cannot be enumerated

    # -- normalisation ---------------------------------------------------------------------------
    def canon(self, raw):
        """Canonical value, or None if the raw string is not a value this field accepts."""
        if raw is None:
            return None
        v = raw.strip().lower()
        v = self.aliases.get(v, v)
        if self.open_valued:
            return raw.strip() or None
        return v if v in self.caps else None

    # -- the derived authority direction ---------------------------------------------------------
    def fallback_grants_authority(self):
        """Does falling back to the default hand the system authority it need not have?

        DERIVED, not declared: true when the default's capability set is a strict superset of at
        least one other allowed value's. No human writes 'authority expanding' anywhere.
        """
        if self.open_valued:
            return False
        d = self.caps.get(self.default)
        if d is None:
            return True                       # a default that is not even a legal value
        return any(d > other for k, other in self.caps.items() if k != self.default)


_BOOL_ALIASES = {"1": "on", "true": "on", "yes": "on",
                 "0": "off", "false": "off", "no": "off", "": "off"}

FIELDS = (
    Field(
        "PYTHON_REPL_COMPUTE_ONLY", "off",
        {"on": (), "off": (CAP_FS_WRITE, CAP_PROC, CAP_NET)},
        "namespace",
        "off leaves the interpreter holding filesystem, process and network authority",
        aliases=_BOOL_ALIASES,
    ),
    Field(
        "ADMISSIBILITY_GATE", "off",
        {"on": (), "off": (CAP_UNADJUDICATED,)},
        "branch",
        "off means no envelope, no verdict and no ledger row for any action",
        aliases=_BOOL_ALIASES,
    ),
    Field(
        "ADMISSIBILITY_MODE", "shadow",
        {"enforce": (), "shadow": (CAP_UNADJUDICATED,)},
        "branch",
        "shadow produces verdicts that cannot stop anything",
    ),
    Field(
        "ADMISSIBILITY_FAIL", "open",
        {"closed": (), "open": (CAP_UNADJUDICATED,)},
        "branch",
        "open lets an action proceed when the adjudicator itself is broken",
    ),
    Field(
        "ADMISSIBILITY_ADAPTER", "noop",
        {"noop": (CAP_UNADJUDICATED,), "policy": (), "partner_a": (), "partner_b": (), "partner_c": ()},
        "branch",
        "noop always returns ALLOW, so the gate evaluates nothing",
    ),
    Field(
        "PARTNER_C_TIER_MIN", "reversible-bounded", {}, "none",
        "ceiling tier written into the partner envelope; free-form by design",
        open_valued=True,
    ),
    Field(
        "DEEPSEEK_MODEL", "deepseek-v4-flash", {}, "none",
        "model name; carries no authority",
        open_valued=True,
    ),
)

BY_NAME = {f.name: f for f in FIELDS}

# Faults observed during resolution. Populated by resolve(); read by startup_check() and emit().
_FAULTS = []


def _fault(kind, field, raw, resolved):
    rec = {"kind": kind, "field": field, "raw": raw, "resolved": resolved}
    if rec not in _FAULTS:
        _FAULTS.append(rec)
    return rec


def resolve(name):
    """Canonical value for one field, recording a fault when the raw value is not accepted.

    Deliberately still reads os.environ on every call. Freezing would break the callers that
    manipulate the environment in their own self-tests, and the defect being fixed is the SILENCE,
    not the re-read. `startup_check()` is what turns a fault into a refusal to start.
    """
    f = BY_NAME[name]
    raw = os.getenv(name)
    val = f.canon(raw)
    if val is None:
        if raw is None or (not f.open_valued and raw.strip() == ""):
            _fault("absent", name, raw, f.default)
        else:
            # The one that used to be silent: a real value that means nothing to us.
            _fault("invalid", name, raw, f.default)
        return f.default
    return val


def snapshot():
    """Every field resolved, plus the faults found doing it."""
    del _FAULTS[:]
    values = {f.name: resolve(f.name) for f in FIELDS}
    return {"values": values, "faults": list(_FAULTS)}


def capabilities(values=None):
    """The union of capabilities the system holds under a set of values."""
    values = values or snapshot()["values"]
    held = set()
    for f in FIELDS:
        held |= f.caps.get(values.get(f.name), frozenset())
    return sorted(held)


def manifest(values=None):
    """The compiled capability manifest. THIS is the security object, not the raw config.

    Hashing the config would be wrong: two machines with identical env values but different policy
    code would report the same hash while holding different authority. The hash therefore covers the
    compiled capability sets and the enforcement primitive behind each one.
    """
    values = values or snapshot()["values"]
    fields = []
    for f in FIELDS:
        v = values.get(f.name)
        prim, strength = STRENGTH[f.enforced_by]
        fields.append({
            "field": f.name,
            "value": v,
            "capabilities_held": sorted(f.caps.get(v, frozenset())),
            "fallback_grants_authority": f.fallback_grants_authority(),
            "enforced_by": prim,
            "strength": strength,
        })
    return {"schema": "policy-capability-manifest/1",
            "fields": fields,
            "capabilities_held": capabilities(values)}


def manifest_hash(values=None):
    blob = json.dumps(manifest(values), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def faults(snap=None):
    """(hard, soft). HARD = the run must not start. SOFT = start, but say so out loud.

    HARD is any fault whose fallback GRANTS authority, derived from the capability sets:
      - an invalid value that lands on an authority-granting default (the silent-disarm case)
      - an absent field whose default grants authority
    SOFT is everything else: an absent or invalid field whose default is neutral or restrictive.
    """
    snap = snap or snapshot()
    hard, soft = [], []
    for fl in snap["faults"]:
        (hard if BY_NAME[fl["field"]].fallback_grants_authority() else soft).append(fl)
    return hard, soft


def emit(stream=None):
    """Print the effective policy. The original defect was invisible because nothing ever did this."""
    out = stream or sys.stdout
    snap = snapshot()
    hard, soft = faults(snap)
    w = out.write
    w("[policy] env            : %s\n" % ENV_PATH)
    w("[policy] manifest       : %s\n" % manifest_hash(snap["values"]))
    for f in FIELDS:
        v = snap["values"][f.name]
        why = [fl for fl in snap["faults"] if fl["field"] == f.name]
        tag = ("  <- %s value, using default" % why[0]["kind"]) if why else ""
        w("[policy]   %-26s = %-22s%s\n" % (f.name, v, tag))
    w("[policy] capabilities   : %s\n" % (", ".join(capabilities(snap["values"])) or "none"))
    if soft:
        for fl in soft:
            w("[policy] NOTE  %s %s (raw=%r) -> %r; default does not grant authority\n"
              % (fl["kind"], fl["field"], fl["raw"], fl["resolved"]))
    if hard:
        for fl in hard:
            w("[policy] FAULT %s %s (raw=%r) -> %r; THIS DEFAULT GRANTS AUTHORITY\n"
              % (fl["kind"], fl["field"], fl["raw"], fl["resolved"]))
    return snap, hard, soft


class PolicyConfigError(RuntimeError):
    pass


def startup_check(strict=True, stream=None):
    """Call once at bootstrap. Emits the effective policy, then refuses to start on a hard fault.

    strict=False downgrades the refusal to a printed fault, for a caller that has decided
    availability matters more on that path. It is an explicit argument at the call site rather than
    an env var, deliberately: a permissive default read from configuration is the defect this whole
    module exists to remove, and it would be absurd to reintroduce it here.
    """
    snap, hard, soft = emit(stream)
    if hard and strict:
        raise PolicyConfigError(
            "refusing to start: %d security-relevant field(s) resolved to a default that grants "
            "authority: %s" % (len(hard), ", ".join("%s(%s)" % (f["field"], f["kind"]) for f in hard)))
    return snap


if __name__ == "__main__":
    _snap, _hard, _soft = emit()
    print()
    print("hard faults: %d   soft faults: %d" % (len(_hard), len(_soft)))
    sys.exit(1 if _hard else 0)
