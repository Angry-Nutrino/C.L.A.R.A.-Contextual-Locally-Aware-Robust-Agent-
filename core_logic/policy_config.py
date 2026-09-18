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
    # Added 2026-09-18 with PYTHON_REPL_ISOLATION. The FIRST non-behavioural primitive here, and
    # the scope is written into the label on purpose. A Windows restricted token denies the
    # filesystem write in the kernel, so a language-level escape that reaches the real open() is
    # refused anyway (measured: the escape succeeds, the write returns PermissionError). It denies
    # WRITES ONLY. Reads, network egress and process creation were all measured as still working,
    # which is why the field's capability set keeps CAP_PROC and CAP_NET.
    "process": ("a Windows restricted token on a separate process",
                "os-enforced (filesystem writes only)"),
    "none": ("nothing; this value is descriptive only", "none"),
}


class Field(object):
    """One security-relevant setting, its allowed values, and the capability set each value implies.

    `caps` maps an allowed value -> frozenset of capabilities the system HOLDS at that value.
    The authority direction of the fallback is derived from these sets, never declared.
    """

    def __init__(self, name, default, caps, enforced_by, note, aliases=None, open_valued=False,
                 values_fn=None, unknown_caps=None):
        self.name = name
        self.default = default
        self.caps = {k: frozenset(v) for k, v in caps.items()}
        self.enforced_by = enforced_by
        self.note = note
        self.aliases = aliases or {}          # raw value -> canonical value
        self.open_valued = open_valued        # free-form string, cannot be enumerated

        # ── BRIEF_63 option B, implemented 2026-09-18 ────────────────────────────────────────
        # Some fields have a legal value set that is only known at RUN time. ADMISSIBILITY_ADAPTER
        # is the case: adapters register themselves into a live registry, so a static list here is
        # wrong the moment anyone adds one, and canonicalising against the static list rejects a
        # perfectly valid adapter and silently falls back to the default. The default is `noop`,
        # and noop always returns ALLOW, so that fallback DISABLES ADJUDICATION ENTIRELY. That is
        # the exact silence BRIEF_62 was written to end, which is why option A (open-valuing the
        # field and throwing the capability model away) was rejected in favour of this.
        #
        # `values_fn` is a callback the OWNING MODULE registers, so policy_config still knows
        # nothing about its callers at import time. Until it is registered the static list applies,
        # which is the correct conservative behaviour during early startup.
        self.values_fn = values_fn

        # Capabilities for a value that only exists at run time cannot be declared in advance, so
        # they are ASSUMED, and the assumption is the pessimistic one: an adapter we know nothing
        # about might not adjudicate at all. Anything else would let a dynamically registered
        # adapter quietly look safer than the one it replaced.
        self.unknown_caps = frozenset(unknown_caps or ())

    # -- the live legal value set ----------------------------------------------------------------
    def legal_values(self):
        """Every value this field accepts right now: the declared ones plus any registered live."""
        vals = set(self.caps)
        if self.values_fn is not None:
            try:
                vals |= {str(v).strip().lower() for v in self.values_fn() if str(v).strip()}
            except Exception:
                pass          # a broken callback must never make a governed field unreadable
        return vals

    def caps_for(self, value):
        """Capabilities held at `value`. Declared values use the table; a value that exists only in
        the live registry gets the pessimistic assumption above."""
        if value in self.caps:
            return self.caps[value]
        return self.unknown_caps

    # -- normalisation ---------------------------------------------------------------------------
    def canon(self, raw):
        """Canonical value, or None if the raw string is not a value this field accepts."""
        if raw is None:
            return None
        v = raw.strip().lower()
        v = self.aliases.get(v, v)
        if self.open_valued:
            return raw.strip() or None
        return v if v in self.legal_values() else None

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
        # G38 / BRIEF_60, added 2026-09-18. The OS-enforced sibling of PYTHON_REPL_COMPUTE_ONLY.
        # The capability sets are written from MEASUREMENT, not from intent: the sandbox was probed
        # on 2026-09-18 and filesystem writes were refused (PermissionError) while process creation
        # and network egress both still SUCCEEDED. So `on` drops CAP_FS_WRITE and keeps the other
        # two. Writing () here would overclaim the boundary in the one file that is supposed to be
        # the honest record of what each value actually grants.
        "PYTHON_REPL_ISOLATION", "off",
        {"on": (CAP_PROC, CAP_NET),
         "shadow": (CAP_FS_WRITE, CAP_PROC, CAP_NET),
         "off": (CAP_FS_WRITE, CAP_PROC, CAP_NET)},
        "process",
        "off and shadow both execute in-process, where a namespace escape reaches the real open()",
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
        # BRIEF_63, resolved 2026-09-18 with option B. This is the one field whose legal values are
        # only fully known at run time: adapters register into a live registry, so a static list
        # here rejects a valid adapter and falls back to `noop`, and `noop` always returns ALLOW.
        # `admissibility.py` calls register_values_fn() at import to hand over the live set.
        "ADMISSIBILITY_ADAPTER", "noop",
        {"noop": (CAP_UNADJUDICATED,), "policy": (), "partner_a": (), "partner_b": (), "partner_c": ()},
        "branch",
        "noop always returns ALLOW, so the gate evaluates nothing",
        # An adapter registered at run time carries no declared capability set, so it is assumed to
        # be the permissive case. An unknown adapter must never look safer than a declared one.
        unknown_caps=(CAP_UNADJUDICATED,),
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


def register_values_fn(name, fn):
    """Let the module that OWNS a dynamic registry tell policy_config how to read it.

    BRIEF_63 option B. The coupling runs one way only: the owner calls in, and this module still
    imports nothing from its callers. Called once at import time by the owning module; before that
    call the static list applies, which is the conservative behaviour during early startup.
    """
    BY_NAME[name].values_fn = fn

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
        # caps_for, not caps.get: a value that exists only in the live registry must still
        # contribute its (pessimistic) capability set instead of contributing nothing.
        held |= f.caps_for(values.get(f.name))
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
