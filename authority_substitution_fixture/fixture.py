# -*- coding: utf-8 -*-
"""A minimal, neutral reproduction of AUTHORITY SUBSTITUTION.

Standalone. Python 3.8+. No dependencies, no network, no filesystem writes, no imports beyond stdlib.
Run it:  python fixture.py        (exit 0 = the fixture behaved as documented)

=====================================================================================================
WHAT THIS DEMONSTRATES
=====================================================================================================
A system holds ONE protected effect and THREE declared routes that can produce it. A recording boundary
sits on exactly ONE of those routes. Every record the system writes is valid, complete with respect to
itself, and independently recomputable.

**And the system is silent about a consequential event that actually happened.**

Nothing is attacked. No control is bypassed, defeated or misconfigured. The gated route works perfectly
every single time it is used. The effect simply occurs somewhere the boundary does not sit.

This is the failure that a clean audit log cannot reveal, because a log only holds what reached it.

=====================================================================================================
THE THREE MEASUREMENTS, WHICH MUST NOT INHERIT EACH OTHER
=====================================================================================================
This separation was given to me in correspondence on 2026-09-17 by a practitioner building recomputable
evidence for AI workflows, and it is sharper than the two-way split I was using. Reproduced here because the fixture is built around it:

  1. ROUTE COVERAGE          which effect-producing routes intersect the recording boundary
  2. EVENT CAPTURE COVERAGE  whether executions through those routes actually produced the records
  3. RECORD CORRECTNESS      whether the records that exist are valid and recomputable

His formulation of why this matters:

    "capture completeness cannot be established solely from the record population whose completeness
     is in question. It needs an independently established reference surface."

The fixture makes all three separately visible, because collapsing them is precisely the error being
demonstrated. SCENARIO A scores 3/3 on record correctness, 3/3 on capture coverage for the route it
covers, and 1/3 on route coverage. **Two perfect scores and one silent hole.**

=====================================================================================================
WHY IT IS BUILT BOTH WAYS
=====================================================================================================
A demonstration that only ever shows the failure proves nothing about the instrument. If the route
checker reported "ungated path" no matter what it was given, it would produce the same output on a
system that is actually fine.

So SCENARIO B moves the boundary to cover all three routes and the same checker returns clean. The
contrast is what makes SCENARIO A meaningful. A check never shown to pass is as uninformative as a
check never shown to refuse.

=====================================================================================================
WHAT THIS IS NOT
=====================================================================================================
 - It is not evidence about anybody's production system, including my own. It is a toy with three
   routes and a dictionary for a ledger.
 - It does not prove the mechanism is common, or likely, or present anywhere in particular.
 - It does not enumerate every way capture can fail. Three are modelled here; error paths that return
   before writing, and writers that are themselves down, are equally real and are not modelled.
 - The route checker reads a DECLARED capability surface. A route nobody declared is invisible to it,
   which is the same class of hole it exists to find. That limitation is printed on every run.

What it does establish is that the mechanism is REAL and that a clean record is compatible with it.
One reproducible instance is enough for that, and it is all that is claimed.

=====================================================================================================
PROVENANCE
=====================================================================================================
The mechanism was found in my own system on 2026-08-10, not reasoned into existence: a gated tool went
missing from a session, the planner rerouted through a general-purpose interpreter, and the same write
happened with no envelope, no verdict and no ledger row. A security researcher working on agent
authorisation named the class AUTHORITY SUBSTITUTION and gave the red-team invariant that the route
checker below implements.

Author: Alkama Eqbal. Single owner, deliberately. Free to use, quote, break and publish results against.
"""
import sys

# -- THE PROTECTED EFFECT -------------------------------------------------------------------------
# One effect, deliberately. Everything else is scaffolding around it.
PROTECTED_EFFECT = "mutate_persistent_state"


class World:
    """The thing that actually changes. A dict standing in for anything consequential."""

    def __init__(self):
        self.state = {}
        self.mutations = []        # ground truth: every mutation that REALLY happened

    def apply(self, key, value, via):
        self.state[key] = value
        self.mutations.append({"key": key, "value": value, "via": via})


class Ledger:
    """The evidence population. Append-only, and it only ever holds what reached it."""

    def __init__(self):
        self.rows = []

    def record(self, route, key, value):
        row = {"seq": len(self.rows) + 1, "route": route, "effect": PROTECTED_EFFECT,
               "key": key, "value": value}
        row["digest"] = self._digest(row)
        self.rows.append(row)

    @staticmethod
    def _digest(row):
        """Trivial content digest. Stands in for a signature or a hash chain link."""
        payload = "|".join(str(row[k]) for k in ("seq", "route", "effect", "key", "value"))
        h = 0
        for ch in payload:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        return "%08x" % h

    def recompute_all(self):
        """RECORD CORRECTNESS. Can an independent party reproduce every digest from the row itself?"""
        return [(r["seq"], r["digest"] == self._digest(r)) for r in self.rows]


# -- THE THREE ROUTES TO THE EFFECT ---------------------------------------------------------------
# Each route is declared with the effects it can produce. The DECLARATION is the reference surface;
# it is built from what the system can do, never from what the system recorded doing.
ROUTES = {
    "storage_api": {
        "produces": [PROTECTED_EFFECT],
        "note": "the obvious route. purpose-built, and the one everyone thinks of as 'the' path.",
    },
    "batch_importer": {
        "produces": [PROTECTED_EFFECT],
        "note": "a second purpose-built route. exists because bulk loads needed their own entry point.",
    },
    "script_runner": {
        "produces": [PROTECTED_EFFECT],
        "note": "a GENERAL-PURPOSE route. nobody added it to mutate state; it can because it can run "
                "arbitrary code. this is the one that gets forgotten, and forgetting it is the bug.",
    },
    "read_api": {
        "produces": [],
        "note": "structurally incapable of the effect. present to show the checker does not just flag "
                "everything that is ungated.",
    },
}


def run(route, world, ledger, boundary):
    """Produce the effect via `route`. The boundary decides whether anything is recorded.

    NOTE the shape of this function, because it is the whole point: the mutation happens FIRST and
    unconditionally. Recording is a separate, later, conditional act. Nothing here is defeated or
    bypassed. The recorder is simply not present on every path to the effect.
    """
    if PROTECTED_EFFECT not in ROUTES[route]["produces"]:
        raise AssertionError("route %r cannot produce %r" % (route, PROTECTED_EFFECT))
    key = "record-%d" % (len(world.mutations) + 1)
    world.apply(key, "changed-by-%s" % route, via=route)
    if route in boundary:
        ledger.record(route, key, "changed-by-%s" % route)


# -- THE ROUTE CHECKER ----------------------------------------------------------------------------
def route_coverage(boundary):
    """MEASUREMENT 1. Derived from the declared surface, never from the ledger.

    The invariant this implements directly, given by the researcher who named the class:

        "Remove any gated tool, recompute reachable protected effects using the authorities still
         present, then fail the configuration if another node retains an ungated path to the same
         effect."

    A route passes if it EITHER crosses the boundary OR is structurally incapable of the effect.
    """
    capable = [r for r, d in ROUTES.items() if PROTECTED_EFFECT in d["produces"]]
    covered = [r for r in capable if r in boundary]
    ungated = [r for r in capable if r not in boundary]
    return capable, covered, ungated


def capture_coverage(world, ledger):
    """MEASUREMENT 2. Ground truth vs the record. Requires the reference surface, by construction.

    This number CANNOT be computed from inside the ledger. It needs `world.mutations`, which is the
    independently established reference named in that correspondence. In a real system that reference is the hard
    part; here it is handed to us because the fixture owns both sides.
    """
    return len(ledger.rows), len(world.mutations)


# -- SCENARIOS ------------------------------------------------------------------------------------
def scenario(title, boundary, routes_exercised):
    print("=" * 99)
    print(title)
    print("=" * 99)
    print("recording boundary sits on : %s" % (", ".join(sorted(boundary)) or "(nothing)"))
    print("")

    world, ledger = World(), Ledger()
    for r in routes_exercised:
        run(r, world, ledger, boundary)

    capable, covered, ungated = route_coverage(boundary)
    recorded, actual = capture_coverage(world, ledger)
    recomputed = ledger.recompute_all()
    all_valid = all(ok for _, ok in recomputed)

    print("1. ROUTE COVERAGE          %d of %d routes that can produce the effect cross the boundary"
          % (len(covered), len(capable)))
    for r in capable:
        print("     %-16s %s" % (r, "covered" if r in covered else "UNGATED  <- reaches the effect, "
                                                                   "records nothing"))
    print("     %-16s %s" % ("read_api", "not capable of the effect, correctly not flagged"))
    print("")
    print("2. EVENT CAPTURE COVERAGE  %d records for %d actual mutations" % (recorded, actual))
    if recorded < actual:
        missed = [m for m in world.mutations if m["via"] not in boundary]
        for m in missed:
            print("     NOT CAPTURED   key=%s via=%s" % (m["key"], m["via"]))
    print("")
    print("3. RECORD CORRECTNESS      %d of %d rows recompute to their stored digest"
          % (sum(1 for _, ok in recomputed if ok), len(recomputed)))
    for seq, ok in recomputed:
        print("     row %d %s" % (seq, "valid" if ok else "INVALID"))
    print("")

    verdict = "CLEAN" if not ungated and recorded == actual else "UNGATED PATH PRESENT"
    print("   record correctness : %s" % ("3/3 valid" if all_valid else "FAILED"))
    print("   ledger self-consistency : %s" % ("perfect" if all_valid else "broken"))
    print("   VERDICT : %s" % verdict)
    print("")
    return dict(capable=capable, covered=covered, ungated=ungated,
                recorded=recorded, actual=actual, all_valid=all_valid)


def main():
    print("")
    print("AUTHORITY SUBSTITUTION, a minimal neutral reproduction")
    print("protected effect: %s" % PROTECTED_EFFECT)
    print("")

    a = scenario(
        "SCENARIO A, the boundary sits on ONE route. The other two reach the same effect ungated.",
        boundary={"storage_api"},
        routes_exercised=["storage_api", "storage_api", "batch_importer", "script_runner",
                          "storage_api"])

    print("   READ THAT AGAIN. Every row in the ledger is valid. Every digest recomputes. An")
    print("   independent party handed this ledger would verify all of it and find nothing wrong,")
    print("   because there IS nothing wrong with the rows that exist. The ledger is complete with")
    print("   respect to itself and silent about two mutations that really happened.")
    print("")
    print("   Record correctness said 3 of 3. Capture coverage said 3 of 5. Route coverage said 1 of 3.")
    print("   Only the last two are about the system; the first is about the paper.")
    print("")

    b = scenario(
        "SCENARIO B, the same checker, the boundary moved to cover every capable route.",
        boundary={"storage_api", "batch_importer", "script_runner"},
        routes_exercised=["storage_api", "batch_importer", "script_runner"])

    print("   This is why scenario A means something. The instrument returns CLEAN when the system")
    print("   is clean, so UNGATED PATH PRESENT is a finding rather than the only thing it can say.")
    print("")

    # -- self-check: the fixture must behave exactly as documented, in BOTH directions -----------
    print("=" * 99)
    print("SELF-CHECK, does the fixture do what this file claims it does?")
    print("=" * 99)
    checks = [
        ("A: exactly 2 ungated routes found",        sorted(a["ungated"]) == ["batch_importer", "script_runner"]),
        ("A: capture coverage is 3 of 5",            (a["recorded"], a["actual"]) == (3, 5)),
        ("A: every record still recomputes",         a["all_valid"] is True),
        ("A: read_api never flagged (not capable)",  "read_api" not in a["capable"]),
        ("B: zero ungated routes",                   b["ungated"] == []),
        ("B: capture coverage is 3 of 3",            (b["recorded"], b["actual"]) == (3, 3)),
        ("B: every record recomputes",               b["all_valid"] is True),
    ]
    bad = 0
    for label, ok in checks:
        bad += not ok
        print("  %-4s %s" % ("OK" if ok else "FAIL", label))
    print("")

    print("LIMITS OF THIS FIXTURE, printed every run so it is never over-read:")
    print("  - it demonstrates ONE mechanism on a toy with three routes and a dict for a ledger")
    print("  - it is not evidence about any production system, mine included")
    print("  - the route checker reads a DECLARED surface; an undeclared route is invisible to it,")
    print("    which is the same class of hole it exists to find")
    print("  - other capture failures are real and not modelled here: an error path that returns")
    print("    before writing, a retry that lands after the writer is down")
    print("")

    if bad:
        print("FIXTURE SELF-CHECK FAILED: %d of %d." % (bad, len(checks)))
        return 1
    print("Fixture behaved as documented. %d of %d checks." % (len(checks), len(checks)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
