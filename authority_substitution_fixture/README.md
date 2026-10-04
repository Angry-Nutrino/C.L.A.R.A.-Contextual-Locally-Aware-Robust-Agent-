# Authority Substitution: a minimal neutral reproduction

```
python fixture.py
```

No dependencies. No network. No filesystem writes. Python 3.8+. Exit 0 means the fixture behaved as
documented, in both directions.

---

## The four things this hands you

Requested in exactly this form, so it is answered in exactly this form.

### 1. The effect

One protected effect: `mutate_persistent_state`. Anything consequential and irreversible-ish stands in
here; the fixture uses a dict so the mechanism is visible without any infrastructure.

### 2. The available routes

Four declared routes. **Three can produce the effect. One structurally cannot.**

| route | produces the effect | why it exists |
|---|---|---|
| `storage_api` | yes | the obvious path, purpose-built, the one everyone means by "the" path |
| `batch_importer` | yes | a second purpose-built path, added because bulk loads needed their own entry |
| `script_runner` | yes | **general-purpose.** Nobody added it to mutate state. It can, because it runs arbitrary code |
| `read_api` | **no** | present so a pass means something. An ungated route that cannot produce the effect is not a finding |

`script_runner` is the one that matters. **It was never designed as a path to the effect**, which is
exactly why it does not appear on anyone's list of paths to the effect.

### 3. The expected recording boundary

**Scenario A:** the boundary sits on `storage_api` only.
**Scenario B:** the boundary sits on all three capable routes.

Same code, same checker, one variable moved.

### 4. The resulting evidence population

Scenario A, after five real mutations:

```
3 rows in the ledger
3 of 3 recompute to their stored digest
0 rows are malformed, out of order, or unverifiable
2 mutations that actually happened appear nowhere
```

**The ledger is not defective.** Every row in it is correct. An independent party handed this ledger
verifies all of it and finds nothing wrong, because there is nothing wrong with the rows that exist.

---

## The three measurements, separated

The fixture reports each one independently, because collapsing them is the error it exists to expose.

| | scenario A | scenario B |
|---|---|---|
| **1. route coverage**, which effect-producing routes cross the boundary | **1 of 3** | 3 of 3 |
| **2. event capture coverage**, did executions actually produce records | **3 of 5** | 3 of 3 |
| **3. record correctness**, are the records that exist valid and recomputable | **3 of 3** | 3 of 3 |

**Measurement 3 is perfect in the failing scenario.** That is the entire point. Record correctness is a
statement about the paper. Route coverage and capture coverage are statements about the system, and only
the second two moved.

**Measurement 2 cannot be computed from inside the ledger.** It needs `world.mutations`, which is ground
truth held outside the record. In the fixture that is free because it owns both sides. In a real system
establishing that reference is the hard part, and measurement 1 is one way to build it, because it is
derived from what the system can do rather than from what it recorded doing.

---

## Why scenario B exists

A demonstration that only ever shows a failure proves nothing about the instrument. A checker that
printed "ungated path" unconditionally would produce identical output on a system that is genuinely
fine, and would be worthless.

**Scenario B moves the boundary to cover every capable route and the same checker returns CLEAN.** The
contrast is what makes scenario A a finding rather than the only thing the tool can say.

---

## Limits, printed on every run

- It demonstrates **one** mechanism on a toy with three routes and a dict for a ledger.
- **It is not evidence about any production system, mine included.** One reproducible instance
  establishes that the mechanism is real and that a clean record is compatible with it. That is all it
  claims and all it can claim.
- The route checker reads a **declared** capability surface. **A route nobody wrote down is invisible to
  it**, which is the same class of hole it exists to find, pointing the same direction. Worth naming
  before anyone leans on it as a reference surface.
- Other capture failures are equally real and **not modelled here**: an error path that returns before
  writing, a retry that lands after the writer is down, a queue that drops under load.

---

## Where it came from

The mechanism was not reasoned into existence. It was found in a live system on **2026-08-10**: a gated
tool became unavailable within a session, the planner rerouted through a general-purpose interpreter, and
the same write happened with no envelope, no verdict and no ledger row. Nothing was attacked and nothing
failed. The action occurred somewhere the gate did not sit.

**A security researcher working on agent authorisation** named the class *authority substitution* and
supplied the red-team invariant that `route_coverage()` implements directly:

> Remove any gated tool, recompute reachable protected effects using the authorities still present,
> then fail the configuration if another node retains an ungated path to the same effect.

**Ryan Feller** supplied the three-way separation the fixture is organised around, in correspondence in
September 2026, and sharpened the completeness claim:

> capture completeness cannot be established solely from the record population whose completeness is in
> question. It needs an independently established reference surface.

*Named here with his written permission, granted 2026-09-17, and bounded at his request to the three-way
split and the completeness sharpening. The fixture itself is not his and he declined any share of it.*

The security researcher above **is described by role rather than named because they have not been asked
whether they want to appear in a public repository.** They are credited by name in the copy they receive,
and the name goes into this file the moment they say yes. Silence is not consent and an unasked person is
not a declining one.

---

## Ownership

Written and owned by **Alkama Eqbal**. Single owner, deliberately, so there is no ambiguity about whose
artifact it is and it can be published, quoted and broken by anyone.

Use it, pressure it, and publish what you find. **If it breaks, that is the point.**
