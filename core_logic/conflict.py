from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Conflict:
    type: str       # "physical" | "logical" | "temporal"
    task_a: str     # id of the candidate task
    task_b: str     # id of the running task it conflicts with
    reason: str     # human-readable description
    severity: str   # "hard" | "soft"


@dataclass
class ArbitrationResult:
    decision: str   # "dispatch" | "defer" | "reorder" | "notify_user"
    reason: str     # human-readable explanation


# ---------------------------------------------------------------------------
# ConflictDetector
# ---------------------------------------------------------------------------

class ConflictDetector:
    """
    Stateless detector. Compares a candidate task against a list of currently
    running tasks using the resource/reads/writes context convention.

    Tasks that do not declare these keys have empty sets and are always
    treated as conflict-free (correct opt-in default).
    """

    def check(self, candidate, running: list) -> list:
        """
        Check candidate task against all currently running tasks.
        Returns a list of Conflict objects (empty = no conflicts).
        """
        conflicts = []
        c_resources = set(candidate.context.get("resources", []))
        c_writes    = set(candidate.context.get("writes",    []))
        c_reads     = set(candidate.context.get("reads",     []))

        for active in running:
            a_resources = set(active.context.get("resources", []))
            a_writes    = set(active.context.get("writes",    []))
            a_reads     = set(active.context.get("reads",     []))

            # Physical: shared exclusive resource
            shared_resources = c_resources & a_resources
            if shared_resources:
                conflicts.append(Conflict(
                    type="physical",
                    task_a=candidate.id,
                    task_b=active.id,
                    reason=f"Shared resource(s): {shared_resources}",
                    severity="hard",
                ))

            # Logical: both tasks write the same key (write-write conflict)
            shared_writes = c_writes & a_writes
            if shared_writes:
                conflicts.append(Conflict(
                    type="logical",
                    task_a=candidate.id,
                    task_b=active.id,
                    reason=f"Both write to: {shared_writes}",
                    severity="hard",
                ))

            # Temporal: candidate reads what active writes (read stale data risk)
            read_write_overlap = c_reads & a_writes
            if read_write_overlap:
                conflicts.append(Conflict(
                    type="temporal",
                    task_a=candidate.id,
                    task_b=active.id,
                    reason=f"Candidate reads {read_write_overlap} which active task writes",
                    severity="soft",
                ))

            # Temporal: candidate writes what active reads (active may read stale data)
            write_read_overlap = c_writes & a_reads
            if write_read_overlap:
                conflicts.append(Conflict(
                    type="temporal",
                    task_a=candidate.id,
                    task_b=active.id,
                    reason=f"Candidate writes {write_read_overlap} which active task reads",
                    severity="soft",
                ))

        return conflicts


# ---------------------------------------------------------------------------
# ArbitrationEngine
# ---------------------------------------------------------------------------

class ArbitrationEngine:
    """
    Stateless decision engine. Given a candidate task and its conflicts,
    returns an ArbitrationResult telling the Orchestrator what to do.
    """

    def arbitrate(self, candidate, conflicts: list, running: list) -> ArbitrationResult:
        if not conflicts:
            return ArbitrationResult(decision="dispatch", reason="No conflicts.")

        hard_conflicts = [c for c in conflicts if c.severity == "hard"]
        soft_conflicts = [c for c in conflicts if c.severity == "soft"]

        if hard_conflicts:
            conflicting_ids   = {c.task_b for c in hard_conflicts}
            conflicting_tasks = [t for t in running if t.id in conflicting_ids]

            all_lower_priority = all(
                t.priority < candidate.priority for t in conflicting_tasks
            )
            if all_lower_priority:
                # Candidate wins on priority — interrupt model handles pausing
                return ArbitrationResult(
                    decision="dispatch",
                    reason=(
                        f"Hard conflict exists but candidate has higher priority. "
                        f"Interrupt model will pause conflicting tasks."
                    ),
                )

            # Candidate is lower or equal priority
            if candidate.origin == "system":
                return ArbitrationResult(
                    decision="defer",
                    reason=(
                        f"Hard conflict with running task(s). "
                        f"System task deferred — will retry next tick."
                    ),
                )

            # User task vs equal/higher-priority running task — cannot silently defer
            return ArbitrationResult(
                decision="notify_user",
                reason=(
                    f"Hard conflict: {hard_conflicts[0].reason}. "
                    f"Cannot execute simultaneously with active task."
                ),
            )

        # Soft conflicts only (temporal ordering preference)
        if soft_conflicts:
            if candidate.origin == "system":
                return ArbitrationResult(
                    decision="defer",
                    reason=(
                        f"Soft temporal conflict. "
                        f"System task deferred to preserve ordering."
                    ),
                )
            # User tasks: dispatch with a soft-conflict note — not a hard block
            return ArbitrationResult(
                decision="dispatch",
                reason=(
                    f"Soft temporal conflict noted but user task dispatched. "
                    f"Conflict: {soft_conflicts[0].reason}"
                ),
            )

        return ArbitrationResult(decision="dispatch", reason="No blocking conflicts.")
