# Research: Mid-Execution Resource Conflict Detection in Concurrent ReAct-Loop Multi-Agent Systems

**Date:** 2026-05-10
**Objective:** Determine what prior art exists for detecting and resolving resource conflicts that emerge *during* (not at dispatch-time) concurrent agent task execution — specifically: whether any system can pause a running ReAct loop turn mid-execution when a runtime conflict with another task is discovered, and assess the novelty of this problem space.
**Target Context:** Python asyncio, single-machine deployment, consumer GPU (RTX 3050 Mobile), side-effectful tool calls (file writes, shell commands) that cannot be rolled back, concurrent ReAct loops dispatched by an orchestrator queue.

---

## 1. Overview

The core problem is a timing gap: at the moment two agent tasks are dispatched, neither the orchestrator nor the tasks themselves know what filesystem or tool resources will be touched. That information only emerges during execution, as the LLM generates Action calls and the tool executor runs them. By the time a conflict is detectable — Task A is writing `config.json` in turn 4 while Task B is reading it in turn 2 — both loops are already running as live asyncio coroutines with no external pause mechanism.

Classical OS concurrency theory (mutexes, semaphores) and database concurrency control (STM, OCC) offer the intellectual scaffolding, but LLM agent tool calls introduce constraints that make direct application difficult: tool calls are long-running and non-atomic, their side effects are often irreversible (file writes, shell commands), and the loop's "critical section" is not a tightly bounded memory transaction but a streaming LLM response that may contain multiple tool calls interleaved with reasoning tokens.

This research surveyed academic papers (arXiv, ACL 2025), framework documentation (LangGraph, AutoGen, CrewAI, Magentic-One, AgentScope, OpenAI Assistants), practitioner blogs (Anthropic Engineering, Restate, Temporal, Inngest), and GitHub issues/discussions to establish the state of the art as of May 2026.

---

## 2. How the Problem Manifests Technically

A CLARA-style orchestrator runs two asyncio tasks concurrently via `asyncio.create_task()`. Each task is an independent invocation of `run_task()`, which internally drives a ReAct loop: LLM stream → parse Action → execute tool → append observation → repeat. The loop is a single `async def` coroutine. The tool executor calls `await mcp_client.call(server, tool_name, args)` per tool invocation.

The conflict emerges when:
1. Task A's tool executor calls `write_file("config.json", new_content)` — a side-effectful operation.
2. Task B's tool executor, running concurrently on a different asyncio task, is about to call `read_file("config.json")`.
3. Task B reads a file that is mid-write, producing corrupt or inconsistent content.
4. Neither task knows about the other's intent at the moment of conflict.

A resource ledger can record `{path → (task_id, operation)}` at tool-call time, but the standard `asyncio.Lock` on a path only blocks new acquisitions — it does not interrupt a coroutine that is already `await`-ing inside the loop turn. The problem is not preventing new tool calls from starting; it is interrupting or deferring an already-in-flight one.

---

## 3. What Existing Frameworks Do

### 3.1 LangGraph

LangGraph provides two relevant mechanisms: static breakpoints (`interrupt_before` on named nodes) and a dynamic `interrupt()` function callable anywhere inside a node. The `interrupt()` call persists graph state via a checkpointer and blocks until resumed by an external `Command`. This is the closest existing mechanism to a mid-turn pause.

**Key finding:** LangGraph's `interrupt()` is designed for human-in-the-loop approval flows — a human sees a pending action and approves or rejects it. It is not designed to pause one graph instance based on a runtime conflict detected in *a different, concurrently running* graph instance. There is no cross-instance signaling mechanism. Two simultaneous graph executions in LangGraph are isolated; one cannot observe or pause the other. A GitHub issue (December 2025, #6626) documents that `interrupt()` calls in parallel tools generate identical IDs, making multi-interrupt resume impossible — indicating the parallel execution path is still immature.

**Source:** LangChain official docs (interrupts.html), GitHub issues #6626 and #6624. [HIGH confidence]

### 3.2 AutoGen (v0.4+ Actor Model)

AutoGen 0.4 migrated to an actor model in early 2024. Each agent is an actor that processes messages from a mailbox serially. This prevents internal agent state corruption but does not prevent two actors from concurrently calling the same external tool. The actor model serializes *intra-agent* execution, not *inter-agent* tool calls.

A GitHub discussion (#7144) documents practitioner solutions: "locked blackboard" patterns (propose → validate → commit with priority-based preemption), eventual consistency with short TTLs, and agent-local vs. team-visible state separation. These are application-level conventions, not framework primitives. None pause a running actor mid-message-processing.

**Source:** GitHub microsoft/autogen discussion #7144, AutoGen documentation. [HIGH confidence]

### 3.3 CrewAI

CrewAI version 1.10.2a1 (2025) fixed a `LockException` under multi-process concurrency during parallel task execution. This confirms the framework was experiencing concurrent tool access bugs in production. The fix addresses lock acquisition on the process level, not intra-turn mid-execution suspension. Task-level concurrency in CrewAI is controlled by `async_execution` flags on tasks, but conflict detection between concurrently executing tasks is left to the application.

**Source:** CrewAI changelog, toolnavs.com release notes. [MEDIUM confidence — limited documentation detail]

### 3.4 Magentic-One (Microsoft)

Magentic-One uses a sequential Orchestrator design: the Orchestrator selects "which agent speaks next" and agents run one at a time by default. Its architecture avoids the concurrency conflict problem by not running agents truly in parallel within a task. This is a deliberate tradeoff — simplicity over throughput. After each plan update, "all agents clear their contexts and reset their states," serializing execution windows.

**Source:** arXiv:2411.04468v1 (Magentic-One paper). [HIGH confidence]

### 3.5 AgentScope

AgentScope Runtime v1.1.0 (2026) introduced a Distributed Interrupt Service enabling "manual task preemption during agent execution." This is the closest existing production feature to mid-turn pause. However, it is triggered manually (by the developer or a supervisor agent), not automatically based on runtime conflict detection with another task. The interrupt is implemented via asyncio cancellation, which raises `CancelledError` into the coroutine — a destructive operation, not a suspend-and-resume.

**Source:** AgentScope documentation, agentscope-runtime PyPI. [MEDIUM confidence]

### 3.6 OpenAI Assistants API

The Assistants API locks the entire Thread when a Run is `in_progress`. No new Messages can be added, and no new Runs can be created on the same Thread. This is a coarse-grained mutex at the conversation level, not at the resource/file level. Two agents on different Threads can conflict on external resources with no detection.

**Source:** OpenAI platform documentation (runs endpoint). [HIGH confidence]

### 3.7 Anthropic's Multi-Agent Research System

Anthropic's published engineering article explicitly describes synchronous subagent execution: "the LeadResearcher waits for subagents to finish before moving forward." This avoids the concurrent conflict problem by serializing execution. The article acknowledges that asynchronous execution "could remove these bottlenecks though it would add complexity in managing state, coordinating results, and handling errors" — confirming the problem is recognized but unsolved even internally.

**Source:** anthropic.com/engineering/multi-agent-research-system. [HIGH confidence]

---

## 4. Academic Prior Art

### 4.1 Atomix (arXiv:2602.14849, February 2026) — Closest Prior Art

**The most directly relevant paper found.** Atomix is a runtime shim providing epoch-based transactional semantics for LLM agent tool calls. Key mechanisms:

- **Epoch tagging:** Every tool call receives a logical timestamp (epoch). Operations are totally ordered.
- **Resource frontier tracking:** Each resource maintains a frontier value `f(r)`. A transaction at epoch `e` can commit only when `frontier(r) ≥ e` for all resources in scope — meaning all earlier work has completed.
- **Effect taxonomy:** Effects are classified as bufferable (file writes, delayed until commit) or externalized (irreversible API calls, tracked with compensation). Buffered writes are invisible until commit; externalized effects require compensation handlers on abort.
- **Mid-execution blocking:** If a frontier has not advanced (because a competing agent's work is incomplete), the transaction's `can_commit()` check blocks the visibility of that agent's effects. The paper states: "Atomix prioritizes safety over liveness: partitions or crashes result in stalls rather than inconsistency."

**What Atomix does NOT do:** It blocks *effect visibility*, not *loop execution*. The LLM generation loop continues generating tokens; the effects are buffered. The agent is not suspended mid-turn; its writes are held in a buffer until the commit gate opens. This is a transactional memory pattern applied to tool outputs, not a ReAct loop suspension mechanism.

**Results:** 37–57% task success under 30% per-call fault injection vs. 0–7% for baselines. Zero contamination from aborted branches in speculation experiments. Zero irreversible effect leaks.

**Limitations:** Single-process only (no distributed frontier store). In-memory deduplication (not crash-safe). Frontier staleness if an agent crashes without completing — requires orchestrator-side timeouts. File writes can be buffered (local), but remote API calls must externalize immediately and require compensation.

**Source:** arXiv:2602.14849, GitHub mpi-dsg/atomix. [HIGH confidence]

### 4.2 Semantic Consensus / Semantic Intent Divergence (arXiv:2604.16339, April 2026)

Proposes a pre-execution Semantic Intent Graph (SIG) that captures agent intentions before any action is taken, analyzes overlapping entity impacts, and detects three conflict types: direct contradictions, resource contention (multiple agents claiming exclusive access), and ordering dependencies. A three-tier resolution protocol (organizational policy > capability authority > temporal priority) resolves detected conflicts.

**Key limitation:** Pre-execution only. The framework intercepts agent decisions *before execution*, preventing conflicts from manifesting. It cannot handle conflicts that emerge mid-execution from tool calls not predicted at planning time. Also explicitly flagged: "The framework struggles with traditional file I/O... cannot fully predict outcomes when multiple agents interact with actual filesystem state."

**Source:** arXiv:2604.16339. [HIGH confidence]

### 4.3 HiveMind (arXiv:2604.17111, April 2026)

Applies OS-inspired scheduling primitives to concurrent LLM coding agents sharing a rate-limited API endpoint. The five primitives: admission control (condition variables), rate-limit tracking (sliding window counters), AIMD backpressure with circuit breaking, token budget management (per-agent ceilings from a global pool), and priority queuing with dependency DAGs.

**Relevance to the specific problem:** HiveMind is about API-level resource contention (rate limits, token budgets), not filesystem or tool-level conflicts. It does not pause individual running agent turns mid-execution. The circuit breaker fast-fails incoming requests but lets in-flight requests complete. HiveMind's own paper notes: "Filesystem operations and tool calls remain unsynchronized between agents."

**Results:** Reduces agent failure rates from 72–100% to 0–18% under API contention. Under 3ms proxy overhead.

**Source:** arXiv:2604.17111. [HIGH confidence]

### 4.4 MegaAgent (arXiv:2408.09955, ACL 2025 Findings)

A large-scale system (up to 590 agents). Implements a **Git-based version control mechanism** for concurrent file access: when an agent reads a file, it retrieves the current Git commit hash. Before modifying, it submits the hash to a file management system which commits updates, merges into HEAD, and prompts the agent to resolve merge conflicts. "All Git operations are serialized using a global mutex lock."

**Relevance:** This is the only production multi-agent system found that explicitly handles same-file concurrent write conflicts between agents. However, the mutex is at the Git-operation level — the agent's ReAct loop is not suspended; instead, the write operation itself blocks at the mutex until the previous write completes, and then the agent may need to re-read and re-resolve a conflict. The loop keeps running; the conflict is surfaced as a Git merge conflict that the agent must reason about.

**Source:** arXiv:2408.09955v3, ACL 2025. [HIGH confidence]

### 4.5 CodeCRDT (arXiv:2510.18893, October 2024)

Applies Conflict-free Replicated Data Types (CRDTs) to multi-agent code generation. Agents work independently on shared code files; changes are merged using CRDT commutativity at merge time. Conflict detection is at merge time, not runtime. No mid-execution suspension. Showed +25% runtime improvement but -7.7% code quality reduction.

**Source:** arXiv:2510.18893. [MEDIUM confidence — limited full-text access]

### 4.6 AgentSpec (arXiv:2503.18666, March 2025)

Runtime enforcement via a domain-specific rule language. Rules define triggers, predicates, and enforcement mechanisms (stop, user inspection, LLM self-examine, invoke action). Pre-execution interception at the action pipeline level. Can halt an agent by inserting a finish action. Does not address concurrent multi-agent resource conflicts — single-agent scope only.

**Source:** arXiv:2503.18666v1. [HIGH confidence]

### 4.7 AgentCgroup (arXiv:2602.09345, February 2026)

Addresses OS-level resource (memory, CPU) isolation for AI coding agents in sandboxed containers. Uses hierarchical cgroups, eBPF enforcement, and `cgroup.freeze` for process suspension. Scope is OS-resource isolation (memory OOM prevention, CPU throttling) between tenant agents — not semantic tool call conflict detection between cooperating agents on shared files.

**Source:** arXiv:2602.09345. [HIGH confidence]

### 4.8 AsyncLM (arXiv:2412.07017, December 2024)

Introduces asynchronous function calling within a single agent: the LLM can continue generating tokens while a function executes. Uses `[TRAP]` tokens as pause points for dependency waiting. The trap mechanism pauses token generation to wait for a prior function result — an intra-agent dependency mechanism, not a cross-agent conflict mechanism.

**Source:** arXiv:2412.07017v1. [HIGH confidence]

### 4.9 Multi-Tool Orchestration Survey (arXiv:2603.22862, March 2026)

Identifies in-execution transaction management approaches including SagaLLM and Atomix, which introduce "epoch-based concurrency isolation and resource frontier tracking with delayed commits and compensation logic." Confirms Atomix is the frontier of this space. Also identifies that "race conditions from concurrent write operations create state pollution propagating across tool chains."

**Source:** arXiv:2603.22862v2. [HIGH confidence]

---

## 5. Production Infrastructure Approaches

### 5.1 Temporal Mutex Workflow

Temporal provides a production-ready mutex workflow pattern: a dedicated `MutexWorkflow` manages lock state for a resource. When a workflow calls `mutex.Lock(ctx, resourceID, timeout)`, it sends a signal to the MutexWorkflow and blocks until it receives the `acquire-lock-event` signal back. The calling workflow is suspended at the lock boundary — not mid-turn, but at explicit lock acquisition points.

**Key limitation for the CLARA use case:** The lock must be explicitly inserted into the workflow code at a known decision point. In a ReAct loop, the agent decides dynamically which tool to call; there is no statically known "this is where I need config.json" point that Temporal can insert a lock call before. It would require the agent to call a resource-acquisition tool before every file operation, which requires a separate LLM-callable tool and additional turns.

**Source:** pkg.go.dev/github.com/temporalio/samples-go/mutex, Temporal community forum. [HIGH confidence]

### 5.2 Python asyncio Primitives

Standard library: `asyncio.Lock`, `asyncio.Event`, `asyncio.Condition`, `asyncio.Semaphore`. These are cooperative multitasking primitives — a coroutine `await lock.acquire()` yields the event loop if the lock is held, allowing other tasks to run. This is the mechanism that would be used in a CLARA-side implementation.

Key nuance from Inngest production findings: `asyncio.Condition` has a lost-update bug under rapid state transitions — if state changes twice before a waiting consumer wakes, the intermediate state is missed. The `ValueWatcher` pattern (per-consumer queues of `(old_value, new_value)` tuples) avoids this.

**Source:** Python 3 documentation, inngest.com/blog/no-lost-updates-python-asyncio. [HIGH confidence]

### 5.3 Restate (Durable Execution)

Restate provides durable execution (crash recovery, state persistence, workflow resumption) and Virtual Objects (single-instance-per-key with queued interactions). Supports suspend/resume for HITL patterns. Does not address cross-workflow resource conflict detection — two different workflows can still concurrently access the same external resource without Restate detecting it.

**Source:** restate.dev/blog/durable-ai-loops. [HIGH confidence]

---

## 6. What Is Genuinely Novel About This Problem

The literature review establishes the following gap map:

| Problem Dimension | Existing Solutions | What Is Unaddressed |
|---|---|---|
| Pre-execution conflict prevention | Semantic Consensus SIG, AgentSpec rules | Conflicts emerging from dynamic tool decisions mid-loop |
| Post-execution effect isolation | Atomix (frontier-gated commits, compensation) | The loop itself is not paused — effects are buffered |
| API-level concurrency control | HiveMind (admission, AIMD, token budget) | Filesystem/tool-level conflicts between cooperating agents |
| Same-file write coordination | MegaAgent (Git mutex) | Global mutex, no turn-level granularity or prioritization |
| Workflow-level pause/resume | LangGraph interrupt(), Temporal mutex, Restate | Not triggered by runtime-detected inter-task conflict |
| Mid-turn execution suspension | AgentScope interrupt (asyncio cancel) | Destructive cancellation, not suspend-and-resume |
| Intra-agent async coordination | AsyncLM traps | Cross-agent conflict signaling |

**The specific unaddressed problem:** A mechanism that (1) continuously monitors a live resource ledger during tool execution across all running ReAct loops, (2) detects when a tool call about to execute conflicts with an ongoing operation in another loop, and (3) suspends the conflicting coroutine at the `await mcp_client.call(...)` boundary until the conflict clears, then resumes it transparently — without cancelling the loop, without requiring the LLM to re-generate the action, and without a pre-planned lock acquisition point.

This differs from all found prior art:
- Atomix buffers *effects* but not *loop execution* — the LLM keeps generating.
- Temporal mutex pauses at a statically known lock point — the ReAct loop has no such point.
- LangGraph interrupt() requires either a human or a pre-coded condition — not a runtime cross-task signal.
- AgentScope cancel is destructive — the turn must be replanned.
- MegaAgent Git mutex blocks at the file operation level — correct but only for file writes, and at global granularity.

The closest theoretical analog is **cooperative preemption in an OS scheduler at the system call boundary**: a process that issues a blocking syscall (write to file) can be suspended at that boundary until the resource is available. The asyncio event loop already provides the mechanism (yield at `await`); what is missing is the *conflict detection layer* that decides whether to yield or proceed when a specific resource claim is observed.

---

## 7. Implementation Objective

This section is a developer-ready implementation brief for CLARA's conflict detection and suspension system.

### The Specific Change

Add a `ResourceLedger` class and a `conflict_gate()` async context manager. Wrap every tool call in `tool_executor.py` with `conflict_gate(resource_path, operation, task_id)`. The gate checks the ledger, acquires a per-resource `asyncio.Lock` if a write conflict exists, yields the event loop, and releases when the conflict clears.

### Libraries Required

No new pip dependencies. Uses: `asyncio.Lock`, `asyncio.Event`, `collections.defaultdict`. All standard library.

### Architecture

```python
# core_logic/resource_ledger.py

import asyncio
from collections import defaultdict
from typing import Literal

Operation = Literal["read", "write"]

class ResourceLedger:
    """
    Tracks active resource operations across concurrent ReAct loop tasks.
    Per-resource asyncio.Lock provides suspension-and-resume semantics.
    A write operation holds the lock; reads wait until write completes.
    Concurrent reads do NOT conflict with each other (read-read safe).
    """

    def __init__(self):
        # Per-resource lock — held for the duration of write operations
        self._write_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        # Active operations for observability: {resource: [(task_id, op), ...]}
        self._active: dict[str, list[tuple[str, Operation]]] = defaultdict(list)
        self._ledger_lock = asyncio.Lock()  # protects _active mutations

    async def acquire(self, resource: str, operation: Operation, task_id: str):
        """
        Called before a tool touches `resource`.
        - write: acquires exclusive lock on this resource (blocks concurrent writes and reads)
        - read: waits if a write is in progress, then proceeds (read-read concurrent OK)
        """
        if operation == "write":
            await self._write_locks[resource].acquire()
        else:  # read — wait until no write is in progress
            # Try to acquire + immediately release to drain any pending write
            lock = self._write_locks[resource]
            async with lock:
                pass  # write lock was free (or just became free), safe to read

        async with self._ledger_lock:
            self._active[resource].append((task_id, operation))

    async def release(self, resource: str, operation: Operation, task_id: str):
        """Called after the tool call completes or raises."""
        async with self._ledger_lock:
            try:
                self._active[resource].remove((task_id, operation))
            except ValueError:
                pass

        if operation == "write":
            try:
                self._write_locks[resource].release()
            except RuntimeError:
                pass  # already released (shouldn't happen with context manager use)

    def snapshot(self) -> dict:
        """Returns current ledger state for observability."""
        return dict(self._active)


from contextlib import asynccontextmanager

@asynccontextmanager
async def conflict_gate(ledger: ResourceLedger, resource: str,
                        operation: Operation, task_id: str):
    """
    Async context manager wrapping any tool call that touches a named resource.
    Suspends the calling coroutine at the await boundary if a write conflict exists.
    Resumes transparently when the conflict clears — no loop restart required.

    Usage in tool_executor.py:
        async with conflict_gate(ledger, "config.json", "write", task_id):
            result = await mcp_client.call("desktop_commander", "write_file", args)
    """
    await ledger.acquire(resource, operation, task_id)
    try:
        yield
    finally:
        await ledger.release(resource, operation, task_id)
```

### Limitations of This Approach

**Read semantics above are conservative:** The `read` path above acquires-and-releases the write lock as a drain mechanism, which means a read will wait for any in-progress write to complete before proceeding. However, it does NOT prevent a new write from starting immediately after the read acquires — there is a TOCTOU gap. For stronger read isolation, use `asyncio.Condition` with a writer-count and reader-count pattern (readers-writer lock). This adds ~20 lines but gives proper concurrent-read / exclusive-write semantics.

**Resource identification is non-trivial:** The ledger key must be a canonical resource identifier. For filesystem paths: normalize to absolute path + drive letter. For other tools (databases, APIs): define a resource key convention per tool. The Interpreter's `args` dict must contain parseable path information.

**Irreversible effects remain irreversible:** This gate suspends the *call*, preventing it from starting during a conflict. It does not roll back a write that already completed. This is appropriate — the goal is prevention, not compensation (unlike Atomix).

**Write starvation:** If reads are frequent and long, writes may be starved. Add a write-priority flag to the Condition variant if this manifests in practice.

**MCP call granularity:** `mcp_client.call()` is atomic from asyncio's perspective — once started, it cannot be interrupted mid-call (it is an `await` that yields once then runs to completion). The gate must therefore be applied *before* the call, not inside it. This is already the correct pattern above.

### Files to Change

| File | Change |
|---|---|
| `core_logic/resource_ledger.py` | Create new (the class above) |
| `core_logic/tool_executor.py` | Import ledger; wrap `execute_fast` and `execute_deliberate` with `conflict_gate` for filesystem and write tools |
| `api.py` | Instantiate `ResourceLedger` at startup; inject into `tool_executor` via `set_ledger()` |
| `core_logic/agent.py` | Pass `task_id` through to tool executor for ledger attribution |

### Key Parameters

- No tunable parameters for the basic `asyncio.Lock` variant.
- For readers-writer variant: `max_readers = None` (unlimited concurrent reads).
- For timeout: wrap `conflict_gate` with `asyncio.wait_for(..., timeout=30.0)` to prevent indefinite suspension if a task crashes while holding a write lock. Log timeout as `[CONFLICT_GATE_TIMEOUT]`.

### What NOT to Do

- Do not use `threading.Lock` — it will block the entire event loop, not just the waiting coroutine.
- Do not use `asyncio.to_thread` to run tool calls unless you mirror the ledger into a thread-safe structure. The ledger above is asyncio-only.
- Do not apply the gate to read-only tools like `date_time`, `python_repl` (no shared external state), or `web_search`. Only apply to filesystem and persistent-state tools.
- Do not apply the gate at the Interpreter level — it must be at the `tool_executor` level, where the actual resource path is known.
- Do not use MegaAgent's Git-mutex approach — it introduces a full Git layer and requires agents to handle merge conflict reasoning, which is prohibitive for a local single-machine system.

### Expected Post-Implementation Behavior

- Concurrent tasks touching different files: no suspension, full concurrency preserved.
- Task A writing `config.json` + Task B reading `config.json`: Task B suspends at `conflict_gate` boundary, resumes after Task A's write completes. B's ReAct loop loop does not restart; no LLM re-generation required.
- Task A and Task B both writing `config.json`: Second writer suspends until first completes; writes are serialized. First writer's content is visible to second writer.
- Latency overhead: zero for non-conflicting paths. Conflict overhead = write duration of the blocking task (bounded by MCP call timeout, typically 5–30s).

**Confidence: MEDIUM** — the mechanism is sound and asyncio-compatible; the devil is in correct resource key extraction from tool args and handling edge cases (task crash while holding lock, ledger key normalization for relative vs. absolute paths).

---

## Key Findings

- **[HIGH]** No existing agentic framework (LangGraph, AutoGen, CrewAI, Magentic-One, AgentScope, OpenAI Assistants) provides automatic mid-execution suspension of a running ReAct loop turn based on a runtime-detected resource conflict with another concurrently running task. This is confirmed by documentation, academic papers, and GitHub issue analysis. [Source: 7 independent systems examined, May 2026]

- **[HIGH]** Atomix (arXiv:2602.14849, Feb 2026) is the closest academic prior art: it provides epoch-based frontier-gated transactional semantics for tool calls with compensation on abort, achieving 7× task success improvement under fault injection. However, Atomix buffers effect *visibility*, not loop *execution* — the LLM generation continues; writes are held in a buffer. It is not a loop suspension mechanism. [Source: arXiv:2602.14849, single paper, high detail]

- **[HIGH]** MegaAgent (ACL 2025) is the only production multi-agent system with explicit file-write conflict serialization: a global Git mutex ensures sequential file commits. This works but is coarse (global lock, not per-resource), requires agents to reason about Git merge conflicts, and does not suspend the ReAct loop — only the write operation. [Source: arXiv:2408.09955v3]

- **[HIGH]** Temporal's mutex workflow pattern is the most mature production solution for inter-workflow resource locking. A calling workflow blocks at `mutex.Lock()` until granted, then proceeds. Applicable to CLARA if tool calls are refactored as explicit lock-acquire steps, but requires the agent to issue a `acquire_resource_lock` tool call before touching any file — adding a turn of latency per lock. [Source: pkg.go.dev/github.com/temporalio/samples-go/mutex]

- **[MEDIUM]** Python `asyncio.Lock` applied at the tool-call boundary in `tool_executor.py` provides a viable asyncio-native implementation with zero external dependencies. A coroutine suspended at `await lock.acquire()` yields the event loop, letting other tasks run — this is cooperative preemption at the system-call boundary, exactly the mechanism needed. Reader-writer lock semantics require a Condition-variable pattern. [Source: Python documentation, Inngest production findings]

- **[LOW]** No academic paper was found that frames the problem as pausing a *running LLM generation coroutine* based on a runtime cross-task resource conflict detected by a live ledger. The closest framing is AsyncLM's `[TRAP]` token (intra-agent dependency waiting) and Atomix's frontier gate (effect visibility control). The specific CLARA problem — cooperative preemption of an asyncio coroutine at the MCP call await boundary, triggered by a cross-task conflict signal — appears to be novel at the implementation level. [Source: full literature survey, May 2026]

**Verdict:** Implement a per-resource `asyncio.Lock`-based `ResourceLedger` in `tool_executor.py`. This is asyncio-native, zero-dependency, and directly addresses the gap. Atomix's epoch-frontier model is academically superior but requires a full transaction runtime shim; the lock approach is simpler, sufficient for CLARA's single-machine use case, and implementable in under 100 lines. The problem at the granularity described (mid-loop coroutine suspension at the MCP await boundary) is not solved by any existing framework and represents a genuine implementation gap in the field.

---

*Sources consulted: arXiv papers 2602.14849, 2604.17111, 2602.09345, 2604.16339, 2408.09955, 2503.18666, 2510.18893, 2412.07017, 2603.22862, 2511.00739, 2604.11378; LangGraph official docs; AutoGen GitHub discussions; CrewAI changelog; Magentic-One paper; AgentScope docs; OpenAI Assistants API docs; Anthropic engineering blog; Temporal mutex samples; Restate blog; Inngest blog; Anthropic multi-agent coordination patterns blog; Python asyncio documentation.*
