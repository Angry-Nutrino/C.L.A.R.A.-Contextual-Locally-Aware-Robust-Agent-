# C.L.A.R.A. System State

## Current Objective
- Migrating Gatekeeper from Grok API to local Phi-3 (50ms latency target).

## Architecture Decisions (The "Why")
- [2026-03-06] Severed memory consolidation from `self.llm` to prevent 30k TPM quadratic bloat. Memory now uses a stateless, disposable ghost-instance(chat_snapshot).
- [2026-03-06] Replaced synchronous sampling with a state-machine stream. React frontend UI utilizes `turn_id` to stack thoughts instead of overwriting.

## Recent Kills/Bugs (Changelog)
- [2026-03-06] Fixed React visual cascade glitch by overwriting the active stream block.
- [2026-03-06] Added regex guillotine `\n?(?:Action|Final Answer|Observation):?` to prevent syntax bleed into the UI stream.
- [2026-03-08] Fixed `m.role` integer/string mismatch that was leaking the system prompt into the memory summarizer (memorize_episode, chat_snapshot).

## Active Bugs / Debt
- 6-second API wall on Gatekeeper intent classification (Pending local Phi-3 swap).
- Nvidia Riva ASR integration pending.