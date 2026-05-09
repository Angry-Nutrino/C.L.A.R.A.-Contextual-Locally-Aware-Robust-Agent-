# CLARA STRESS TEST v2 - QUERY LIST

01. Analyze today's benchmarks/ log. Write benchmarks/latency_correlation_report.py to correlate INTERP_MS with total token count. Identify if spikes are driven by prompt size or reasoning.

02. Examine core_logic/conflict.py. Explain how ConflictDetector would fail if 'User' and 'Autonomous' events arrived at the same millisecond. Propose 3-way arbitration code.

03. Create core_logic/vram_sentinel.py using the existing observer pattern to print a warning whenever VRAM usage (via Soul endpoint) exceeds 3.5GB.

04. Compare CLAUDE.md (Memory Consolidation) to core_logic/crud.py. Verify if _vault_lock is implemented as documented. Is there a race condition in the cosine similarity check?

05. Audit core_logic/. List every file NOT in the 'Key Modules' table of CLAUDE.md. Categorize them as 'legacy', 'temporary', or 'missing documentation'.

06. Find the largest gap in TIMELINE.md since February. Scan logs/ for that period and summarize what was actually being worked on during that undocumented time.

07. Cross-reference user_profile.preferences in memory.json with the last 20 episodic logs. Does response_style still match interaction density?

08. Search consult_archive for 'latency issues' from March. Compare those to today's benchmarks. Did our changes actually solve the bottleneck?

09. Write tools/git_pulse.py to fetch GitHub commits and PRs for this week to see if hitting 'September Mode' goals.

10. Check known_locations in memory.json. Find any file in those paths modified in the last 24 hours and summarize the changes.

11. Check TRAD_INTRADAY project's config.py. Based on current Nifty 50 performance, calculate the INITIAL_CAPITAL needed for breakeven given the fee drag.

12. Write tests/test_adaptive_sl_edge.py that would cause the trading system to fail if SL_CONFIRMATION_CANDLES were set to 1 instead of 6.

13. Analyze the Vision Tool implementation. Predict if resizing to 1280px degrades the ability to read 8pt font from a Windows 11 screenshot.

14. Explain what happens to a task with priority 0.95 in the EventQueue if five priority 1.0 tasks are added while it is pending. Will it ever dispatch?

15. If a 50MB PDF is added to core_logic/docs/, predict the CPU% spike during the FAISS rebuild. Will it interfere with CHAT stream latency on an RTX 3050?

16. Write a Python script to visualize token usage trends (Prompt vs. Completion) over the last 7 days using the benchmarks/ data.

17. Based on the last 10 'FAST' failures, propose new confidence and uncertainty thresholds for the Router to minimize DELIBERATE escalations.

18. Search for top-performing SLMs (under 10B) released in the last two months. Compare benchmarks against the Phi3-mini used for routing.

19. Draft a new class in core_logic/proactive_commit.py that would trigger a git commit summary whenever a file passes unit tests (no integration).

20. Review PERSONA in system_prompt.py. Identify three guardrails drifted from in the last 24 hours and propose a system-message fix.



# Result
- Passed all 20 tests in 2 go's.