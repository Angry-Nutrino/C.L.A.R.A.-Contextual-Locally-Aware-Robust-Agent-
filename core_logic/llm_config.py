"""Single source of truth for the DeepSeek model name (G24, 2026-08-01).

The model-name string was hardcoded in 7 call sites (agent.py, interpreter.py, ambient_loop.py) and this
exact class broke every LLM call TWICE — the Grok->DeepSeek migration and the 2026-07-25
`deepseek-chat`->`deepseek-v4-flash` rename — each silently dropping CLARA onto the fallback path for hours.
Centralizing it here makes the next rename a one-line change (set the env var, or change this default).

Override at runtime with the `DEEPSEEK_MODEL` env var (read once at import — entrypoints load .env first).
The default tracks CLARA's current tier (V4-Flash per CLAUDE.md). Behavior-preserving: with no env var set,
this resolves to exactly the string every site used before.
"""
import os

# 2026-09-07: the docstring above claimed "entrypoints load .env first". That was false and this module
# is the SECOND instance of the same defect (tools.py was the first, found while arming
# PYTHON_REPL_COMPUTE_ONLY). agent.py:14 imports this module, which runs the os.getenv below, and
# agent.py:183 loads core_logic/.env 169 lines LATER. A bare load_dotenv() also resolves nothing from the
# repo root because there is no .env there. So DEEPSEEK_MODEL could never have been overridden by the
# env file, silently. Harmless to date only because the key was never set; setting it would have been
# ignored without any error. Load the file by explicit path before reading, exactly as agent.py does.
# 2026-09-13 (G44): the explicit-path load above was the right FIX for the defect described, but it
# left this module reading a POLICY-GOVERNED key directly, which is the pattern BRIEF_62 removed
# everywhere else. policy_config is now the single owner: it loads .env by explicit path at import and
# RECORDS a fault when a value is absent or unusable, so a mis-set model name is visible at startup
# instead of silently falling back. DEEPSEEK_MODEL is open_valued, so this is behaviour-identical to
# the os.getenv it replaces, minus the silence. Found by tests/test_env_access_boundary.py.
try:                                      # package import (normal runtime)
    from . import policy_config as _policy
except ImportError:                       # direct execution of this file
    import policy_config as _policy

DEEPSEEK_MODEL = _policy.resolve("DEEPSEEK_MODEL")
