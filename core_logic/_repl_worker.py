"""The child half of the python_repl OS sandbox. Runs in a separate process under a
RESTRICTED TOKEN, so the boundary is enforced by Windows rather than by this file.

Read this together with `repl_sandbox.py`, which launches it.

WHY THIS EXISTS, in one paragraph. The in-process compute-only namespace
(`tools._build_exec_namespace`) is not a containment boundary and BRIEF_60 section 5.2 already
retracted the claim that it was. `tests/test_compute_only_containment.py` measures the escape every
run: 217 classes are reachable through `().__class__.__base__.__subclasses__()` and the real
`open()` is reachable through their `__globals__`. No arrangement of a Python namespace closes that.
This module does not try. It assumes the escape succeeds and removes the AUTHORITY instead, so the
interpreter can hold a real `open()` and still be refused by the operating system.

WHAT THIS FILE IS NOT. Nothing here is a security control. Every line below runs INSIDE the
sandbox with whatever authority the token was given. If you are looking for the boundary it is
`CreateRestrictedToken` in `repl_sandbox.py`, not anything written here.
"""
import sys
import json
import io
import traceback


def _main():
    # argv: result_path, code_path. The code arrives in a FILE rather than on the command line, so
    # quoting, length limits and encoding cannot mangle it before it executes.
    result_path, code_path = sys.argv[1], sys.argv[2]

    out = io.StringIO()
    payload = {"output": "", "error": "", "kind": ""}

    def _capture_print(*args, sep=" ", end="\n", **kwargs):
        out.write(sep.join(str(a) for a in args) + end)

    def _utf8_open(*a, **kw):
        # Same UTF-8 default as the in-process path, for the same reason: a bare open() on Windows
        # defaults to cp1252 and a charmap error on a UTF-8 file used to cascade into malformed
        # JSON Actions. Keeping the behaviour identical is what makes the two paths comparable in
        # shadow mode; a difference in output should mean the SANDBOX did something, not that the
        # encoding default moved.
        import builtins as _bi
        mode = a[1] if len(a) > 1 else kw.get("mode", "r")
        if "b" not in mode and "encoding" not in kw:
            kw["encoding"] = "utf-8"
        return _bi.open(*a, **kw)

    try:
        code = _utf8_open(code_path, "r").read()
    except Exception as e:                                  # cannot even read our own input
        payload["error"] = "sandbox worker could not read the code file: %s" % e
        payload["kind"] = "worker_input"
        _write(result_path, payload)
        return

    try:
        # ONE dict as both globals and locals, matching the in-process path exactly. With separate
        # mappings, top-level assignments land in locals while comprehensions resolve against
        # globals, so `content = ...` followed by `[x for x in content]` raises NameError.
        ns = {"open": _utf8_open, "print": _capture_print, "__name__": "__main__"}
        exec(code, ns)
        payload["output"] = out.getvalue()
    except Exception as e:
        # Partial output is kept. A script that printed three lines and then raised has produced
        # three real lines, and discarding them makes the failure harder to read than it needs to be.
        payload["output"] = out.getvalue()
        payload["error"] = "%s: %s" % (type(e).__name__, e)
        payload["kind"] = "user_code"
        payload["traceback"] = traceback.format_exc()[-2000:]

    _write(result_path, payload)


def _write(path, payload):
    import builtins as _bi
    try:
        with _bi.open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        # The parent treats an absent result file as a hard sandbox failure and says so. There is
        # nothing useful to do here and nowhere trustworthy to say it.
        pass


if __name__ == "__main__":
    _main()
