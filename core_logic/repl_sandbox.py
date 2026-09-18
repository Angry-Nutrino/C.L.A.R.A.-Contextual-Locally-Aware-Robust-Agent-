"""OS-enforced sandbox for `python_repl` (G38 / BRIEF_60, built 2026-09-18).

THE FINDING THIS ANSWERS. `python_repl` executes arbitrary Python and is not in `MUTATING_TOOLS`,
so `admissibility.gate()` short-circuits and the action produces no envelope, no verdict and no
ledger row. Proven live on 2026-08-10: with `write_file` unregistered, the agent completed a file
write through this tool and left no receipt. The exemption is not a hole an adversary has to aim
at, it is where the agent routes AUTOMATICALLY when the gated tool is unavailable.

WHY NOT THE NAMESPACE. `PYTHON_REPL_COMPUTE_ONLY` rebuilds the exec namespace without `os`, `open`
and friends. It is useful and it is NOT containment, which BRIEF_60 section 5.2 already retracted
in writing. `tests/test_compute_only_containment.py` re-measures the escape on every run: 217
classes reachable via `().__class__.__base__.__subclasses__()`, real `open()` reachable through
their `__globals__`. A language-level namespace cannot close a language-level escape.

WHAT THIS DOES INSTEAD, and why it is a different kind of claim. The code runs in a separate
process under a Windows RESTRICTED TOKEN. The escape still works inside that process. It stops
mattering, because the token has no write authority and the kernel refuses the operation. The
question moves from "can the interpreter reach the function" to "can this process perform the
operation", and only the second one is a boundary.

HOW THE TOKEN IS BUILT, and the measurement that decided it. This repository's ACL is exactly:

    NT AUTHORITY\\Authenticated Users:(M)      <- the WRITE grant
    BUILTIN\\Users:(RX)                        <- READ + EXECUTE only

Three configurations were measured on 2026-09-18 before any of this was written:
  * `DISABLE_MAX_PRIVILEGE` alone      -> every write still SUCCEEDED. Privileges do not grant
                                          file access, so removing them changes nothing.
  * deny-only EVERY group              -> process died at startup, 0xC0000135 DLL_NOT_FOUND. The
                                          token had lost READ access to the Python runtime itself.
  * deny-only all EXCEPT Everyone and
    BUILTIN\\Users                      -> process runs, reads fine, and every write is refused.

So the rule is: deny `Authenticated Users` (which is where write comes from) and keep `Users`
(which is where read and execute come from). No ACL is modified and no account is created.

MEASURED RESULT of the third configuration, from the spike:
    repo_write_plain            DENIED (PermissionError)
    gadget_reached_real_open    True                        <- the escape still succeeds
    repo_write_via_gadget       DENIED (PermissionError)    <- and is refused anyway
    repo_read                   SUCCEEDED                   <- expected, Users has RX
    own_workdir_write           SUCCEEDED                   <- results travel back this way

WHAT THIS DOES NOT CLOSE. Stated here rather than discovered later, because a control described
more broadly than it was measured is the failure this whole line of work exists to prevent.
  1. READS ARE NOT BLOCKED. `core_logic/.env` was read successfully from inside the sandbox. This
     design removes write authority, not read authority. Secrets remain readable to executed code.
  2. NETWORK IS NOT BLOCKED. A socket to 1.1.1.1:53 connected. Exfiltration is not addressed.
  3. PROCESS CREATION IS NOT BLOCKED. `cmd /c` ran. The child inherits the restricted token, so it
     inherits the write denial, but the capability itself remains.
  4. WINDOWS ONLY. There is no POSIX path here. On any other platform this degrades to the
     in-process behaviour and says so loudly rather than pretending to be enforcing.
  5. The workdir IS writable by design, because that is how output returns. Code can write there.
     It is a fresh temp directory per call and is removed afterwards.

STATUS: DORMANT. `PYTHON_REPL_ISOLATION` defaults to `off` and nothing changes until it is moved.
`shadow` runs BOTH paths and logs where they disagree, which is the evidence that should decide the
flip, and is the same measure-before-arming doctrine used for the gate and for compute-only.
"""
import os
import sys
import json
import shutil
import tempfile

IS_WINDOWS = sys.platform == "win32"

# Everyone, and BUILTIN\Users. These two carry read+execute on the runtime and the repo; denying
# them is what killed the process with DLL_NOT_FOUND in the second measured configuration.
_KEEP_SIDS = {"S-1-1-0", "S-1-5-32-545"}

DEFAULT_TIMEOUT_S = 30


class SandboxUnavailable(RuntimeError):
    """The sandbox could not be established. Never swallowed: the caller decides whether to fall
    back to in-process execution, and that decision gets logged as a downgrade rather than passing
    silently as if the boundary had held."""


def available():
    """Can this machine actually enforce the boundary? Checked rather than assumed."""
    if not IS_WINDOWS:
        return False, "not Windows; no restricted-token path exists"
    try:
        import win32security, win32process, win32event, win32con, win32api, win32job  # noqa: F401
    except ImportError as e:
        return False, "pywin32 missing (%s)" % e
    return True, "ok"


def _deny_only_groups():
    """Every group in the current token except the ones that carry read+execute."""
    import win32security, win32api
    htok = win32security.OpenProcessToken(
        win32api.GetCurrentProcess(),
        win32security.TOKEN_DUPLICATE | win32security.TOKEN_QUERY)
    deny = []
    for sid, attr in win32security.GetTokenInformation(htok, win32security.TokenGroups):
        if attr & win32security.SE_GROUP_LOGON_ID:
            continue                                   # the logon session SID; denying it is not meaningful
        if win32security.ConvertSidToStringSid(sid) in _KEEP_SIDS:
            continue
        deny.append((sid, 0))
    return deny


def run(code, timeout_s=DEFAULT_TIMEOUT_S, python_exe=None):
    """Execute `code` in a restricted-token subprocess.

    Returns {"output", "error", "kind", "timed_out", "exit_code"}.
    Raises SandboxUnavailable if the boundary could not be established, so a caller can never
    mistake a failed sandbox for a clean run.
    """
    ok, why = available()
    if not ok:
        raise SandboxUnavailable(why)

    import win32security, win32process, win32event, win32con, win32api, win32job

    python_exe = python_exe or sys.executable
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_repl_worker.py")
    if not os.path.exists(worker):
        raise SandboxUnavailable("worker missing at %s" % worker)

    workdir = tempfile.mkdtemp(prefix="clara_repl_")
    code_path = os.path.join(workdir, "code.py")
    res_path = os.path.join(workdir, "result.json")
    hjob = None
    try:
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        htok = win32security.OpenProcessToken(
            win32api.GetCurrentProcess(),
            win32security.TOKEN_DUPLICATE | win32security.TOKEN_QUERY |
            win32security.TOKEN_ASSIGN_PRIMARY | win32security.TOKEN_IMPERSONATE)
        rtok = win32security.CreateRestrictedToken(
            htok, win32security.DISABLE_MAX_PRIVILEGE, _deny_only_groups(), None, None)

        # A job object so a runaway child cannot outlive us. KILL_ON_JOB_CLOSE means the process
        # dies when this handle closes, including if the parent crashes between spawn and wait.
        hjob = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(hjob, win32job.JobObjectExtendedLimitInformation)
        info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(
            hjob, win32job.JobObjectExtendedLimitInformation, info)

        si = win32process.STARTUPINFO()
        # -I is isolated mode: no user site-packages, no PYTHON* env vars, cwd off sys.path. It
        # keeps the sandbox from importing something out of the workdir it can write to.
        cmd = '"%s" -I "%s" "%s" "%s"' % (python_exe, worker, res_path, code_path)
        hp, ht, pid, tid = win32process.CreateProcessAsUser(
            rtok, python_exe, cmd, None, None, False,
            win32con.CREATE_NO_WINDOW | win32con.CREATE_SUSPENDED, None, workdir, si)
        try:
            win32job.AssignProcessToJobObject(hjob, hp)
        except Exception:
            pass                                       # job assignment is containment-of-runaways, not the boundary
        win32process.ResumeThread(ht)

        waited = win32event.WaitForSingleObject(hp, int(timeout_s * 1000))
        timed_out = (waited == win32event.WAIT_TIMEOUT)
        if timed_out:
            try:
                win32process.TerminateProcess(hp, 1)
            except Exception:
                pass
            return {"output": "", "timed_out": True, "exit_code": None, "kind": "timeout",
                    "error": "python_repl timed out after %ss in the sandbox and was killed." % timeout_s}

        exit_code = win32process.GetExitCodeProcess(hp)
        if not os.path.exists(res_path):
            # The process started and produced nothing. Almost always a startup failure, and the
            # exit code is the useful part (0xC0000135 = the token lost read access to the runtime).
            return {"output": "", "timed_out": False, "exit_code": exit_code, "kind": "sandbox",
                    "error": "sandbox produced no result (exit=%s). The code did not run."
                             % _fmt_exit(exit_code)}
        with open(res_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload.setdefault("output", "")
        payload.setdefault("error", "")
        payload.setdefault("kind", "")
        payload["timed_out"] = False
        payload["exit_code"] = exit_code
        return payload
    finally:
        if hjob is not None:
            try:
                win32api.CloseHandle(hjob)
            except Exception:
                pass
        shutil.rmtree(workdir, ignore_errors=True)


def _fmt_exit(code):
    if code is None:
        return "None"
    known = {
        0xC0000135: "0xC0000135 DLL_NOT_FOUND (token lost read access to the Python runtime)",
        0xC0000142: "0xC0000142 DLL_INIT_FAILED (same class: runtime unreadable under this token)",
    }
    return known.get(code & 0xFFFFFFFF, str(code))


# ── Self-test ────────────────────────────────────────────────────────────────────────────────────
# Both directions. A sandbox that has only ever been shown to ALLOW things is indistinguishable from
# one that is not running. The general form of that point was made to this project by an external
# reviewer: a check never shown to refuse cannot be told apart from a check that cannot refuse.
if __name__ == "__main__":
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    probe = os.path.join(REPO, "_sandbox_selftest_probe.txt")
    ok, why = available()
    print("repl_sandbox self-test")
    print("  available: %s (%s)" % (ok, why))
    if not ok:
        print("  SKIPPED - no boundary on this platform.")
        sys.exit(0)

    fails = 0

    def check(label, got, want):
        global fails
        good = (got == want)
        if not good:
            fails += 1
        print("  %-4s %-42s got=%s want=%s" % ("OK" if good else "FAIL", label, got, want))

    # POSITIVE CONTROL first: if ordinary compute does not work, nothing below means anything.
    r = run("print(6*7)")
    check("compute runs", r["output"].strip(), "42")
    check("compute has no error", r["error"], "")

    r = run("print('a');print('b')")
    check("multi-line output preserved", r["output"].strip().replace("\r", ""), "a\nb")

    r = run("x = [i*i for i in range(4)]\nprint(sum(x))")
    check("shared globals/locals namespace", r["output"].strip(), "14")

    # NEGATIVE CONTROLS: the boundary must refuse.
    r = run("open(%r, 'w').write('x')" % probe)
    check("repo write refused", "PermissionError" in r["error"], True)
    check("repo write left no file", os.path.exists(probe), False)

    r = run(
        "ro=None\n"
        "for c in ().__class__.__base__.__subclasses__():\n"
        "    g=getattr(getattr(c,'__init__',None),'__globals__',None)\n"
        "    if g and '__builtins__' in g:\n"
        "        b=g['__builtins__']; b=b if isinstance(b,dict) else b.__dict__\n"
        "        if 'open' in b: ro=b['open']; break\n"
        "print('reached', ro is not None)\n"
        "ro(%r,'w').write('x')\n" % probe)
    check("gadget still reaches real open", "reached True" in r["output"], True)
    check("gadget write refused anyway", "PermissionError" in r["error"], True)
    check("gadget write left no file", os.path.exists(probe), False)

    r = run("import os\nos.remove(%r)" % os.path.join(REPO, "CLAUDE.md"))
    check("repo delete refused", "PermissionError" in r["error"], True)
    check("CLAUDE.md still present", os.path.exists(os.path.join(REPO, "CLAUDE.md")), True)

    r = run("import time\ntime.sleep(30)", timeout_s=3)
    check("timeout kills the process", r["timed_out"], True)

    r = run("raise ValueError('boom')")
    check("user error surfaces", "ValueError: boom" in r["error"], True)
    check("user error is classified", r["kind"], "user_code")

    if os.path.exists(probe):
        os.remove(probe)
    print("")
    print("repl_sandbox self-test: %s" % ("PASSED" if not fails else "FAILED (%d)" % fails))
    print("  Reminder: this measures WRITE denial. Reads, network and process creation are NOT")
    print("  blocked by this design and are listed in the module docstring.")
    sys.exit(1 if fails else 0)
