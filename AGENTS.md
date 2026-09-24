## 15. Working rules for the coding agent (save this section as `AGENTS.md` in the repo root)

These rules are binding for the agent doing the work. If an agent reading this file and
there is no `AGENTS.md` in the repo, create one containing this section verbatim, then follow it.

1. **Never claim a result you did not observe.** If you did not run it, write "not run". Paste real
   command output for every verification claim. A fabricated result is the worst possible failure
   mode for this project.
2. **Never put a number on screen that was not measured or sourced.** Missing counter ⇒ the UI says
   so. Peak spec ⇒ labelled `peak` with a source URL in `config/platform_profiles.json`.
3. **Static shapes only.** The NPU rejects dynamic shapes.
4. **`AUTO` + `ov::device::priorities`, never `HETERO`** for splitting work on the NPU.
5. **`EXECUTION_DEVICES` is the only proof of placement.** Assert on it.
6. **`config/models.json` and `config/platform_profiles.json` are verified data, not suggestions.**
   Re-run `tools/verify_sources.py` and record the result if a source must change.
7. **No network at runtime.** Models, labels and videos are downloaded once, up front.
8. **Ask before adding a dependency.** The venv is pinned.
9. **One phase at a time** (§9). Each phase ends runnable. After each phase run
   `tools/preflight.py`, paste the table into `bench_report.md`, and commit
   (`phase N: <what works now>`).
10. **Keep layout tokens in `app/theme.py`.** No metric values or magic numbers in UI code.
11. **Telemetry sampling runs on a background thread at 5 Hz** and reaches the UI only through Qt
    signals. Never touch widgets from a non-UI thread.
12. **Never crash in front of a customer.** Catch per-stream exceptions, mark that stream failed,
    keep the rest running, log a traceback.
13. After any Intel driver update, run `tools/reset_cache.ps1` — NPU blob cache compatibility is
    not guaranteed across driver versions.
14. **On the demo machine, script execution may be blocked by policy.** Never depend on running a
    `.ps1` or `.bat` file: issue the commands directly in the shell instead. Batch/PowerShell
    wrapper scripts in this repo are conveniences for a human operator, not the build path — the
    build path is the commands themselves.
15. **The application must run as a standard user.** Never require elevation at runtime, and never
    weaken a security setting (Defender exclusions, ASR rules, execution policy, AppLocker) to make
    something run. If a command needs elevation, it is an install step, not a runtime step — say so
    and let the human run it.

**Verification before declaring a phase done:**

```powershell
python tools\verify_sources.py     # URLs verified by content, not status code
python tools\probe_telemetry.py    # telemetry map present and consistent
python tools\preflight.py          # every row PASS (WARN only where this spec allows it)
python -m app.main --selftest      # runs with no network, exits 0
```

Then run the actual demo for the duration the phase requires and report the numbers you saw.
