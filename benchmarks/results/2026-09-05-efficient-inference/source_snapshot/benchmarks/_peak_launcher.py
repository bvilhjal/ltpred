"""Spawn one command and report *its* peak RSS, not its parent's.

``getrusage``'s ``ru_maxrss`` is a property of the task, and ``execve`` does not
reset it: a child forked from a fat parent reports that parent's high-water
mark before it has allocated anything of its own. The kernel *does* give the
exec'd program a fresh address space, so ``/proc/<pid>/status`` ``VmHWM`` is
clean -- but polling it can miss a spike, and misses short-lived processes.

This launcher keeps ``wait4``'s exactness by removing the inherited floor. It
imports nothing outside the standard library, so its own resident set is a few
MB. A benchmark driver holding simulated families spawns *this*, and the
measured command forks from here. What ``wait4`` then returns is the command's
own peak plus a few MB.

Adapted from ldpred3's ``benchmarks/_peak_launcher.py`` (same inherited-floor
problem). Darwin ``wait4`` reports ``ru_maxrss`` in bytes; Linux reports KiB.

Usage (JSON in, JSON out)::

    python _peak_launcher.py request.json

where ``request.json`` is ``{"cmd": [...], "cwd": "...", "out": "..."}`` and
the optional ``env`` map, if present, is the child's environment. The command's
stdout and stderr are merged into ``out``. The reply includes ``peak_rss_bytes``
and a ``floor_kb`` (launcher footprint; 0 when ``/proc`` is unavailable).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time


def _vmhwm_kb(pid="self"):
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def _wait4_rss_units(ru_maxrss, platform=sys.platform):
    """Return legacy kB plus exact bytes from a ``wait4`` RSS value."""
    if platform == "darwin":
        return ru_maxrss / 1000.0, int(ru_maxrss)
    return ru_maxrss, int(ru_maxrss) * 1024


def run_peak(cmd, cwd=None, env=None):
    """Run ``cmd`` via this launcher; return ``(CompletedProcess, peak_rss_bytes)``.

    ``CompletedProcess.stdout`` is the child's merged stdout+stderr.
    ``CompletedProcess.floor_kb`` is the launcher footprint (0 on macOS).
    """
    launcher = os.path.abspath(__file__)
    with tempfile.TemporaryDirectory(prefix="peak_") as box:
        req_path = os.path.join(box, "request.json")
        out_path = os.path.join(box, "output.txt")
        payload = {
            "cmd": [str(c) for c in cmd],
            "cwd": os.path.abspath(cwd or os.getcwd()),
            "out": out_path,
        }
        if env is not None:
            payload["env"] = {str(k): str(v) for k, v in env.items()
                              if v is not None}
        with open(req_path, "w") as fh:
            json.dump(payload, fh)
        launch = subprocess.run(
            [sys.executable, launcher, req_path],
            capture_output=True, text=True)
        try:
            out = open(out_path, encoding="utf-8", errors="replace").read()
        except FileNotFoundError:
            out = ""
        if launch.returncode != 0:
            raise RuntimeError(
                "peak launcher failed:\n"
                f"{launch.stdout}\n{launch.stderr}\n{out}")
        reply = json.loads(launch.stdout.strip().splitlines()[-1])

    proc = subprocess.CompletedProcess(cmd, reply["returncode"], out, "")
    proc.floor_kb = reply["floor_kb"]
    proc.peak_rss_bytes = int(reply["peak_rss_bytes"])
    return proc, proc.peak_rss_bytes


def main():
    with open(sys.argv[1]) as fh:
        req = json.load(fh)
    floor = _vmhwm_kb()
    child_env = req.get("env")
    with open(req["out"], "w") as out:
        p = subprocess.Popen(
            req["cmd"], cwd=req["cwd"], env=child_env,
            stdout=out, stderr=subprocess.STDOUT, text=True)
        if hasattr(os, "wait4"):
            while True:
                try:
                    _pid, status, usage = os.wait4(p.pid, 0)
                    break
                except InterruptedError:
                    continue
            rc = os.waitstatus_to_exitcode(status)
            peak, peak_bytes = _wait4_rss_units(usage.ru_maxrss)
        else:  # pragma: no cover
            peak = 0
            while p.poll() is None:
                peak = max(peak, _vmhwm_kb(p.pid))
                time.sleep(0.01)
            peak = max(peak, _vmhwm_kb(p.pid))
            p.wait()
            rc = p.returncode
            peak_bytes = int(peak) * 1024
    json.dump({
        "peak_kb": peak,
        "peak_rss_bytes": peak_bytes,
        "floor_kb": floor,
        "returncode": rc,
    }, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
