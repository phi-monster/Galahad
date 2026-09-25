#!/usr/bin/env python3
"""One-command measurement battery.

Reproduces the object-substitution table of the paper. Starts the policy server on the
checkpoint you point it at, runs the four faces against it, parses the results, and prints
the table.

    python -m galahad.battery                      # run it
    python -m galahad.battery --dry-run            # print the exact commands, run nothing
    python -m galahad.battery --face task --face occ

or from Python:

    from galahad.battery import run
    print(run(ckpt="/path/to/checkpoint")["task"].rate)

The four faces, and what each one is for:

    task      fetch the named object                  the headline number
    occ       both cameras blanked                    a grounded policy must collapse
    swap      rename the target, scene held fixed     the positive control: does the arm
                                                      obey the NEW name? (OBEY), and does it
                                                      avoid the object it was trained to
                                                      fetch? (the reported rate, near 0)
    nonsense  instruction replaced by "xxx"           must collapse; kills paraphrase luck

Requirements: a CUDA GPU, a LIBERO-PRO checkout, and a checkpoint. See README.md.
"""
import argparse
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))

#: A face's parsed outcome. ``obey`` is populated for the swap face only.
Result = namedtuple("Result", "face successes trials rate obey obey_trials")

#: face -> extra flags for libero_pro_eval.py, and the env it needs.
FACES = {
    "task":     ([], {}),
    "occ":      (["--occ"], {}),
    "swap":     (["--swap_scene"], {"PROBE": "1"}),   # PROBE=1 is what emits OBEYED_NAME
    "nonsense": (["--nonsense"], {}),
}

_SR = re.compile(r"SR\s*=\s*(\d+)/(\d+)")
_OBEY = re.compile(r"OBEY=(\d+)/(\d+)")


def _libero_pro_root(explicit=None):
    """Locate the LIBERO-PRO checkout, or fail with an actionable message."""
    root = explicit or os.environ.get("LIBERO_PRO", "/root/LIBERO-PRO")
    bddl = os.path.join(root, "libero", "libero", "bddl_files")
    if not os.path.isdir(bddl):
        raise SystemExit(
            "LIBERO-PRO not found at %r (looked for %s).\n"
            "Point at your checkout with --libero-pro PATH or LIBERO_PRO=PATH.\n"
            "It is a separate install; see README.md." % (root, bddl)
        )
    return root


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _server_cmd(port):
    return [sys.executable, os.path.join(HERE, "arm_eval_server.py"), "--port", str(port)]


def _face_cmd(face, root, port, n_trials, suite):
    flags, _ = FACES[face]
    return [
        sys.executable, os.path.join(HERE, "libero_pro_eval.py"),
        "--bddl_dir", os.path.join(root, "libero", "libero", "bddl_files", suite["bddl"]),
        "--init_dir", os.path.join(root, "libero", "libero", "init_files", suite["init"]),
        "--port", str(port),
        "--n_trials", str(n_trials),
    ] + flags


def _start_server(ckpt, port, env, timeout=900):
    """Launch the policy server and block until it is accepting connections.

    Wait for ARM_EVAL_SERVER_READY, not the earlier ARM_EVAL_SERVER loaded: the latter is
    printed when the weights finish loading, which is before bind()/listen(). The scorers
    connect once with no retry, so returning on the earlier line races them.
    """
    env = dict(env, CKPT=ckpt)
    proc = subprocess.Popen(_server_cmd(port), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    deadline = time.time() + timeout
    for line in proc.stdout:
        sys.stderr.write("[server] " + line)
        if "ARM_EVAL_SERVER_READY" in line:
            return proc
        if proc.poll() is not None or time.time() > deadline:
            break
    proc.kill()
    raise SystemExit("policy server failed to come up on port %d; see [server] output above" % port)


def _parse(face, text):
    sr = _SR.search(text)
    if not sr:
        raise SystemExit("no 'SR = a/b' line in the %s output; the eval did not finish" % face)
    succ, tot = int(sr.group(1)), int(sr.group(2))
    obey = obey_tot = None
    if face == "swap":
        pairs = _OBEY.findall(text)
        if pairs:
            obey = sum(int(a) for a, _ in pairs)
            obey_tot = sum(int(b) for _, b in pairs)
    return Result(face, succ, tot, 100.0 * succ / max(tot, 1), obey, obey_tot)


def run(faces=("task", "occ", "swap", "nonsense"), ckpt=None, libero_pro=None,
        port=None, n_trials=20, suite=None, dry_run=False):
    """Run the battery and return ``{face: Result}``.

    ``ckpt`` defaults to $CKPT. ``libero_pro`` defaults to $LIBERO_PRO. With ``dry_run``
    the commands are printed and nothing is executed (no GPU needed).
    """
    suite = suite or {"bddl": "libero_object_task", "init": "libero_object"}
    ckpt = ckpt or os.environ.get("CKPT")
    if not ckpt and not dry_run:
        raise SystemExit("no checkpoint: pass --ckpt PATH or set CKPT=PATH")
    # --dry-run only prints; it must work on a laptop with no simulator installed, so the
    # path is taken at face value and never validated.
    root = (libero_pro or os.environ.get("LIBERO_PRO", "<LIBERO_PRO>")) if dry_run \
        else _libero_pro_root(libero_pro)
    port = port or (5600 if dry_run else _free_port())

    if dry_run:
        print("CKPT=%s \\\n  %s" % (ckpt or "<CKPT>", " ".join(map(shlex.quote, _server_cmd(port)))))
        for face in faces:
            _, extra = FACES[face]
            prefix = "".join("%s=%s " % kv for kv in sorted(extra.items()))
            print(prefix + " ".join(map(shlex.quote, _face_cmd(face, root, port, n_trials, suite))))
        return {}

    results, server = {}, _start_server(ckpt, port, os.environ.copy())
    try:
        for face in faces:
            _, extra = FACES[face]
            env = dict(os.environ, **extra)
            print("\n=== %s ===" % face.upper(), flush=True)
            out = subprocess.run(_face_cmd(face, root, port, n_trials, suite),
                                 env=env, capture_output=True, text=True).stdout
            sys.stdout.write(out)
            results[face] = _parse(face, out)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
    return results


def _print_table(results):
    print("\n%-10s %-38s %s" % ("face", "question", "result"))
    print("-" * 72)
    rows = [
        ("task", "fetches the named object?", lambda r: "%.1f%% (%d/%d)" % (r.rate, r.successes, r.trials)),
        ("swap", "goes to the new name?", lambda r: "%d/%d" % (r.obey, r.obey_trials) if r.obey is not None
         else "n/a (PROBE did not report)"),
        ("swap", "avoids the object it was trained on?", lambda r: "%d/%d" % (r.successes, r.trials)),
        ("occ", "needs vision?", lambda r: "%d/%d" % (r.successes, r.trials)),
        ("nonsense", "needs a real name?", lambda r: "%.1f%%" % r.rate),
    ]
    for face, question, fmt in rows:
        if face in results:
            print("%-10s %-38s %s" % (face, question, fmt(results[face])))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=None, help="checkpoint path (default: $CKPT)")
    ap.add_argument("--libero-pro", default=None, help="LIBERO-PRO checkout (default: $LIBERO_PRO)")
    ap.add_argument("--face", action="append", choices=sorted(FACES),
                    help="run only this face; repeatable (default: all four)")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--port", type=int, default=None, help="default: an unused port")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args = ap.parse_args(argv)

    results = run(faces=tuple(args.face) if args.face else ("task", "occ", "swap", "nonsense"),
                  ckpt=args.ckpt, libero_pro=args.libero_pro, port=args.port,
                  n_trials=args.n_trials, dry_run=args.dry_run)
    if results:
        _print_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
