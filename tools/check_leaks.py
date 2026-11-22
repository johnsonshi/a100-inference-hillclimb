"""Leak check: fail if anything environment-specific would reach git. The terms are every value in .env (kube
context, node names, registry, subscription, ...) plus HILLCLIMB_PRIVATE_TERMS, matched case-insensitively, so the
list itself is never committed.
  python3 tools/check_leaks.py --staged    paths and contents staged for commit (pre-commit hook)
  python3 tools/check_leaks.py --history   every path, file version and commit message reachable from any ref,
                                           and every author/committer email (pre-push hook; default)
Commit emails must be GitHub noreply addresses. Install both hooks with tools/install_hooks.sh and never bypass
them (git --no-verify): scrub the content instead.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from harness import settings  # noqa: E402

EMAIL_SUFFIX = "@users.noreply.github.com"
MIN_LEN = 4  # shorter values (lane ids such as n0) are too generic to search for


def terms(values):
    """Search terms from parsed .env values: every HILLCLIMB_* value, lists split on commas and '='."""
    found = set()
    for key, value in values.items():
        if key.startswith("HILLCLIMB_"):
            for part in value.replace("=", ",").split(","):
                part = part.strip().lower()
                if len(part) >= MIN_LEN:
                    found.add(part)
    return sorted(found)


def hits(data, needles):
    low = data.lower()
    return [n for n in needles if n.encode() in low]


def git(*args, stdin=None):
    return subprocess.run(["git", "-C", str(REPO), *args], input=stdin, capture_output=True, check=True).stdout


def staged(needles):
    problems = []
    for name in filter(None, git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").decode().split("\0")):
        for n in hits(name.encode(), needles):
            problems.append(f"path {name} contains {n!r}")
        for n in hits(git("show", f":{name}"), needles):
            problems.append(f"staged {name} contains {n!r}")
    return problems


def history(needles):
    problems, paths = [], {}
    for line in git("rev-list", "--objects", "--all").decode().splitlines():
        sha, _, path = line.partition(" ")
        paths[sha] = path
        for n in hits(path.encode(), needles):
            problems.append(f"path {path} contains {n!r}")
    batch = git("cat-file", "--batch", stdin="\n".join(paths).encode() + b"\n")
    pos = 0
    while pos < len(batch):
        end = batch.index(b"\n", pos)
        sha, kind, size = batch[pos:end].decode().split()
        body = batch[end + 1:end + 1 + int(size)]
        pos = end + 2 + int(size)
        if kind == "blob":
            for n in hits(body, needles):
                problems.append(f"{paths[sha] or sha} (blob {sha[:8]}) contains {n!r}")
        elif kind == "commit":
            head, _, message = body.partition(b"\n\n")
            for row in head.decode(errors="replace").splitlines():
                if row.startswith(("author ", "committer ")):
                    email = row[row.find("<") + 1:row.find(">")]
                    if not email.endswith(EMAIL_SUFFIX):
                        problems.append(f"commit {sha[:8]} {row.split()[0]} email {email!r} is not a noreply address")
            for n in hits(message, needles):
                problems.append(f"commit {sha[:8]} message contains {n!r}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--staged", action="store_true", help="check the index instead of the whole history")
    args = ap.parse_args()
    needles = terms(settings.read())
    if not needles:
        sys.exit("check_leaks: no .env with HILLCLIMB_* values, so nothing to check against (see .env.example)")
    problems = staged(needles) if args.staged else history(needles)
    for p in problems:
        print("check_leaks:", p, file=sys.stderr)
    if problems:
        sys.exit(f"check_leaks: {len(problems)} problem(s); scrub them (placeholders as in .env.example), "
                 "never bypass with --no-verify")
    print(f"check_leaks: clean ({'staged files' if args.staged else 'all reachable history'}, {len(needles)} terms)")


if __name__ == "__main__":
    main()
