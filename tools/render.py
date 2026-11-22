"""Render a deploy/*.tmpl.yaml manifest to stdout. Fills __NAME__ placeholders from KEY=VALUE arguments and from
.env: __REGISTRY__ = HILLCLIMB_REGISTRY, __LANES__ = HILLCLIMB_LANES, __CONTROLLER_NODE__ = lane n0's node.
Fails if a placeholder is left unfilled. Example:
  python3 tools/render.py deploy/40-controller.tmpl.yaml TAG=<git-sha> CAMPAIGN=main-v3 \\
    | kubectl --context "$HILLCLIMB_KUBE_CONTEXT" apply -f -
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import settings  # noqa: E402

PLACEHOLDER = re.compile(r"__([A-Z][A-Z0-9_]*)__")


def values(args=()):
    settings.load()
    vals = {"REGISTRY": os.environ.get("HILLCLIMB_REGISTRY", "").rstrip("/"),
            "LANES": os.environ.get("HILLCLIMB_LANES", ""),
            "CONTROLLER_NODE": settings.lanes().get("n0", "")}
    for arg in args:
        key, sep, val = arg.partition("=")
        if not sep:
            raise SystemExit(f"render: bad argument {arg!r}, expected KEY=VALUE")
        vals[key] = val
    return {k: v for k, v in vals.items() if v}


def render(text, vals):
    out = PLACEHOLDER.sub(lambda m: vals.get(m.group(1), m.group(0)), text)
    left = sorted(set(PLACEHOLDER.findall(out)))
    if left:
        raise SystemExit(f"render: unfilled placeholders {', '.join(left)} (set them in .env or pass KEY=VALUE)")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    sys.stdout.write(render(Path(sys.argv[1]).read_text(), values(sys.argv[2:])))
