"""Environment-specific settings (kubectl context, lane nodes, image registry) come from the process environment
or from a gitignored .env at the repo root (template: .env.example), so nothing environment-specific is committed.
Inside the cluster there is no .env: the controller Deployment sets the variables (tools/deploy.sh)."""
import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def read(path=ENV_FILE):
    """KEY=VALUE lines of a .env file as a dict ({} if the file is missing). Comments, blank lines and an
    `export ` prefix are ignored; one pair of surrounding quotes is stripped."""
    path = Path(path)
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load(path=ENV_FILE):
    """Copy .env into os.environ without overriding variables that are already set."""
    for key, value in read(path).items():
        os.environ.setdefault(key, value)


def lanes():
    """HILLCLIMB_LANES="n0=<node>,n1=<node>" as {"n0": "<node>", "n1": "<node>"}."""
    pairs = (item.split("=", 1) for item in os.environ.get("HILLCLIMB_LANES", "").split(",") if "=" in item)
    return {lane.strip(): node.strip() for lane, node in pairs if lane.strip() and node.strip()}
