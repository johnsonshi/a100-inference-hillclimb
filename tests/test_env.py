"""Environment settings: .env parsing, lane parsing, manifest rendering and the leak check's term matching.
Run: python3 -m tests.test_env"""
import importlib.util
import os
import tempfile
from pathlib import Path

from harness import settings

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ".env"
        path.write_text('# comment\nexport HC_TEST_A=one\nHC_TEST_B="two words"\n\nnot a pair\nHC_TEST_C=x=y\n')
        assert settings.read(path) == {"HC_TEST_A": "one", "HC_TEST_B": "two words", "HC_TEST_C": "x=y"}
        os.environ["HC_TEST_A"] = "from-env"
        settings.load(path)
        assert os.environ["HC_TEST_A"] == "from-env" and os.environ["HC_TEST_B"] == "two words"
        assert settings.read(Path(tmp) / "missing") == {}

    os.environ["HILLCLIMB_LANES"] = " n0=node-a , n1=node-b,bad,n2= "
    assert settings.lanes() == {"n0": "node-a", "n1": "node-b"}

    render = tool("render")
    vals = {"REGISTRY": "reg.example", "TAG": "abc"}
    assert render.render("image: __REGISTRY__/x:__TAG__ # __init__\n", vals) == "image: reg.example/x:abc # __init__\n"
    try:
        render.render("node: __NODE__", vals)
        raise AssertionError("unfilled placeholder must fail")
    except SystemExit as e:
        assert "NODE" in str(e)

    leaks = tool("check_leaks")
    needles = leaks.terms({"HILLCLIMB_LANES": "n0=node-alpha,n1=node-beta", "HILLCLIMB_PRIVATE_TERMS": "MyAlias, x",
                           "OTHER": "ignored-value"})
    assert needles == ["myalias", "node-alpha", "node-beta"], needles
    assert leaks.hits(b"pinned to NODE-ALPHA by myalias", needles) == ["myalias", "node-alpha"]
    assert leaks.hits(b"lane n0 on <node-0>", needles) == []

    # vLLM keeps its CPU KV offload region in /dev/shm and refuses to start if the tmpfs can't hold it
    from harness import runner
    for gib, want in ((0, "32Gi"), (64, "96Gi"), (128, "160Gi")):
        cfg = {"engine": "vllm", "variant": "int8w8a8", "kv_cache_dtype": "fp8", "kv_offloading_gib": gib}
        vols = runner.render_pod("t-shm", cfg, "t3", "node-a")[1]["spec"]["volumes"]
        assert next(v for v in vols if v["name"] == "dshm")["emptyDir"]["sizeLimit"] == want, (gib, vols)
    print("ENV TEST OK")


if __name__ == "__main__":
    main()
