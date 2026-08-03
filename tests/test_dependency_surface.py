import ast
import subprocess
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
LIVE_PATH = REPO_ROOT / "lib"

RETIRED = ("tensorflow", "tensorflow_hub", "aubio")
# The viewer runs in its own process; a GIL shared with it costs the show sheds.
VIEWER_ONLY = ("dash", "flask", "plotly")


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _names(requirements) -> set:
    names = set()
    for requirement in requirements:
        head = requirement.split(";")[0]
        for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
            head = head.split(separator)[0]
        names.add(head.strip().lower().replace("_", "-"))
    return names


def _imported_modules(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def test_the_show_declares_the_stack_it_runs_on():
    base = _names(_pyproject()["project"]["dependencies"])
    assert {"torch", "transformers", "onnxruntime"} <= base, (
        "the model, its encoder and its inference session run the lights -- an "
        "undeclared dependency on the live path is one a fresh clone does not get")


def test_no_extra_re_pins_what_base_already_declares():
    project = _pyproject()["project"]
    base = _names(project["dependencies"])
    for extra, requirements in project.get("optional-dependencies", {}).items():
        overlap = base & _names(requirements)
        assert not overlap, f"{extra} re-pins base dependencies: {sorted(overlap)}"


def test_the_torch_index_is_bound_to_torch_alone():
    uv = _pyproject()["tool"]["uv"]
    indexes = {index["name"]: index for index in _pyproject()["tool"]["uv"]["index"]}
    assert indexes["pytorch-cu128"]["explicit"] is True
    assert uv["sources"]["torch"] == [{"index": "pytorch-cu128"}]


def test_the_live_path_imports_neither_tensorflow_nor_aubio():
    offenders = {}
    for path in sorted(LIVE_PATH.rglob("*.py")):
        found = _imported_modules(path) & set(RETIRED)
        if found:
            offenders[str(path.relative_to(REPO_ROOT))] = sorted(found)
    assert not offenders, offenders


def _live_modules() -> list:
    return sorted(".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
                  for path in LIVE_PATH.rglob("*.py")
                  if path.name != "__init__.py")


_probe_result: list = []


def _load_the_live_path():
    if not _probe_result:
        watched = RETIRED + VIEWER_ONLY
        probe = (
            "import sys, importlib;"
            f"[importlib.import_module(name) for name in {_live_modules()!r}];"
            f"print(','.join(sorted(m for m in {watched!r} if m in sys.modules)))"
        )
        _probe_result.append(subprocess.run(
            [sys.executable, "-c", probe], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=600))
    return _probe_result[0]


def _pulled_in() -> set:
    result = _load_the_live_path()
    assert result.returncode == 0, result.stderr[-2000:]
    return set(filter(None, result.stdout.strip().split(",")))


def test_every_module_on_the_live_path_still_imports():
    result = _load_the_live_path()
    assert result.returncode == 0, result.stderr[-2000:]


def test_loading_the_whole_live_path_pulls_in_neither_tensorflow_nor_aubio():
    assert _pulled_in() & set(RETIRED) == set()


def test_the_show_process_never_imports_the_viewer():
    assert _pulled_in() & set(VIEWER_ONLY) == set()


def test_the_two_playback_delays_are_one_number():
    import lib.main
    import simulate.runner

    assert lib.main.PLAYBACK_DELAY_SEC == simulate.runner.PLAYBACK_DELAY_SEC
