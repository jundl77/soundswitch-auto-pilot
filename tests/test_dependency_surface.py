"""What the show is allowed to depend on, asserted against pyproject and lib/.

The NN integration inverts this project's dependency story.  torch,
transformers and onnxruntime stop being offline research tools and become the
things that run the lights, so they belong in base.  TensorFlow and aubio stop
being the front-end and become nothing, so they leave -- and the two claims are
independent: declaring the new stack was done first, deleting the old one's
importers was the demolition.  The two tests that describe the end state were
strict xfails until the demolition landed and they XPASSed; the markers came off
in that commit, which is the only reason they are unmarked now.

"Does lib/ still import at all" stays a separate, unmarked question.  Folding it
into the retired-module reading would let an import-time crash satisfy the
retired-module claim vacuously -- a crashed probe prints nothing.
"""
import ast
import subprocess
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
LIVE_PATH = REPO_ROOT / "lib"

RETIRED = ("tensorflow", "tensorflow_hub", "aubio")


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _names(requirements) -> set:
    """Distribution names out of a requirements list, markers and pins stripped."""
    names = set()
    for requirement in requirements:
        head = requirement.split(";")[0]
        for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
            head = head.split(separator)[0]
        names.add(head.strip().lower().replace("_", "-"))
    return names


def _imported_modules(path: Path) -> set:
    """Top-level module names imported anywhere in a file, lazy imports included."""
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
    """Two pins for one package is two answers to which version ships."""
    project = _pyproject()["project"]
    base = _names(project["dependencies"])
    for extra, requirements in project.get("optional-dependencies", {}).items():
        overlap = base & _names(requirements)
        assert not overlap, f"{extra} re-pins base dependencies: {sorted(overlap)}"


def test_the_torch_index_is_bound_to_torch_alone():
    """`explicit = true` keeps uv from shadowing PyPI with an incomplete mirror."""
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
    """Import every module under lib/ in a fresh process, once.

    Every module, not just the entry point: `lib.main` defers almost everything
    it needs, so importing it alone would pass whether the front-end had been
    removed or not.
    """
    if not _probe_result:
        probe = (
            "import sys, importlib;"
            f"[importlib.import_module(name) for name in {_live_modules()!r}];"
            f"print(','.join(sorted(m for m in {RETIRED!r} if m in sys.modules)))"
        )
        _probe_result.append(subprocess.run(
            [sys.executable, "-c", probe], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=600))
    return _probe_result[0]


def test_every_module_on_the_live_path_still_imports():
    """Unmarked and unconditional, and separate from the reading below.

    A strict xfail swallows every reason its assertion could fail, so folding
    this into it would let an import-time crash Task 4 leaves behind read as
    "expected failure" while lib/ was broken -- and the demolition-done signal
    would arrive only via other tests failing for less obvious reasons.
    """
    result = _load_the_live_path()
    assert result.returncode == 0, result.stderr[-2000:]


def test_loading_the_whole_live_path_pulls_in_neither_tensorflow_nor_aubio():
    """The static scan cannot see a transitive import; a real load can.

    The return-code assertion is repeated here rather than left to the test
    above: a crashed probe prints nothing, so without it the retired-module
    claim would hold vacuously and announce a demolition that had not happened.
    """
    result = _load_the_live_path()
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == ""


def test_the_two_playback_delays_are_one_number():
    """`lib/main.py` and `simulate/runner.py` each carry their own literal,
    joined only by a comment -- and the whole sim=prod contract, the eval-set
    baseline and every report's `look_ahead_sec` rest on them agreeing.  A
    change to one silently re-scores the benchmark against a pipeline
    production is not running."""
    import lib.main
    import simulate.runner

    assert lib.main.PLAYBACK_DELAY_SEC == simulate.runner.PLAYBACK_DELAY_SEC
