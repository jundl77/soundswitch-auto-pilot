"""An offline script may not let import order choose its decoder."""
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PHASE_B_MARKER = "soundswitch-phase-b-worktree"
DECOY_NN = {
    "__init__.py": "",
    "decoder.py": textwrap.dedent("""
        import dataclasses

        DECOY = True

        @dataclasses.dataclass(frozen=True)
        class DecodeParams:
            lag_bars: int = 2

        class FixedLagViterbi:
            def __init__(self, *args, **kwargs):
                pass

        def temper(posterior, temperature):
            return posterior
        """),
    "evaluate_v1.py": "def write_json(path, document):\n    pass\n",
    "priors.py": textwrap.dedent("""
        MODELS_DIR = "models"
        PRIORS_FILE = "priors.json"

        class Priors:
            @staticmethod
            def load(path):
                raise NotImplementedError
        """),
}


def _decoy_checkout(tmp_path):
    """A second checkout offering exactly the names the script imports.

    Import-compatible on purpose: a decoy that raised ImportError would be
    caught by any run, and the defect is the one that raises nothing.
    """
    nn = tmp_path / "decoy" / "training" / "nn"
    nn.mkdir(parents=True)
    for name, body in DECOY_NN.items():
        (nn / name).write_text(body, encoding="utf-8")
    return tmp_path / "decoy"


def _scripts_naming_another_checkout():
    """Scripts that both name a second checkout and import from `training.nn`.

    Naming one is not the hazard; a script that only hands the path to a
    subprocess picks nothing. Importing while two copies are reachable is.
    """
    found = []
    corpus = REPO / "training" / "data"
    for path in sorted((REPO / "training").rglob("*.py")):
        if corpus in path.parents:   # the gitignored corpus and its ops copies
            continue
        text = path.read_text(encoding="utf-8")
        if PHASE_B_MARKER in text and re.search(r"^\s*(from|import) training\.nn",
                                                text, re.M):
            found.append(path)
    return found


def test_the_pattern_is_still_worth_guarding():
    assert _scripts_naming_another_checkout()


@pytest.mark.parametrize("script", _scripts_naming_another_checkout(),
                         ids=lambda p: p.stem)
def test_a_script_naming_another_checkout_states_which_one_it_imports(script):
    assert "module_source.require(" in script.read_text(encoding="utf-8"), (
        f"{script.relative_to(REPO)} puts another checkout on sys.path without "
        f"declaring which one its training.nn must come from")


def test_a_decoder_already_imported_from_elsewhere_is_refused(tmp_path):
    decoy = _decoy_checkout(tmp_path)
    script = REPO / "training" / "decoder_rebirth" / "rebirth_case.py"
    program = textwrap.dedent(f"""
        import runpy, sys
        sys.path.insert(0, r"{decoy}")
        import training.nn.decoder
        runpy.run_path(r"{script}", run_name="probe")
        """)
    proc = subprocess.run([sys.executable, "-c", program],
                          capture_output=True, text=True, cwd=str(REPO))
    assert proc.returncode != 0, (
        "the script accepted a training.nn that a previous import chose for it")
    assert "training.nn resolved to" in proc.stderr
