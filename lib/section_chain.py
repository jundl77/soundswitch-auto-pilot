"""Assembling the show's NN path out of the shipped artifacts.

The three stages know nothing about each other's geometry by design, so
something has to hold the wiring, and both entry points -- `lib/main.py` and
`simulate/runner.py` -- need exactly the same wiring or sim=prod is a claim
about two different pipelines.  That is all this module is.

**Every number comes off an artifact** (D2).  The extractor geometry is read
from the input affine it was fitted under, the head's window and future from the
graph's own sidecar, and the feature latency -- the first half of B1's chain --
is computed from both rather than retyped.  The one constant here is which
model version ships, because a directory name is the only thing no file records.

**The artifacts are not in git and cannot be.**  1.3 GB of encoder plus the
student's graph and priors live in the gitignored corpus directory, so a fresh
clone can build every test that does not need the model and none that does.
`artifacts_present` is what lets a caller say that in one line instead of
crashing on a path.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

MODEL_VERSION = "student_kd_t2_w05_s1234"
_PHASE_B = "phase_b"
_AFFINE = "input_affine_F3.npz"
_GRAPH = "online_step.onnx"
_PRIORS = "priors.json"


class Artifacts(NamedTuple):
    affine: Path
    graph: Path
    priors: Path

    def missing(self) -> list:
        return [str(path) for path in self if not path.exists()]


class SectionChain(NamedTuple):
    """What the engine is handed: a posterior source and a committer."""

    stream: object
    decoder: object
    feature_latency_sec: float


def corpus_dir() -> Path:
    """The gitignored corpus root, resolved the way every other reader does.

    Imported late: `run_eval_set` is the benchmark harness and pulls the whole
    eval pipeline with it, which a show has no reason to pay for at import.
    """
    import sys

    training = str(Path(__file__).resolve().parents[1] / "training")
    if training not in sys.path:
        sys.path.insert(0, training)
    import run_eval_set

    return Path(run_eval_set.corpus_dir())


def artifacts(data_dir=None) -> Artifacts:
    root = Path(data_dir) if data_dir is not None else corpus_dir()
    phase_b = root / "models" / _PHASE_B
    return Artifacts(affine=phase_b / _AFFINE,
                     graph=phase_b / MODEL_VERSION / _GRAPH,
                     priors=root / "models" / f"{_PHASE_B}_{MODEL_VERSION}"
                     / _PRIORS)


def artifacts_present(data_dir=None) -> bool:
    try:
        return not artifacts(data_dir).missing()
    except Exception:
        return False


def build_section_chain(data_dir=None, *, device: str | None = None,
                        fp16: bool = True) -> SectionChain:
    """Encoder -> student -> committer, with the geometry read off the files."""
    from lib.analyser import mert_stream as M
    from lib.analyser.section_model import (PosteriorStream, SectionModel,
                                            load_head_geometry)
    from lib.engine.section_decoder import (SHIPPING_DECODER_CONFIG, Priors,
                                            SectionDecoder, load_decoder_config)

    found = artifacts(data_dir)
    absent = found.missing()
    if absent:
        raise FileNotFoundError(
            f"the shipped model artifacts are missing: {', '.join(absent)} -- "
            f"they live in the gitignored corpus directory, not in the "
            f"repository (see $RAVEFORM_DATA_DIR)")

    head = load_head_geometry(found.graph)
    geometry = M.load_stream_geometry(found.affine,
                                      label_frame_sec=head.label_frame_sec)
    mean, _ = M.load_input_affine(found.affine)

    encoder = M.load_encoder(geometry, device=device or M.best_device(),
                             fp16=fp16)
    stream = PosteriorStream(M.MertStream(encoder, geometry=geometry),
                             SectionModel(found.graph, mean=mean,
                                          geometry=head))

    # The half of B1's chain latency that cannot move: how much future audio a
    # cell's posterior depends on.  The other half is the decoder's, and it is
    # proportional to bar length, so the decoder measures its own.
    feature_latency_sec = (geometry.margin_sec + geometry.hop_sec
                           + head.future_sec)
    decoder = SectionDecoder(Priors.load(found.priors),
                             load_decoder_config(SHIPPING_DECODER_CONFIG),
                             feature_latency_sec=feature_latency_sec)
    logging.info(f'[chain] {MODEL_VERSION} on {encoder.device} | '
                 f'feature latency {feature_latency_sec:.4f}s '
                 f'(F {geometry.margin_sec:g} + hop {geometry.hop_sec:g} + '
                 f'head {head.future_sec:g})')
    return SectionChain(stream, decoder, feature_latency_sec)
