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

    def stop(self) -> None:
        """Ends the GPU thread, if this chain has one.

        The one `getattr` in the wiring, and it is here rather than at each
        entry point: whether the stream is threaded is a property of how the
        chain was built, and nothing above it should have to remember.
        """
        stop = getattr(self.stream, "stop", None)
        if stop is not None:
            stop()


def corpus_dir() -> Path:
    """The gitignored corpus root, resolved the way every other reader does.

    `training/corpus_root.py` is stdlib-only and exists for this call: asking
    the benchmark harness where the corpus is pulled the whole eval pipeline --
    the table builder, the label evaluator, the raveform scripts and a git
    subprocess -- into a show's startup, for a path.
    """
    import sys

    training = str(Path(__file__).resolve().parents[1] / "training")
    if training not in sys.path:
        sys.path.insert(0, training)
    import corpus_root

    return corpus_root.corpus_dir()


def artifacts(data_dir=None) -> Artifacts:
    root = Path(data_dir) if data_dir is not None else corpus_dir()
    phase_b = root / "models" / _PHASE_B
    return Artifacts(affine=phase_b / _AFFINE,
                     graph=phase_b / MODEL_VERSION / _GRAPH,
                     priors=root / "models" / f"{_PHASE_B}_{MODEL_VERSION}"
                     / _PRIORS)


def artifacts_present(data_dir=None) -> bool:
    """Whether the show can be built -- a question about files, not imports.

    Narrow on purpose.  Swallowing everything meant any failure anywhere in the
    resolution reported "no NN artifacts on this machine": prod would play one
    held intent all night on a box that HAS the model, and the three tests that
    would catch it would skip citing artifacts sitting on disk.
    """
    try:
        return not artifacts(data_dir).missing()
    except (OSError, ValueError) as error:
        logging.warning(f'[chain] cannot resolve the corpus directory '
                        f'({error!r}) — treating the model as absent')
        return False


class Geometry(NamedTuple):
    stream: object
    head: object
    mean: object


def read_geometry(data_dir=None) -> Geometry:
    """Everything the shipped files record about shape, and no weights.

    Split out because the simulation's cell cache (D12) has to key on the
    extractor geometry before deciding whether this run needs a 1.3 GB encoder
    at all -- asking for it must not be what loads one.
    """
    from lib.analyser import mert_stream as M
    from lib.analyser.section_model import load_head_geometry

    found = artifacts(data_dir)
    absent = found.missing()
    if absent:
        raise FileNotFoundError(
            f"the shipped model artifacts are missing: {', '.join(absent)} -- "
            f"they live in the gitignored corpus directory, not in the "
            f"repository (see $RAVEFORM_DATA_DIR)")
    head = load_head_geometry(found.graph)
    stream = M.load_stream_geometry(found.affine,
                                    label_frame_sec=head.label_frame_sec)
    mean, _ = M.load_input_affine(found.affine)
    return Geometry(stream, head, mean)


def resolve_backend(device: str | None = None, fp16: bool = True) -> dict:
    """The arithmetic a chain built right now would run its encoder under.

    Neither half is requested by the callers that matter: `build_section_chain`
    takes `device=None` and picks `best_device()`, and `fp16` defaults on.  So
    the same call is cuda-fp16 on this box and cpu-fp32 on a box with no GPU,
    and those produce different cell bytes -- which makes this part of the
    cell cache's identity (#161's argument, applied to the numbers rather than
    to the decoder).  One function, read by the builder and by the key, so the
    two cannot answer differently.
    """
    from lib.analyser import mert_stream as M

    return {"device": device or M.best_device(),
            "precision": "fp16" if fp16 else "fp32"}


def _check_class_space(priors) -> None:
    """Every class the model can decode has to light something, at startup.

    `intent_for_class` raises on an unknown class, and it is called from inside
    the commit path -- so a retrained model naming a sixth class builds fine,
    runs fine, and kills the show at the first bar of that class, possibly an
    hour into a set.  D7 says a wrong look must stop the show being BUILT; this
    is where that becomes true.
    """
    from lib.engine.effect_definitions import intent_for_class

    for name in priors.classes:
        intent_for_class(name)


def build_section_chain(data_dir=None, *, device: str | None = None,
                        fp16: bool = True, watchdog=None,
                        extractor=None) -> SectionChain:
    """Encoder -> student -> committer, with the geometry read off the files.

    **A watchdog is the request for a thread** (D3).  The GPU stage reports its
    health to one and reads its shed level off it, so handing one over is
    exactly the statement "run this off the caller's thread"; without one the
    stages run inline, which is what the virtual-clock simulation needs and what
    keeps its reports byte-identical.  One switch, and it is the object the two
    halves would have had to share anyway.

    ``extractor`` is D12's seam: a callable handed the stream geometry that
    returns a replacement feature stage, or None.  When it returns one, no
    encoder is loaded at all -- which is the whole point of the simulation's
    cell cache, and the reason it is a factory rather than a built object.
    """
    from lib.analyser import mert_stream as M
    from lib.analyser.section_model import PosteriorStream, SectionModel
    from lib.engine.section_decoder import (SHIPPING_DECODER_CONFIG, Priors,
                                            SectionDecoder, load_decoder_config)

    found = artifacts(data_dir)
    geometry, head, mean = read_geometry(data_dir)

    stage = None if extractor is None else extractor(geometry)
    build_encoder = None
    if stage is None:
        backend = resolve_backend(device, fp16)

        def build_encoder():
            return M.load_encoder(geometry, device=backend["device"], fp16=fp16)

        stage = M.MertStream(build_encoder(), geometry=geometry)

    stream = PosteriorStream(stage, SectionModel(found.graph, mean=mean,
                                                 geometry=head))
    if watchdog is not None:
        from lib.analyser.gpu_stage import GpuStage

        stream = GpuStage(stream, watchdog, reinit=build_encoder)
        stream.start()

    # The half of B1's chain latency that cannot move: how much future audio a
    # cell's posterior depends on.  The other half is the decoder's, and it is
    # proportional to bar length, so the decoder measures its own.
    feature_latency_sec = (geometry.margin_sec + geometry.hop_sec
                           + head.future_sec)
    priors = Priors.load(found.priors)
    _check_class_space(priors)
    decoder = SectionDecoder(priors,
                             load_decoder_config(SHIPPING_DECODER_CONFIG),
                             feature_latency_sec=feature_latency_sec)
    logging.info(f'[chain] {MODEL_VERSION} on {_where(stage)} | '
                 f'feature latency {feature_latency_sec:.4f}s '
                 f'(F {geometry.margin_sec:g} + hop {geometry.hop_sec:g} + '
                 f'head {head.future_sec:g})')
    return SectionChain(stream, decoder, feature_latency_sec)


def _where(stage) -> str:
    encoder = getattr(stage, "_encoder", None)
    return "replayed cells" if encoder is None else encoder.device
