"""Assembling the show's NN path out of the shipped artifacts."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

MODEL_VERSION = "ng_DACLEANTV_w128_s1234"
_GENERATION = "l9d"
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
    stream: object
    decoder: object
    feature_latency_sec: float

    def stop(self) -> None:
        stop = getattr(self.stream, "stop", None)
        if stop is not None:
            stop()


class SectionStream:
    """The posterior stream and the optional bar tracker as one stage.

    One object for both entry points: the GPU thread drives feed/due/run_pass,
    the inline simulation drives push_audio, and the mixed item list (Posterior
    and TrackerChunk) rides the existing hand-off unchanged.
    """

    def __init__(self, posteriors, tracker=None) -> None:
        self.posteriors = posteriors
        self.tracker = tracker

    @property
    def stream(self):
        return self.posteriors.stream

    @property
    def model(self):
        return self.posteriors.model

    def push_audio(self, samples):
        from lib.analyser.section_model import Drained

        self.feed(samples)
        out: list = []
        while self.due():
            out.extend(self.run_pass())
        return Drained(False, out)

    def feed(self, samples) -> None:
        self.posteriors.feed(samples)
        if self.tracker is not None:
            self.tracker.feed(samples)

    def due(self) -> bool:
        return (self.posteriors.due()
                or (self.tracker is not None and self.tracker.due()))

    def run_pass(self) -> list:
        if self.posteriors.due():
            return self.posteriors.run_pass()
        if self.tracker is not None and self.tracker.due():
            return self.tracker.run_pass()
        return []

    def resync(self):
        record = self.posteriors.resync()
        if self.tracker is not None:
            self.tracker.resync()
        return record

    def set_encoder(self, encoder) -> None:
        self.posteriors.set_encoder(encoder)

    def reset(self) -> None:
        self.posteriors.reset()
        if self.tracker is not None:
            self.tracker.reset()

    def stop(self) -> None:
        for stage in (self.posteriors, self.tracker):
            stop = getattr(stage, "stop", None)
            if stop is not None:
                stop()


def corpus_dir() -> Path:
    import sys

    training = str(Path(__file__).resolve().parents[1] / "training")
    if training not in sys.path:
        sys.path.insert(0, training)
    import corpus_root

    return corpus_root.corpus_dir()


def artifacts(data_dir=None) -> Artifacts:
    root = Path(data_dir) if data_dir is not None else corpus_dir()
    generation = root / "models" / _GENERATION
    return Artifacts(affine=generation / _AFFINE,
                     graph=generation / MODEL_VERSION / _GRAPH,
                     priors=generation / _PRIORS)


def artifacts_present(data_dir=None) -> bool:
    try:
        return not artifacts(data_dir).missing()
    except (OSError, ValueError) as error:
        logging.warning(f'[chain] cannot resolve the corpus directory '
                        f'({error!r}) — treating the model as absent')
        return False


def generation_dir(data_dir=None) -> Path:
    root = Path(data_dir) if data_dir is not None else corpus_dir()
    return root / "models" / _GENERATION


def bar_tracker_present(data_dir=None) -> bool:
    from lib.analyser import bar_tracker

    try:
        return bar_tracker.tracker_present(generation_dir(data_dir))
    except (OSError, ValueError):
        return False


class Geometry(NamedTuple):
    stream: object
    head: object
    mean: object


def read_geometry(data_dir=None) -> Geometry:
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
    from lib.analyser import mert_stream as M

    return {"device": device or M.best_device(),
            "precision": "fp16" if fp16 else "fp32"}


def _check_class_space(priors, model_classes, config_classes) -> None:
    """The GATED_SPACE flip's fatal gate: every layer speaks the full vocabulary.

    ``check_class_space`` tolerates a subset so the offline decoder can replay
    older generations; the chain that lights a room may not.  A model, a priors
    file and a decoder config that disagree about the class axis would decode
    shifted or commit classes nobody swept, so construction refuses anything
    short of the whole vocabulary, in order, at every layer (#268).
    """
    from lib.engine.effect_definitions import intent_for_class
    from lib.label_space import SECTION_LABELS, check_class_space

    check_class_space(priors.classes, "the decoder priors")
    expected = SECTION_LABELS
    if tuple(priors.classes) != expected:
        raise ValueError(
            f"the decoder priors name {len(priors.classes)} classes "
            f"({', '.join(priors.classes)}), not the full vocabulary "
            f"({', '.join(expected)}) the show decodes")
    if tuple(config_classes) != expected:
        raise ValueError(
            f"the decoder config's class_space is {', '.join(config_classes)}, "
            f"not the vocabulary {', '.join(expected)} -- it was swept in a "
            f"different space than the one it would decode")
    if model_classes != len(expected):
        raise ValueError(
            f"the model's label head is "
            f"{'undeclared' if model_classes is None else f'{model_classes}-wide'}"
            f" against the vocabulary's {len(expected)} classes -- the graph "
            f"and the priors would disagree about what each column means")
    for name in expected:
        intent_for_class(name)


def build_section_chain(data_dir=None, *, device: str | None = None,
                        fp16: bool = True, watchdog=None,
                        extractor=None, tracker=None) -> SectionChain:
    from lib.analyser import mert_stream as M
    from lib.analyser.section_model import PosteriorStream, SectionModel
    from lib.engine.section_decoder import (SHIPPING_DECODER_CONFIG, Priors,
                                            SectionDecoder,
                                            decoder_config_classes,
                                            load_decoder_config)

    found = artifacts(data_dir)
    geometry, head, mean = read_geometry(data_dir)

    # The space gate fires before the encoder loads: a chain whose layers
    # disagree about the class axis must refuse construction, not the first bar.
    model = SectionModel(found.graph, mean=mean, geometry=head)
    priors = Priors.load(found.priors)
    params = load_decoder_config(SHIPPING_DECODER_CONFIG)
    _check_class_space(priors, model.num_classes,
                       decoder_config_classes(SHIPPING_DECODER_CONFIG))

    stage = None if extractor is None else extractor(geometry)
    build_encoder = None
    if stage is None:
        backend = resolve_backend(device, fp16)

        def build_encoder():
            return M.load_encoder(geometry, device=backend["device"], fp16=fp16)

        stage = M.MertStream(build_encoder(), geometry=geometry)

    if tracker is None:
        tracker = _build_tracker(data_dir, device)

    stream = SectionStream(PosteriorStream(stage, model), tracker)
    if watchdog is not None:
        from lib.analyser.gpu_stage import GpuStage

        stream = GpuStage(stream, watchdog, reinit=build_encoder)
        stream.start()

    phase = None
    if tracker is not None:
        from lib.engine.bar_phase import BarPhaseFusion

        phase = BarPhaseFusion(tracker, watchdog)

    feature_latency_sec = (geometry.margin_sec + geometry.hop_sec
                           + head.future_sec)
    decoder = SectionDecoder(priors, params,
                             feature_latency_sec=feature_latency_sec,
                             phase=phase)
    logging.info(f'[chain] {MODEL_VERSION} on {_where(stage)} | '
                 f'feature latency {feature_latency_sec:.4f}s '
                 f'(F {geometry.margin_sec:g} + hop {geometry.hop_sec:g} + '
                 f'head {head.future_sec:g}) | bar grid: '
                 f'{"counting" if tracker is None else "fused tracker"}')
    return SectionChain(stream, decoder, feature_latency_sec)


def _build_tracker(data_dir, device):
    """The tracker is optional-absent, never optional-broken: no artifacts is
    the counting grid, artifacts that fail verification are fatal (#332)."""
    from lib.analyser import bar_tracker

    generation = generation_dir(data_dir)
    if not bar_tracker.tracker_present(generation):
        logging.info('[chain] no bar tracker artifacts — the bar grid is the '
                     'counting rule')
        return None
    return bar_tracker.load_bar_tracker(generation, device=device)


def _where(stage) -> str:
    encoder = getattr(stage, "_encoder", None)
    return "replayed cells" if encoder is None else encoder.device
