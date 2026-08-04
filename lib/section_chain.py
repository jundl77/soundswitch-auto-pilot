"""Assembling the show's NN path out of the shipped artifacts."""
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
    stream: object
    decoder: object
    feature_latency_sec: float

    def stop(self) -> None:
        stop = getattr(self.stream, "stop", None)
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
    phase_b = root / "models" / _PHASE_B
    return Artifacts(affine=phase_b / _AFFINE,
                     graph=phase_b / MODEL_VERSION / _GRAPH,
                     priors=root / "models" / f"{_PHASE_B}_{MODEL_VERSION}"
                     / _PRIORS)


def artifacts_present(data_dir=None) -> bool:
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


def _check_class_space(priors) -> None:
    from lib.engine.effect_definitions import intent_for_class
    from lib.label_space import check_class_space

    check_class_space(priors.classes, "the decoder priors")
    for name in priors.classes:
        intent_for_class(name)


def build_section_chain(data_dir=None, *, device: str | None = None,
                        fp16: bool = True, watchdog=None,
                        extractor=None) -> SectionChain:
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
