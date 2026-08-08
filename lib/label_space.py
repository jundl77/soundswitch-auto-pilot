"""The repo's one label vocabulary: the raw Raveform sections, unfolded.

Everything that names a musical section reads it from here -- the corpus join,
the dataset targets, the decoder priors, the evaluator and the show's intent
mapping. The canonical-7 and label_v1-5 folds this replaced are retired; the
only surviving fold is a reporting view in the evaluator, applied to already
joined labels so no pipeline stage ever stores a folded one.

It lives beside ``audio_config`` rather than under ``training/`` because the
runtime is a consumer too: the decoder commits these names and the engine maps
them to lights, and ``lib`` may not import the training tree.
"""

from typing import Iterable

SECTION_LABELS = (
    'intro',
    'altintro',
    'buildup',
    'breakdown',
    'bridge',
    'drop',
    'cooldown',
    'outro',
    'altoutro',
)

# `end` marks where the annotation stops, not a phase anyone plays. Time inside
# it belongs to no section and is dropped rather than re-attributed.
DROPPED_LABELS = frozenset({'end'})

NUM_SECTION_CLASSES = len(SECTION_LABELS)
LABEL_INDEX = {label: index for index, label in enumerate(SECTION_LABELS)}


def is_section_label(label: str) -> bool:
    return label in LABEL_INDEX


def is_dropped_label(label: str) -> bool:
    return label in DROPPED_LABELS


def check_class_space(classes: Iterable[str], source: str) -> None:
    """Refuse a class space that is not vocabulary names in vocabulary order.

    The checkpoint-geometry discipline applied to the class axis. A priors file,
    an exported graph and a decoder config each carry the space positionally --
    per-class floors and duration hazards are indexed lists -- so a space that
    is merely the right *length* loads cleanly and decodes the wrong classes.
    Order is checked for that reason, and a subset is allowed so the shipping
    chain keeps running until its successor is trained.
    """
    names = tuple(classes)
    if not names:
        raise ValueError(f"{source} declares an empty class space")
    unknown = [name for name in names if name not in LABEL_INDEX]
    if unknown:
        raise ValueError(
            f"{source} names {', '.join(repr(name) for name in unknown)}, which "
            f"the label vocabulary does not know; it is {', '.join(SECTION_LABELS)}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"{source} lists a duplicate class: {', '.join(names)}")
    expected = tuple(sorted(names, key=LABEL_INDEX.__getitem__))
    if names != expected:
        raise ValueError(
            f"{source} lists its classes in the wrong order: {', '.join(names)}; "
            f"the vocabulary's order is {', '.join(expected)}"
        )
