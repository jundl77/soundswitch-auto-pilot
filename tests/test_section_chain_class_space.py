import types

import pytest

from lib.label_space import SECTION_LABELS
from lib.section_chain import _check_class_space


def _priors(*classes):
    return types.SimpleNamespace(classes=tuple(classes))


def test_a_nine_class_priors_file_is_accepted():
    _check_class_space(_priors(*SECTION_LABELS))


def test_the_shipping_five_class_priors_file_still_loads():
    _check_class_space(_priors('intro', 'buildup', 'breakdown', 'drop', 'outro'))


def test_a_class_outside_the_vocabulary_is_refused():
    with pytest.raises(ValueError, match='chorus'):
        _check_class_space(_priors('intro', 'chorus'))


def test_every_vocabulary_class_survives_the_intent_lookup_too():
    # The vocabulary gate fires first, so the intent loop behind it is only
    # reachable for names the vocabulary knows -- all of which must resolve.
    for label in SECTION_LABELS:
        _check_class_space(_priors(label))


def test_a_space_in_the_wrong_order_is_refused_rather_than_decoded_shifted():
    with pytest.raises(ValueError, match='order'):
        _check_class_space(_priors('drop', 'intro', 'outro'))


def test_a_duplicated_class_is_refused():
    with pytest.raises(ValueError, match='duplicate'):
        _check_class_space(_priors('intro', 'intro'))
