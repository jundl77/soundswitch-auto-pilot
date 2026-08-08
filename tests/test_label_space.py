import pytest

from lib.label_space import (
    DROPPED_LABELS,
    LABEL_INDEX,
    NUM_SECTION_CLASSES,
    SECTION_LABELS,
    check_class_space,
    is_dropped_label,
    is_section_label,
)


def test_the_vocabulary_is_the_raw_raveform_nine_in_arrangement_order():
    assert SECTION_LABELS == (
        'intro', 'altintro', 'buildup', 'breakdown', 'bridge',
        'drop', 'cooldown', 'outro', 'altoutro',
    )
    assert NUM_SECTION_CLASSES == 9


def test_end_is_the_only_dropped_label():
    assert DROPPED_LABELS == frozenset({'end'})
    assert is_dropped_label('end')
    assert not is_dropped_label('outro')
    assert not is_section_label('end')


def test_every_vocabulary_label_indexes_to_its_position():
    assert LABEL_INDEX == {label: i for i, label in enumerate(SECTION_LABELS)}
    assert all(is_section_label(label) for label in SECTION_LABELS)


@pytest.mark.parametrize('retired', ['canonical', 'v1', 'quiet', 'groove'])
def test_a_retired_space_name_is_not_a_label(retired):
    assert not is_section_label(retired)


def test_check_class_space_accepts_the_whole_vocabulary():
    check_class_space(SECTION_LABELS, 'priors.json')


def test_check_class_space_accepts_a_subset_in_vocabulary_order():
    # The shipping five-class chain: still every name the vocabulary knows, in
    # the vocabulary's own order, so it stays runnable until its successor lands.
    check_class_space(('intro', 'buildup', 'breakdown', 'drop', 'outro'), 'priors.json')


def test_check_class_space_rejects_a_name_the_vocabulary_does_not_know():
    with pytest.raises(ValueError) as excinfo:
        check_class_space(('intro', 'groove', 'drop'), 'priors.json')
    assert 'groove' in str(excinfo.value)
    assert 'priors.json' in str(excinfo.value)


def test_check_class_space_rejects_a_reordered_space():
    with pytest.raises(ValueError) as excinfo:
        check_class_space(('drop', 'intro', 'outro'), 'model.onnx')
    assert 'order' in str(excinfo.value).lower()
    assert 'model.onnx' in str(excinfo.value)


def test_check_class_space_rejects_a_duplicate():
    with pytest.raises(ValueError) as excinfo:
        check_class_space(('intro', 'intro', 'drop'), 'priors.json')
    assert 'duplicate' in str(excinfo.value).lower()


def test_check_class_space_rejects_an_empty_space():
    with pytest.raises(ValueError):
        check_class_space((), 'priors.json')
