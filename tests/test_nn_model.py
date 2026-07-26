"""Tests for the CRNN and its training objective (``training/nn/model.py``,
``training/nn/train.py``).  CPU only -- nothing here needs the 3070.

Three families, all chosen because they fail *silently* on a GPU run:

**Shapes.**  The label head speaks at half the frame rate and the boundary head
at the full rate; the dataset hands back exactly that pair of lengths.  If the
two ever disagree by a frame the losses still compute (broadcasting is happy to
oblige) and the model trains on shifted targets.

**Losses.**  Every one of the three terms is masked, and a mask that silently
does nothing is the failure mode: a fully-masked batch must contribute exactly
zero, and values at masked positions must not move the loss at all.  Focal loss
is additionally pinned against ``cross_entropy`` at gamma=0 so the formulation
itself cannot drift.

**Metrics.**  These are the numbers the sanity gates are read off.  Each is
checked against a hand-computed value rather than against itself.
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="training extra not synced")
import torch.nn.functional as F  # noqa: E402

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from nn.dataset import (  # noqa: E402
    IGNORE_INDEX,
    LABEL_FRAMES,
    LABEL_POOL,
    NUM_CLASSES,
    WINDOW_FRAMES,
    TrackTargets,
)
from nn.model import PARAM_BUDGET, SectionCRNN, count_parameters  # noqa: E402
from nn.train import (  # noqa: E402
    accumulate_target_stats,
    boundary_bce,
    boundary_pos_weight,
    capture_rng,
    class_weights,
    confusion_matrix,
    focal_loss,
    macro_f1,
    per_class_ece,
    per_class_f1,
    pr_auc,
    restore_rng,
    select_smoke_subset,
    tv_penalty,
    weight_hash,
)

MEL_BANDS = 40


def _model(**kwargs) -> SectionCRNN:
    torch.manual_seed(0)
    return SectionCRNN(**kwargs).eval()


# --------------------------------------------------------------------------- #
# Model shape / capacity
# --------------------------------------------------------------------------- #


def test_forward_returns_the_dataset_target_shapes():
    model = _model()
    mel = torch.zeros(3, WINDOW_FRAMES, MEL_BANDS)

    labels, boundary = model(mel)

    assert labels.shape == (3, LABEL_FRAMES, NUM_CLASSES)
    assert boundary.shape == (3, WINDOW_FRAMES)
    assert LABEL_FRAMES == WINDOW_FRAMES // LABEL_POOL


@pytest.mark.parametrize("frames", [WINDOW_FRAMES, 512, 64])
def test_time_axis_is_not_baked_in(frames):
    """The decoder runs whole tracks through in one pass, and the ONNX graph
    declares a dynamic time axis -- so the module must accept any length."""
    labels, boundary = _model()(torch.zeros(1, frames, MEL_BANDS))

    assert labels.shape == (1, frames // LABEL_POOL, NUM_CLASSES)
    assert boundary.shape == (1, frames)


def test_parameter_count_is_within_budget():
    count = count_parameters(_model())

    assert count <= PARAM_BUDGET, f"{count} params exceeds the {PARAM_BUDGET} budget"
    # A floor too: an architecture edit that accidentally drops the GRU or the
    # 1D conv would still pass every shape test above.
    assert count > 300_000


def test_frequency_is_pooled_away_and_time_is_preserved():
    model = _model()
    # 40 mel bands, halved by each of the three conv blocks.
    assert model.freq_out == MEL_BANDS // 8
    assert model.feature_dim == 64 * (MEL_BANDS // 8)


def test_mel_band_count_must_survive_the_frequency_pooling():
    with pytest.raises(ValueError, match="frequency"):
        SectionCRNN(n_mels=12)


def test_forward_rejects_a_mis_shaped_batch():
    with pytest.raises(ValueError, match="mel"):
        _model()(torch.zeros(2, 1, WINDOW_FRAMES, MEL_BANDS))


def test_eval_forward_is_deterministic():
    model = _model()
    mel = torch.randn(2, WINDOW_FRAMES, MEL_BANDS, generator=torch.Generator().manual_seed(7))

    first = model(mel)
    second = model(mel)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_both_heads_reach_every_parameter():
    """A head wired to the wrong tensor still trains -- just not end to end."""
    model = SectionCRNN().train()
    labels, boundary = model(torch.randn(2, WINDOW_FRAMES, MEL_BANDS))
    (labels.square().mean() + boundary.square().mean()).backward()

    missing = [name for name, param in model.named_parameters()
               if param.grad is None or not torch.isfinite(param.grad).all()]
    assert missing == []


# --------------------------------------------------------------------------- #
# Focal loss
# --------------------------------------------------------------------------- #


def _weights(scale=1.0) -> torch.Tensor:
    return torch.full((NUM_CLASSES,), float(scale))


def test_focal_loss_at_gamma_zero_is_cross_entropy():
    """The one formulation check: gamma=0 with unit weights collapses to the
    reference implementation, ignore_index and all."""
    torch.manual_seed(3)
    logits = torch.randn(4, 6, NUM_CLASSES)
    targets = torch.randint(0, NUM_CLASSES, (4, 6))
    targets[0, :3] = IGNORE_INDEX

    got = focal_loss(logits, targets, weight=_weights(), gamma=0.0)
    want = F.cross_entropy(logits.reshape(-1, NUM_CLASSES), targets.reshape(-1),
                           ignore_index=IGNORE_INDEX)

    assert torch.allclose(got, want, atol=1e-6)


def test_focal_loss_ignores_masked_positions_entirely():
    torch.manual_seed(4)
    logits = torch.randn(2, 5, NUM_CLASSES)
    targets = torch.randint(0, NUM_CLASSES, (2, 5))
    targets[:, :2] = IGNORE_INDEX

    baseline = focal_loss(logits, targets, weight=_weights(), gamma=2.0)
    logits[:, :2] += 100.0  # nonsense where nothing is supervised
    after = focal_loss(logits, targets, weight=_weights(), gamma=2.0)

    assert torch.allclose(baseline, after, atol=1e-6)


def test_focal_loss_of_a_fully_masked_batch_is_zero_and_differentiable():
    logits = torch.randn(2, 5, NUM_CLASSES, requires_grad=True)
    targets = torch.full((2, 5), IGNORE_INDEX)

    loss = focal_loss(logits, targets, weight=_weights(), gamma=2.0)
    loss.backward()

    assert float(loss.detach()) == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))


def test_focal_loss_discounts_easy_examples():
    """gamma is the whole point: a confident correct frame must weigh less than
    it does under plain cross-entropy, an uncertain one essentially the same."""
    easy = torch.tensor([[[8.0, 0.0, 0.0, 0.0, 0.0]]])
    hard = torch.tensor([[[0.2, 0.0, 0.0, 0.0, 0.0]]])
    target = torch.zeros(1, 1, dtype=torch.long)

    easy_ratio = (focal_loss(easy, target, weight=_weights(), gamma=2.0)
                  / focal_loss(easy, target, weight=_weights(), gamma=0.0))
    hard_ratio = (focal_loss(hard, target, weight=_weights(), gamma=2.0)
                  / focal_loss(hard, target, weight=_weights(), gamma=0.0))

    assert float(easy_ratio) < 1e-3
    assert float(hard_ratio) > 0.5


def test_focal_loss_scales_with_the_class_weight():
    logits = torch.zeros(1, 1, NUM_CLASSES)
    target = torch.zeros(1, 1, dtype=torch.long)
    weight = _weights()
    weight[0] = 3.0

    assert torch.allclose(focal_loss(logits, target, weight=weight, gamma=2.0),
                          3.0 * focal_loss(logits, target, weight=_weights(), gamma=2.0))


def test_class_weights_are_inverse_frequency_normalised():
    counts = np.array([100, 100, 100, 100, 400], dtype=np.int64)

    weights = class_weights(counts)

    assert weights[0] == pytest.approx(4.0 * weights[4])
    assert float(weights.mean()) == pytest.approx(1.0)


def test_class_weights_survive_an_absent_class():
    """A 10-track smoke subset can genuinely lack a class; a division by zero
    here would poison every gradient in the run with NaN."""
    weights = class_weights(np.array([10, 0, 10, 10, 10], dtype=np.int64))

    assert np.isfinite(weights).all()
    assert weights[1] == 0.0


# --------------------------------------------------------------------------- #
# Boundary loss
# --------------------------------------------------------------------------- #


def test_boundary_bce_at_unit_pos_weight_is_masked_mean_bce():
    torch.manual_seed(5)
    logits = torch.randn(2, 8)
    targets = torch.rand(2, 8)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, 2:] = True

    got = boundary_bce(logits, targets, mask, pos_weight=1.0)
    want = F.binary_cross_entropy_with_logits(logits[:, 2:], targets[:, 2:])

    assert torch.allclose(got, want, atol=1e-6)


def test_boundary_bce_ignores_deleted_positions():
    torch.manual_seed(6)
    logits = torch.randn(2, 8)
    targets = torch.rand(2, 8)
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[:, :3] = False

    baseline = boundary_bce(logits, targets, mask, pos_weight=4.0)
    logits[:, :3] -= 50.0
    after = boundary_bce(logits, targets, mask, pos_weight=4.0)

    assert torch.allclose(baseline, after, atol=1e-6)


def test_boundary_bce_of_a_fully_masked_batch_is_zero():
    logits = torch.randn(2, 8, requires_grad=True)
    loss = boundary_bce(logits, torch.rand(2, 8), torch.zeros(2, 8, dtype=torch.bool),
                        pos_weight=4.0)
    loss.backward()

    assert float(loss.detach()) == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))


def test_boundary_pos_weight_is_the_negative_to_positive_ratio():
    # 1000 valid frames carrying 50 frames' worth of positive mass.
    assert boundary_pos_weight(50.0, 1000) == pytest.approx(19.0)


def test_boundary_pos_weight_falls_back_when_there_is_no_positive_mass():
    assert boundary_pos_weight(0.0, 1000) == pytest.approx(1.0)
    assert boundary_pos_weight(0.0, 0) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Total-variation smoothness
# --------------------------------------------------------------------------- #


def _tv_inputs(logits, *, boundary_value=0.0):
    frames = logits.shape[1] * LABEL_POOL
    label_mask = torch.ones(logits.shape[:2], dtype=torch.bool)
    boundary = torch.full((logits.shape[0], frames), float(boundary_value))
    boundary_mask = torch.ones_like(boundary, dtype=torch.bool)
    return label_mask, boundary, boundary_mask


def test_tv_penalty_is_zero_for_a_constant_posterior():
    logits = torch.zeros(1, 6, NUM_CLASSES)
    logits[:, :, 2] = 4.0

    assert float(tv_penalty(logits, *_tv_inputs(logits), pool=LABEL_POOL)) == pytest.approx(0.0)


def test_tv_penalty_punishes_a_flickering_posterior():
    logits = torch.zeros(1, 6, NUM_CLASSES)
    logits[:, ::2, 0] = 8.0
    logits[:, 1::2, 1] = 8.0

    assert float(tv_penalty(logits, *_tv_inputs(logits), pool=LABEL_POOL)) > 0.1


def test_tv_penalty_stands_down_at_a_boundary():
    """A real section change is a step in the posterior; penalising it would
    teach the net to smear exactly the event the decoder needs."""
    logits = torch.zeros(1, 6, NUM_CLASSES)
    logits[:, ::2, 0] = 8.0
    logits[:, 1::2, 1] = 8.0

    at_boundary = tv_penalty(logits, *_tv_inputs(logits, boundary_value=1.0), pool=LABEL_POOL)

    assert float(at_boundary) == pytest.approx(0.0)


def test_tv_penalty_skips_unsupervised_frames():
    logits = torch.zeros(1, 6, NUM_CLASSES)
    logits[:, ::2, 0] = 8.0
    logits[:, 1::2, 1] = 8.0
    label_mask, boundary, boundary_mask = _tv_inputs(logits)
    label_mask[:] = False

    assert float(tv_penalty(logits, label_mask, boundary, boundary_mask,
                            pool=LABEL_POOL)) == pytest.approx(0.0)


def test_tv_penalty_skips_frames_whose_boundary_target_was_deleted():
    """Where the boundary target is deleted we do not know whether a step is
    legitimate, so the penalty must abstain rather than guess it is not."""
    logits = torch.zeros(1, 6, NUM_CLASSES)
    logits[:, ::2, 0] = 8.0
    logits[:, 1::2, 1] = 8.0
    label_mask, boundary, boundary_mask = _tv_inputs(logits)
    boundary_mask[:] = False

    assert float(tv_penalty(logits, label_mask, boundary, boundary_mask,
                            pool=LABEL_POOL)) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_confusion_matrix_is_true_by_predicted():
    true = np.array([0, 0, 1, 2])
    pred = np.array([0, 1, 1, 1])

    matrix = confusion_matrix(true, pred, NUM_CLASSES)

    assert matrix[0, 0] == 1 and matrix[0, 1] == 1
    assert matrix[1, 1] == 1 and matrix[2, 1] == 1
    assert matrix.sum() == 4


def test_per_class_f1_matches_the_hand_computed_value():
    # class 0: tp=1, fp=0, fn=1 -> P=1, R=0.5, F1=2/3
    # class 1: tp=1, fp=2, fn=0 -> P=1/3, R=1, F1=0.5
    matrix = confusion_matrix(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 1]), NUM_CLASSES)

    scores = per_class_f1(matrix)

    assert scores[0] == pytest.approx(2 / 3)
    assert scores[1] == pytest.approx(0.5)
    assert scores[2] == pytest.approx(0.0)


def test_macro_f1_averages_only_over_classes_that_occur():
    """Averaging over all five when the batch contains two would report 0.27
    for a perfect classifier -- and the sanity gate is read off this number."""
    matrix = confusion_matrix(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), NUM_CLASSES)

    assert macro_f1(matrix) == pytest.approx(1.0)


def test_macro_f1_counts_a_class_the_model_hallucinates():
    matrix = confusion_matrix(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 3]), NUM_CLASSES)

    # class 0: P=1, R=0.75 -> F1=6/7; class 3: never true, once predicted -> 0.
    assert macro_f1(matrix) == pytest.approx((6 / 7) / 2)


def test_pr_auc_matches_average_precision_by_hand():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels = np.array([1, 0, 1, 0])

    assert pr_auc(scores, labels) == pytest.approx(0.5 + 1 / 3)


def test_pr_auc_of_a_perfect_ranking_is_one():
    assert pr_auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == pytest.approx(1.0)


def test_pr_auc_without_a_positive_is_not_a_number():
    """Reporting 0.0 would look like a broken model instead of an empty metric."""
    assert math.isnan(pr_auc(np.array([0.9, 0.1]), np.array([0, 0])))


def test_ece_is_zero_when_confidence_matches_accuracy():
    probs = np.full((100, NUM_CLASSES), 0.2)
    labels = np.arange(100) % NUM_CLASSES

    assert per_class_ece(probs, labels)[0] == pytest.approx(0.0, abs=1e-9)


def test_ece_is_one_for_a_confidently_wrong_class():
    probs = np.zeros((50, NUM_CLASSES))
    probs[:, 0] = 1.0
    labels = np.ones(50, dtype=np.int64)

    assert per_class_ece(probs, labels)[0] == pytest.approx(1.0)


def test_ece_is_zero_for_a_confidently_right_class():
    probs = np.zeros((50, NUM_CLASSES))
    probs[:, 0] = 1.0
    labels = np.zeros(50, dtype=np.int64)

    assert per_class_ece(probs, labels)[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Run plumbing
# --------------------------------------------------------------------------- #


def test_smoke_subset_is_the_first_n_by_track_id():
    """The subset has to be a pure function of the corpus, not of dict order --
    the determinism proof re-runs it in a fresh process."""
    ids = ["zzz", "aaa", "mmm", "bbb"]
    track_ids = {"zzz": "0001.zzz", "aaa": "0009.aaa", "mmm": "0003.mmm", "bbb": "0002.bbb"}

    assert select_smoke_subset(ids, track_ids, 3) == ["zzz", "bbb", "mmm"]
    assert select_smoke_subset(ids, track_ids, 0) == sorted(ids, key=lambda i: track_ids[i])


def test_smoke_subset_refuses_an_unknown_track():
    with pytest.raises(KeyError):
        select_smoke_subset(["aaa"], {}, 1)


def test_target_stats_counts_only_supervised_positions():
    labels = np.array([0, 1, 1, IGNORE_INDEX], dtype=np.int64)
    mask = np.array([True, True, True, False])
    boundary = np.array([0.0, 1.0, 0.5, 9.0], dtype=np.float32)
    boundary_mask = np.array([True, True, True, False])
    targets = TrackTargets(np.zeros(8, np.int16), np.zeros(8, bool),
                           boundary, boundary_mask, labels, mask)

    stats = accumulate_target_stats([targets, targets])

    assert stats.class_counts.tolist() == [2, 4, 0, 0, 0]
    assert stats.boundary_positive == pytest.approx(3.0)
    assert stats.boundary_valid == 6


@pytest.mark.parametrize("map_location", ["cpu", "cuda"])
def test_rng_survives_a_checkpoint_round_trip(tmp_path, map_location):
    """Resume is only exact if the RNG rejoins where it left off -- and the
    checkpoint is loaded with ``map_location=<device>``, which drags the RNG
    ByteTensors onto that device where the setters refuse them outright."""
    if map_location == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA on this machine")

    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)
    state = capture_rng()
    expected = (torch.rand(3), random.random(), np.random.rand())

    path = tmp_path / "ckpt.pt"
    torch.save({"rng": state}, path)
    restore_rng(torch.load(path, map_location=map_location, weights_only=False)["rng"])

    assert torch.equal(torch.rand(3), expected[0])
    assert random.random() == expected[1]
    assert np.random.rand() == expected[2]


def test_weight_hash_is_stable_and_sensitive():
    first = SectionCRNN(n_mels=8, conv_channels=(4, 4, 4), rnn_hidden=8,
                        conv1d_channels=8).state_dict()
    same = {key: value.clone() for key, value in first.items()}
    changed = {key: value.clone() for key, value in first.items()}
    changed["label_head.bias"] = changed["label_head.bias"] + 1e-7

    assert weight_hash(first) == weight_hash(same)
    assert weight_hash(first) != weight_hash(changed)
    assert json.dumps({"hash": weight_hash(first)})  # plain hex, JSON-safe
