import torch

from mushin.benchmark._metrics import classification_battery, compute_metrics


def test_battery_has_expected_metrics():
    battery = classification_battery(num_classes=3)
    assert set(battery) == {"accuracy", "f1", "precision", "recall", "auroc", "ece"}


def test_perfect_classifier_scores():
    battery = classification_battery(num_classes=3)
    preds = torch.tensor([0, 1, 2, 0, 1, 2])
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    probs = torch.nn.functional.one_hot(preds, num_classes=3).float()

    out = compute_metrics(preds, probs, targets, battery)
    assert out["accuracy"] == 1.0
    assert out["f1"] == 1.0


def test_metrics_do_not_carry_state_across_calls():
    battery = classification_battery(num_classes=3)
    good_preds = torch.tensor([0, 1, 2])
    good_probs = torch.nn.functional.one_hot(good_preds, num_classes=3).float()
    targets = torch.tensor([0, 1, 2])

    first = compute_metrics(good_preds, good_probs, targets, battery)
    second = compute_metrics(good_preds, good_probs, targets, battery)
    assert first["accuracy"] == second["accuracy"] == 1.0


def test_segmentation_battery_perfect_masks():
    from mushin.benchmark._metrics import compute_battery, segmentation_battery

    battery = segmentation_battery(num_classes=3)
    assert set(battery) == {"miou", "dice", "pixel_acc", "precision", "recall"}
    preds = torch.randint(0, 3, (2, 8, 8))
    out = compute_battery(battery, preds, preds, prob_metrics=frozenset())
    assert out["miou"] == 1.0
    assert out["dice"] == 1.0
    assert out["pixel_acc"] == 1.0


def test_segmentation_battery_ignore_index():
    from mushin.benchmark._metrics import compute_battery, segmentation_battery

    battery = segmentation_battery(num_classes=3, ignore_index=255)
    pred = torch.zeros(1, 4, 4, dtype=torch.long)
    tgt = torch.zeros(1, 4, 4, dtype=torch.long)
    tgt[0, 0, 0] = 255  # one void pixel, excluded
    out = compute_battery(battery, pred, tgt, prob_metrics=frozenset())
    assert out["pixel_acc"] == 1.0
