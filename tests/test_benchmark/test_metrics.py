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
