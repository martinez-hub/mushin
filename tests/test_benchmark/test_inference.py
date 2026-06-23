def test_evaluate_streams_and_matches_one_shot():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark._inference import evaluate
    from mushin.benchmark._metrics import classification_battery, compute_battery
    from mushin.benchmark._predict import default_classification_predict_fn

    g = torch.Generator().manual_seed(0)
    x = torch.randn(20, 4, generator=g)
    y = torch.randint(0, 3, (20,), generator=g)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    model = torch.nn.Linear(4, 3)

    battery = classification_battery(3)
    streamed = evaluate(
        model,
        loader,
        battery,
        default_classification_predict_fn,
        prob_metrics=frozenset({"auroc", "ece"}),
    )
    with torch.no_grad():
        preds, probs = default_classification_predict_fn(model, x)
    one_shot = compute_battery(
        classification_battery(3), preds, y, frozenset({"auroc", "ece"}), probs=probs
    )
    assert streamed.keys() == one_shot.keys()
    for k in streamed:
        assert abs(streamed[k] - one_shot[k]) < 1e-5
