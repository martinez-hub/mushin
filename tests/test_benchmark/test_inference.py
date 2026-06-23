def test_evaluate_with_dict_output_model_and_custom_predict_fn():
    # Real models (e.g. torchvision segmentation) return a dict {"out": logits},
    # not a tensor, so the default predict_fn can't be used. A custom predict_fn
    # extracting ["out"] must flow through evaluate end-to-end. This locks in the
    # segmentation-dogfood friction.
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark._inference import evaluate
    from mushin.benchmark._metrics import classification_battery

    g = torch.Generator().manual_seed(0)
    x = torch.randn(20, 4, generator=g)
    y = torch.randint(0, 3, (20,), generator=g)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    mapping = {tuple(xi.tolist()): int(yi) for xi, yi in zip(x, y)}

    class DictModel(torch.nn.Module):
        def forward(self, xb):
            idx = torch.tensor([mapping[tuple(r.tolist())] for r in xb])
            logits = torch.nn.functional.one_hot(idx, 3).float() * 10.0
            return {"out": logits}  # dict output, like torchvision seg models

    def predict(model, xb):
        logits = model(xb)["out"]
        probs = torch.softmax(logits, dim=-1)
        return probs.argmax(dim=-1), probs

    out = evaluate(
        DictModel(),
        loader,
        classification_battery(3),
        predict,
        prob_metrics=frozenset({"auroc", "ece"}),
    )
    assert out["accuracy"] == 1.0


def test_evaluate_segmentation_ignores_void_across_batches():
    # ignore_index must exclude void pixels through the *streaming* evaluate path
    # (the existing test only covers the one-shot compute_battery). Model is
    # correct on every non-void pixel; its arbitrary void prediction must not
    # count because the void label is excluded.
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark._inference import evaluate
    from mushin.benchmark._metrics import segmentation_battery
    from mushin.benchmark._predict import default_segmentation_predict_fn

    g = torch.Generator().manual_seed(0)
    N, C, H, W = 8, 3, 6, 6
    x = torch.randn(N, 1, H, W, generator=g)
    true = torch.randint(0, C, (N, H, W), generator=g)  # the real classes
    target = true.clone()
    target[:, 0, 0] = 255  # one void pixel per image
    loader = DataLoader(TensorDataset(x, target), batch_size=4)  # 2 batches
    mapping = {tuple(xi.flatten().tolist()): t for xi, t in zip(x, true)}

    class PerfectSeg(torch.nn.Module):
        def forward(self, xb):
            outs = []
            for xi in xb:
                t = mapping[tuple(xi.flatten().tolist())]  # predict the true class
                outs.append(
                    torch.nn.functional.one_hot(t, C).permute(2, 0, 1).float() * 10
                )
            return torch.stack(outs)

    res = evaluate(
        PerfectSeg(),
        loader,
        segmentation_battery(C, ignore_index=255),
        default_segmentation_predict_fn,
        prob_metrics=frozenset(),
    )
    assert res["pixel_acc"] == 1.0
    assert res["miou"] == 1.0


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


def test_evaluate_explicit_device_and_resets_between_calls():
    # evaluate(device=...) must run on the given device, and reusing the same
    # battery dict across calls must NOT leak metric state (evaluate resets each
    # metric before the batch loop).
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mushin.benchmark._inference import evaluate
    from mushin.benchmark._metrics import classification_battery
    from mushin.benchmark._predict import default_classification_predict_fn

    def loader(seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(16, 4, generator=g)
        y = torch.randint(0, 3, (16,), generator=g)
        return DataLoader(TensorDataset(x, y), batch_size=8)

    model = torch.nn.Linear(4, 3)
    cpu = torch.device("cpu")
    pm = frozenset({"auroc", "ece"})

    battery = classification_battery(3)
    evaluate(
        model,
        loader(1),
        battery,
        default_classification_predict_fn,
        prob_metrics=pm,
        device=cpu,
    )  # first call dirties the metric state
    reused = evaluate(
        model,
        loader(2),
        battery,
        default_classification_predict_fn,
        prob_metrics=pm,
        device=cpu,
    )  # same battery, different data
    fresh = evaluate(
        model,
        loader(2),
        classification_battery(3),
        default_classification_predict_fn,
        prob_metrics=pm,
        device=cpu,
    )

    assert reused.keys() == fresh.keys()
    for k in reused:
        assert abs(reused[k] - fresh[k]) < 1e-6  # no carryover from the first call
