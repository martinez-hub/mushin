import torch

from mushin.benchmark._predict import default_classification_predict_fn


def test_predict_returns_preds_and_probs():
    model = torch.nn.Linear(4, 3)
    x = torch.randn(5, 4)
    preds, probs = default_classification_predict_fn(model, x)

    assert preds.shape == (5,)
    assert probs.shape == (5, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.equal(preds, probs.argmax(dim=-1))


def test_segmentation_predict_returns_pixel_preds_and_probs():
    import torch

    from mushin.benchmark._predict import default_segmentation_predict_fn

    class Seg(torch.nn.Module):
        def forward(self, x):  # x: (N, 1, H, W) -> logits (N, C, H, W)
            return torch.randn(x.shape[0], 3, x.shape[2], x.shape[3])

    x = torch.randn(2, 1, 8, 8)
    preds, probs = default_segmentation_predict_fn(Seg(), x)
    assert preds.shape == (2, 8, 8)
    assert probs.shape == (2, 3, 8, 8)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2, 8, 8), atol=1e-5)
    assert torch.equal(preds, probs.argmax(dim=1))
