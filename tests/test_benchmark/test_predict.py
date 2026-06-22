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
