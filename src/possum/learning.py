import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import log_loss, f1_score, balanced_accuracy_score, brier_score_loss


def temperature(scores, labels):
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(labels), 2) or not np.isfinite(scores).all():
        raise ValueError("Expected finite [N,2] scores")
    f = lambda log_t: log_loss(labels, softmax(scores / np.exp(log_t), axis=1), labels=[0, 1])
    result = minimize_scalar(f, bounds=(-3, 3), method="bounded")
    return float(np.exp(result.x))


def metrics(y, p):
    p = np.clip(np.asarray(p), 1e-7, 1-1e-7)
    conf = np.maximum(p, 1-p)
    correct = ((p >= .5) == y)
    ece = 0.
    for low, high in zip(np.linspace(.5, 1, 11)[:-1], np.linspace(.5, 1, 11)[1:]):
        mask = (conf >= low) & (conf < high)
        if mask.any():
            ece += mask.mean()*abs(conf[mask].mean()-correct[mask].mean())
    return dict(macro_f1=float(f1_score(y, p >= .5, average="macro")), balanced_accuracy=float(balanced_accuracy_score(y, p >= .5)), log_loss=float(log_loss(y, p, labels=[0, 1])), brier=float(brier_score_loss(y, p)), ece=float(ece))


class Gate(torch.nn.Module):
    def __init__(self, experts, fixed=False):
        super().__init__()
        self.fixed = fixed
        if fixed:
            self.logits = torch.nn.Parameter(torch.zeros(experts))
        else:
            self.net = torch.nn.Sequential(torch.nn.Linear(384, 32), torch.nn.Tanh(), torch.nn.Linear(32, experts))
    def weights(self, x):
        return torch.softmax(self.logits.expand(len(x), -1) if self.fixed else self.net(x), -1)
    def forward(self, x, probabilities):
        return (self.weights(x)*probabilities).sum(-1).clamp(1e-7, 1-1e-7)


def fit_gate(x, probabilities, y, fixed=False, seed=42, epochs=150):
    torch.manual_seed(seed)
    model = Gate(probabilities.shape[1], fixed)
    xx, pp, yy = [torch.as_tensor(v, dtype=torch.float32) for v in (x, probabilities, y)]
    opt = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.02)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy(model(xx, pp), yy)
        loss.backward()
        opt.step()
    return model.eval()


def predict_gate(model, x, p):
    with torch.inference_mode():
        return model(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(p, dtype=torch.float32)).numpy()


def fit_neural(x, y, seed=42):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(384, 32), torch.nn.Tanh(), torch.nn.Linear(32, 2))
    xx, yy = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.02)
    for _ in range(100):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(xx), yy).backward()
        opt.step()
    return model.eval()


def bootstrap_delta(y, candidate, reference, groups, seed=42, repeats=500):
    """Negative means candidate has lower log loss. Resample whole groups."""
    unique = np.unique(groups)
    if len(unique) < 3:
        return {"interval": None, "reason": "Fewer than three validation groups; no reliable interval"}
    rng = np.random.default_rng(seed)
    losses = lambda p: -(y*np.log(np.clip(p, 1e-7, 1))+(1-y)*np.log(np.clip(1-p, 1e-7, 1)))
    delta = losses(candidate)-losses(reference)
    draws = []
    for _ in range(repeats):
        ids = np.concatenate([np.flatnonzero(groups == g) for g in rng.choice(unique, len(unique), replace=True)])
        draws.append(delta[ids].mean())
    return {"interval": np.quantile(draws, [.025, .975]).tolist(), "mean": float(delta.mean())}
