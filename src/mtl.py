from dataclasses import dataclass
import inspect

import cvxpy as cp
import numpy as np
from sklearn.model_selection import StratifiedKFold

from .metrics import accuracy, equalized_odds_diff


@dataclass
class MTLModel:
    w0: np.ndarray
    v: dict

    def predict(self, X, s):
        Xb = add_bias(X)
        scores = np.zeros(X.shape[0])
        for g in np.unique(s):
            mask = s == g
            if mask.any():
                w = self.w0 + self.v[int(g)]
                scores[mask] = Xb[mask] @ w
        return (scores >= 0).astype(int)

    def scores(self, X, s):
        Xb = add_bias(X)
        scores = np.zeros(X.shape[0])
        for g in np.unique(s):
            mask = s == g
            if mask.any():
                w = self.w0 + self.v[int(g)]
                scores[mask] = Xb[mask] @ w
        return scores

    def scores_soft(self, X, s_prob):
        """Compute scores using soft group assignment probabilities."""
        Xb = add_bias(X)
        s_prob = np.asarray(s_prob)
        groups = sorted(self.v.keys())

        if s_prob.ndim == 1:
            if len(groups) != 2:
                raise ValueError("1D probabilities require exactly two groups.")
            g0, g1 = groups
            w0 = self.w0 + self.v[int(g0)]
            w1 = self.w0 + self.v[int(g1)]
            return (1 - s_prob) * (Xb @ w0) + s_prob * (Xb @ w1)

        if s_prob.shape[1] != len(groups):
            raise ValueError("Probability columns must match group count.")

        scores = np.zeros(X.shape[0])
        for idx, g in enumerate(groups):
            w = self.w0 + self.v[int(g)]
            scores += s_prob[:, idx] * (Xb @ w)
        return scores

    def predict_soft(self, X, s_prob):
        scores = self.scores_soft(X, s_prob)
        return (scores >= 0).astype(int)


def add_bias(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])


def train_mtl(
    X,
    y,
    s,
    rho=0.1,
    lambda_=0.5,
    theta=0.5,
    fairness_constraints=None,
    solver="CLARABEL",
):
    """Train MTL with shared + group-specific weights using hinge loss."""
    Xb = add_bias(X)
    y_signed = np.where(y == 1, 1, -1)

    d = Xb.shape[1]
    groups = np.unique(s)

    w0 = cp.Variable(d)
    v = {int(g): cp.Variable(d) for g in groups}

    # Shared loss
    shared_scores = Xb @ w0
    shared_loss = cp.sum(cp.pos(1 - cp.multiply(y_signed, shared_scores))) / Xb.shape[0]

    # Group losses
    group_losses = []
    for g in groups:
        mask = s == g
        Xg = Xb[mask]
        yg = y_signed[mask]
        w = w0 + v[int(g)]
        loss_g = cp.sum(cp.pos(1 - cp.multiply(yg, Xg @ w))) / Xg.shape[0]
        group_losses.append(loss_g)

    reg_shared = cp.sum_squares(w0)
    reg_groups = cp.sum([cp.sum_squares(w0 + v[int(g)]) for g in groups]) / len(groups)

    objective = theta * shared_loss + (1 - theta) * cp.sum(group_losses) / len(groups)
    objective += rho * (lambda_ * reg_shared + (1 - lambda_) * reg_groups)

    constraints = []
    if fairness_constraints:
        if callable(fairness_constraints):
            params = inspect.signature(fairness_constraints).parameters
            if len(params) == 1:
                constraints.extend(fairness_constraints(w0))
            else:
                constraints.extend(fairness_constraints(w0, v))
        else:
            constraints.extend(fairness_constraints)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=solver, verbose=False)

    w0_val = w0.value
    v_val = {int(g): v[int(g)].value for g in groups}
    return MTLModel(w0=w0_val, v=v_val)


def two_stage_cv_mtl(
    X,
    y,
    s,
    rho_grid,
    lambda_grid,
    theta_grid,
    fairness_builder_factory=None,
    acc_ratio=0.97,
    cv=3,
    random_state=0,
):
    """Two-stage CV: pick params by accuracy, then fairness (DEOd)."""
    records = []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    for rho in rho_grid:
        for lambda_ in lambda_grid:
            for theta in theta_grid:
                acc_scores = []
                deod_scores = []
                for train_idx, val_idx in skf.split(X, y):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    s_train, s_val = s[train_idx], s[val_idx]

                    builder = None
                    if fairness_builder_factory:
                        builder = fairness_builder_factory(X_train, y_train, s_train)

                    model = train_mtl(
                        X_train,
                        y_train,
                        s_train,
                        rho=rho,
                        lambda_=lambda_,
                        theta=theta,
                        fairness_constraints=builder,
                    )
                    pred = model.predict(X_val, s_val)
                    acc_scores.append(accuracy(y_val, pred))
                    deod_scores.append(equalized_odds_diff(y_val, pred, s_val))

                records.append(
                    {
                        "rho": rho,
                        "lambda": lambda_,
                        "theta": theta,
                        "accuracy": float(np.mean(acc_scores)),
                        "deod": float(np.mean(deod_scores)),
                    }
                )

    best_acc = max(r["accuracy"] for r in records)
    candidates = [r for r in records if r["accuracy"] >= acc_ratio * best_acc]
    best = sorted(candidates, key=lambda r: (r["deod"], -r["accuracy"]))[0]
    return records, best
