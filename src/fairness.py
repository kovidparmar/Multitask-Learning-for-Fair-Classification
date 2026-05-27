import cvxpy as cp
import numpy as np

from .mtl import add_bias


def _u_vectors(X, y, s, label_value):
    Xb = add_bias(X)
    y = np.asarray(y)
    s = np.asarray(s)

    groups = np.unique(s)
    u_vec = {}
    for g in groups:
        mask = (s == g) & (y == label_value)
        if mask.sum() == 0:
            u_vec[int(g)] = np.zeros(Xb.shape[1])
        else:
            u_vec[int(g)] = Xb[mask].mean(axis=0)
    return u_vec, groups


def equal_opportunity_constraints_shared(X, y, s, w0, eps=0.0):
    """Approximate EOp+ constraints for the shared model."""
    u_pos, groups = _u_vectors(X, y, s, label_value=1)

    constraints = []
    base = u_pos[int(groups[0])]
    for g in groups[1:]:
        diff = base - u_pos[int(g)]
        constraints.append(cp.abs(w0 @ diff) <= eps)
    return constraints


def build_equal_opportunity_constraints_shared(X, y, s, eps=0.0, label_value=1):
    """Return a builder for EOp constraints tied to a given w0 variable."""
    u_vec, groups = _u_vectors(X, y, s, label_value=label_value)

    def _builder(w0):
        constraints = []
        base = u_vec[int(groups[0])]
        for g in groups[1:]:
            diff = base - u_vec[int(g)]
            constraints.append(cp.abs(w0 @ diff) <= eps)
        return constraints

    return _builder


def build_equalized_odds_constraints_shared(X, y, s, eps=0.0):
    """Return a builder for EOd constraints (EOp+ and EOp-)."""
    builder_pos = build_equal_opportunity_constraints_shared(X, y, s, eps=eps, label_value=1)
    builder_neg = build_equal_opportunity_constraints_shared(X, y, s, eps=eps, label_value=0)

    def _builder(w0):
        return builder_pos(w0) + builder_neg(w0)

    return _builder


def build_equal_opportunity_constraints_group(X, y, s, eps=0.0, label_value=1):
    """Return a builder for group-specific EOp constraints for MTL heads."""
    u_vec, groups = _u_vectors(X, y, s, label_value=label_value)

    def _builder(w0, v):
        constraints = []
        base_group = int(groups[0])
        base_u = u_vec[base_group]
        base_w = w0 + v[base_group]
        for g in groups[1:]:
            group = int(g)
            w_g = w0 + v[group]
            diff = base_w @ base_u - w_g @ u_vec[group]
            constraints.append(cp.abs(diff) <= eps)
        return constraints

    return _builder


def build_equalized_odds_constraints_group(X, y, s, eps=0.0):
    """Return a builder for group-specific EOd constraints (EOp+ and EOp-)."""
    builder_pos = build_equal_opportunity_constraints_group(X, y, s, eps=eps, label_value=1)
    builder_neg = build_equal_opportunity_constraints_group(X, y, s, eps=eps, label_value=0)

    def _builder(w0, v):
        return builder_pos(w0, v) + builder_neg(w0, v)

    return _builder
