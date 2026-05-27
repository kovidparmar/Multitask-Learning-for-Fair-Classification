import numpy as np


def to_binary(y):
    y = np.asarray(y)
    if set(np.unique(y)) <= {0, 1}:
        return y
    return (y > 0).astype(int)


def accuracy(y_true, y_pred):
    y_true = to_binary(y_true)
    y_pred = to_binary(y_pred)
    return (y_true == y_pred).mean()


def group_accuracy(y_true, y_pred, s):
    y_true = to_binary(y_true)
    y_pred = to_binary(y_pred)
    s = np.asarray(s)
    acc = {}
    for g in np.unique(s):
        mask = s == g
        acc[int(g)] = (y_true[mask] == y_pred[mask]).mean()
    return acc


def equal_opportunity_diff(y_true, y_pred, s):
    """Difference in True Positive Rates across sensitive groups."""
    y_true = to_binary(y_true)
    y_pred = to_binary(y_pred)
    s = np.asarray(s)

    tprs = []
    for g in np.unique(s):
        mask = (s == g) & (y_true == 1)
        if mask.sum() == 0:
            tprs.append(0.0)
        else:
            tprs.append((y_pred[mask] == 1).mean())
    return float(np.max(tprs) - np.min(tprs))


def equal_opportunity_diff_neg(y_true, y_pred, s):
    """Difference in True Negative Rates across sensitive groups (EOp-)."""
    y_true = to_binary(y_true)
    y_pred = to_binary(y_pred)
    s = np.asarray(s)

    tnrs = []
    for g in np.unique(s):
        mask = (s == g) & (y_true == 0)
        if mask.sum() == 0:
            tnrs.append(0.0)
        else:
            tnrs.append((y_pred[mask] == 0).mean())
    return float(np.max(tnrs) - np.min(tnrs))


def equalized_odds_diff(y_true, y_pred, s):
    """DEOd: average of EOp+ and EOp- differences across groups."""
    eop_pos = equal_opportunity_diff(y_true, y_pred, s)
    eop_neg = equal_opportunity_diff_neg(y_true, y_pred, s)
    return 0.5 * (eop_pos + eop_neg)
