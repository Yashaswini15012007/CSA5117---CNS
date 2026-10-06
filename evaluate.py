"""
evaluate.py — F_0.5 evaluation metric for business entity resolution.

Per-entity F_0.5 = (1.25 * P * R) / (0.25 * P + R)
Macro-averaged across all Source 1 entities.

Special cases:
  - True singleton (empty GT) + predicted empty → score 1.0
  - True singleton (empty GT) + predicted non-empty → score 0.0
"""

from typing import Dict, List, Set


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F_beta score."""
    beta2 = beta ** 2
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def entity_f05(gt_ids: Set[str], pred_ids: Set[str]) -> float:
    """
    Compute per-entity F_0.5 score.

    Parameters
    ----------
    gt_ids   : set of true matched IDs for this S1 entity
    pred_ids : set of predicted matched IDs

    Returns
    -------
    F_0.5 score in [0, 1]
    """
    # True singleton: no ground-truth matches
    if len(gt_ids) == 0:
        if len(pred_ids) == 0:
            return 1.0   # Correctly predicted empty
        else:
            return 0.0   # False merge on a true singleton

    # Normal case
    if len(pred_ids) == 0:
        # No predictions made, recall = 0
        return 0.0

    tp = len(gt_ids & pred_ids)
    precision = tp / len(pred_ids)
    recall    = tp / len(gt_ids)
    return f_beta(precision, recall, beta=0.5)


def macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions:  Dict[str, Set[str]],
    s1_ids: List[str],
) -> float:
    """
    Compute macro-averaged F_0.5 over all S1 entities.

    Parameters
    ----------
    ground_truth : dict s1_id → set of matched IDs (from GT)
    predictions  : dict s1_id → set of predicted matched IDs
    s1_ids       : list of ALL S1 entity IDs (must include singletons)

    Returns
    -------
    Macro F_0.5 score
    """
    scores = []
    for s1_id in s1_ids:
        gt   = ground_truth.get(s1_id, set())
        pred = predictions.get(s1_id, set())
        scores.append(entity_f05(gt, pred))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def threshold_sweep(
    s1_ids: List[str],
    candidate_pairs: List,       # list of (s1_id, other_id, proba)
    ground_truth: Dict[str, Set[str]],
    thresholds=None,
) -> tuple:
    """
    Sweep decision threshold and return (best_threshold, best_f05, scores_dict).

    Parameters
    ----------
    s1_ids : all S1 entity IDs in this split
    candidate_pairs : list of (s1_id, other_id, probability)
    ground_truth : dict s1_id → set of matched IDs
    thresholds : iterable of thresholds to try (default 0.05–0.95 step 0.01)

    Returns
    -------
    (best_threshold, best_f05, {threshold: f05})
    """
    import numpy as np
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    scores = {}
    best_t, best_f = 0.5, -1.0

    for t in thresholds:
        preds: Dict[str, set] = {sid: set() for sid in s1_ids}
        for s1_id, o_id, proba in candidate_pairs:
            if proba >= t:
                preds[s1_id].add(o_id)
        f = macro_f05(ground_truth, preds, s1_ids)
        scores[round(float(t), 3)] = f
        if f > best_f:
            best_f = f
            best_t = round(float(t), 3)

    return best_t, best_f, scores
