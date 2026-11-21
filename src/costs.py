"""Business cost assumptions and cost calculation."""
import numpy as np

OFFER_COST = 20     # cost of one retention offer
CHURN_COST = 300    # revenue lost when a customer churns
SAVE_RATE = 0.5     # share of targeted churners the offer actually keeps


def cost_breakdown(y_true, y_proba, threshold):
    """Count outcomes and total cost if we target everyone with probability >= threshold."""
    y_true = np.asarray(y_true)
    target = np.asarray(y_proba) >= threshold

    tp = int((target & (y_true == 1)).sum())
    fp = int((target & (y_true == 0)).sum())
    fn = int((~target & (y_true == 1)).sum())
    tn = int((~target & (y_true == 0)).sum())

    offer_cost = (tp + fp) * OFFER_COST
    churn_cost = (fn + tp * (1 - SAVE_RATE)) * CHURN_COST

    return {
        "threshold": threshold, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "offer_cost": offer_cost, "churn_cost": churn_cost,
        "total_cost": offer_cost + churn_cost,
    }