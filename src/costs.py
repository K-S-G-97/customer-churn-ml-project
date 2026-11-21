"""Business cost assumptions and cost calculation."""
import numpy as np

OFFER_COST = 20     # cost of one retention offer
CHURN_COST = 300    # revenue lost when a customer churns
SAVE_RATE = 0.5     # share of targeted churners the offer actually keeps


def cost_breakdown(y_true, y_proba, threshold,
                   offer_cost=OFFER_COST, churn_cost=CHURN_COST, save_rate=SAVE_RATE):
    """Count outcomes and total cost if we target everyone with probability >= threshold."""
    y_true = np.asarray(y_true)
    target = np.asarray(y_proba) >= threshold

    tp = int((target & (y_true == 1)).sum())
    fp = int((target & (y_true == 0)).sum())
    fn = int((~target & (y_true == 1)).sum())
    tn = int((~target & (y_true == 0)).sum())

    total_offer = (tp + fp) * offer_cost
    total_churn = (fn + tp * (1 - save_rate)) * churn_cost

    return {
        "threshold": threshold, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "offer_cost": total_offer, "churn_cost": total_churn,
        "total_cost": total_offer + total_churn,
    }