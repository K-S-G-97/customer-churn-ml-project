# Customer Churn Prediction

## Project goal
Identify customers who are likely to churn so the retention team can
target them with offers, and choose a decision threshold that minimizes
the total business cost of missed churners and wasted offers.

## Key challenges addressed
- **Class imbalance:** only ~27% of customers churn, so accuracy is misleading
- **Data leakage:** making sure the model only uses information available before churn
- **Threshold selection:** choosing a cutoff based on business cost, not 0.5
- **False positives vs false negatives:** quantifying the cost of each error in dollars

## Business assumptions
- Retention offer costs $20 per customer
- A lost customer costs about $300 in revenue

## Dataset
[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn):
7,043 customers, 21 columns. This is a fictional sample dataset created by IBM.

## Status
🚧 In progress