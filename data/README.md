# Data

All data are **synthetic**, generated in `notebooks/consumer_growth_data_generation.ipynb` (seeded, `np.random.seed(42)`).

| File | Rows | Description |
|---|---:|---|
| `consumer_growth_experiment.csv` | 15,000 | One row per customer: 12 pre-experiment features, random assignment, 30-day outcome |
| `sample_customer_input.csv` | 200 | Feature-only batch for the decision engine (no outcomes) |
| `multi_offer_experiment.csv` | 40,000 | Simulated four-arm coupon test: none, $3, $5 or $8 off a $30 order. Built by `src/simulate_multi_offer.py`; `true_p_*` columns hold the simulator's true order probabilities and are used only to score policies, never as model input |

## Columns in `consumer_growth_experiment.csv`

| Column | Meaning |
|---|---|
| `customer_id` | Unique id |
| `segment` | Lifecycle segment from recency/frequency rules: New, Active, At Risk, Lapsed |
| `acquisition_channel`, `geography` | Categorical customer attributes |
| `customer_tenure_days` | Days since signup |
| `orders_30d`, `orders_previous_30d`, `orders_90d`, `total_orders` | Order counts before the experiment |
| `total_spend`, `average_order_value` | Historical spend (net of discounts) |
| `days_since_last_order`, `frequency_trend` | Recency; orders_30d minus orders_previous_30d |
| `treatment_group` | `Control` (no offer) or `Treatment` ($5 off $30), randomized 50/50 |
| `converted` | 1 if the customer purchased within 30 days |
| `post_intervention_orders`, `post_intervention_value` | Orders and revenue in the 30-day window |
| `predicted_*`, `uplift_*`, `target_recommendation` | Scores from the v1 notebook. **In-sample; do not use for evaluation.** Use `outputs/heldout_scores.csv`. |

## Important

Segment-level treatment effects were assigned inside the generator. This dataset can test whether an analysis recovers a known
structure; it cannot say how real customers behave.
