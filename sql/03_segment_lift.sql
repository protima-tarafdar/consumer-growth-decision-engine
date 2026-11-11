-- Segment-level lift, share of redemptions that would have happened anyway, and net value.
-- Offer cost = $5 per treated conversion. Revenue-based, not margin. Portable across MySQL and SQLite.
WITH arm AS (
    SELECT
        segment,
        SUM(CASE WHEN treatment_group = 'Control'   THEN 1 ELSE 0 END)                                AS n_control,
        SUM(CASE WHEN treatment_group = 'Treatment' THEN 1 ELSE 0 END)                                AS n_treatment,
        SUM(CASE WHEN treatment_group = 'Control'   THEN converted ELSE 0 END)                        AS conv_control,
        SUM(CASE WHEN treatment_group = 'Treatment' THEN converted ELSE 0 END)                        AS conv_treatment,
        SUM(CASE WHEN treatment_group = 'Control'   THEN post_intervention_value ELSE 0 END)          AS value_control,
        SUM(CASE WHEN treatment_group = 'Treatment' THEN post_intervention_value ELSE 0 END)          AS value_treatment
    FROM experiment
    GROUP BY segment
)
SELECT
    segment,
    n_control + n_treatment                                                          AS customers,
    ROUND(100.0 * conv_control   / n_control,   2)                                   AS control_rate_pct,
    ROUND(100.0 * conv_treatment / n_treatment, 2)                                   AS treatment_rate_pct,
    ROUND(100.0 * (1.0 * conv_treatment / n_treatment - 1.0 * conv_control / n_control), 2) AS lift_pp,
    ROUND(100.0 * (1.0 * conv_control / n_control) / (1.0 * conv_treatment / n_treatment), 1) AS wasted_share_pct,
    ROUND((value_treatment - n_treatment * value_control / n_control) - 5.0 * conv_treatment, 2) AS net_value
FROM arm
ORDER BY lift_pp DESC;
