-- Overall experiment result: conversion by arm. Portable across MySQL and SQLite.
SELECT
    treatment_group,
    COUNT(*)                                   AS customers,
    SUM(converted)                             AS converted,
    ROUND(100.0 * SUM(converted) / COUNT(*), 2) AS conversion_rate_pct,
    ROUND(SUM(post_intervention_value), 2)     AS revenue
FROM experiment
GROUP BY treatment_group
ORDER BY treatment_group;
