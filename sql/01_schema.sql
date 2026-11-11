-- Experiment table: one row per customer. Load data/consumer_growth_experiment.csv into it.
-- MySQL:  LOAD DATA LOCAL INFILE 'data/consumer_growth_experiment.csv' INTO TABLE experiment
--         FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' IGNORE 1 LINES;
-- Only the columns used by the queries below are listed; load the CSV with a column list or a staging table.
CREATE TABLE experiment (
    customer_id             INT PRIMARY KEY,
    segment                 VARCHAR(16),      -- New / Active / At Risk / Lapsed
    treatment_group         VARCHAR(16),      -- Control / Treatment
    converted               INT,              -- 1 if purchased within 30 days of assignment
    post_intervention_value DECIMAL(10, 2)    -- revenue in the 30-day window
);
