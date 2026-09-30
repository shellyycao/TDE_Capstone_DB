CREATE SCHEMA IF NOT EXISTS analytics;

CREATE TABLE IF NOT EXISTS analytics.charge_mapping (
  charge_type        text NOT NULL DEFAULT '',
  charge_description text NOT NULL DEFAULT '',
  major_category     text NOT NULL,
  subcategory        text,
  method             text,        -- 'rule', 'tfidf_fallback', 'none'
  similarity         numeric,     -- TF-IDF score (NULL for rule matches)
  row_count          bigint,
  total_value        numeric,
  updated_at         timestamptz DEFAULT now(),
  PRIMARY KEY (charge_type, charge_description)
);

CREATE OR REPLACE VIEW analytics.charge_categorized AS
SELECT c.*,
       coalesce(m.major_category, 'Unmapped') AS major_category,
       coalesce(m.subcategory,    'Unmapped') AS subcategory,
       m.method
FROM staging.charge c
LEFT JOIN analytics.charge_mapping m
  ON  m.charge_type        = coalesce(c.charge_type, '')
  AND m.charge_description = coalesce(c.charge_description, '');

CREATE OR REPLACE VIEW analytics.charge_review AS
SELECT coalesce(c.charge_type, '')        AS charge_type,
       coalesce(c.charge_description, '') AS charge_description,
       coalesce(m.major_category, 'Unmapped') AS major_category,
       m.method,
       count(*)                      AS n_lines,
       round(sum(c.charge_value), 2) AS total_value
FROM staging.charge c
LEFT JOIN analytics.charge_mapping m
  ON  m.charge_type        = coalesce(c.charge_type, '')
  AND m.charge_description = coalesce(c.charge_description, '')
WHERE m.charge_type IS NULL OR m.method = 'none'
GROUP BY 1, 2, 3, 4
ORDER BY abs(sum(c.charge_value)) DESC;
