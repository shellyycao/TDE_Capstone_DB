-- Client review (2026-10-15): review status per label, and line-level exclusion.
-- charge_mapping itself is unchanged. Raw data is untouched: exclusion is a flag computed in
-- the view, so it is visible and reversible.

-- One row per label the client reviewed (the override seed, seeds/charge_mapping_overrides.csv);
-- labels never reviewed have no row. jobs/charge_type.py re-syncs it from the seed on every run.
CREATE TABLE IF NOT EXISTS analytics.charge_mapping_review (
  charge_type        text NOT NULL DEFAULT '',
  charge_description text NOT NULL DEFAULT '',
  review_status      text NOT NULL,   -- confirmed / client_assigned / disputed / excluded
  note               text,
  source             text,
  reviewed_on        date,
  PRIMARY KEY (charge_type, charge_description)
);

-- is_excluded / exclude_reason: lines that are not shipping cost. Currently the factoring rate
-- lookups (client answer, 2026-10-15): carrier "FACTORING RATE CHECK" or client "Factoring Look Up",
-- whatever the label. Charges without a shipment row (or with a NULL name) are never excluded. To lift the exclusion,
-- remove the WHEN branch below (or narrow it) and re-create the view.
CREATE OR REPLACE VIEW analytics.charge_categorized AS
SELECT c.*,
       coalesce(m.major_category, 'Unmapped') AS major_category,
       coalesce(m.subcategory,    'Unmapped') AS subcategory,
       m.method,
       coalesce(upper(trim(s.carrier_name)) = 'FACTORING RATE CHECK'
             OR lower(trim(s.client_name))  = 'factoring look up', false) AS is_excluded,
       CASE WHEN upper(trim(s.carrier_name)) = 'FACTORING RATE CHECK'
              OR lower(trim(s.client_name))  = 'factoring look up'
            THEN 'factoring_lookup' END                               AS exclude_reason,
       r.review_status,
       r.note                                                         AS review_note
FROM staging.charge c
LEFT JOIN analytics.charge_mapping m
  ON  m.charge_type        = coalesce(c.charge_type, '')
  AND m.charge_description = coalesce(c.charge_description, '')
LEFT JOIN analytics.charge_mapping_review r
  ON  r.charge_type        = coalesce(c.charge_type, '')
  AND r.charge_description = coalesce(c.charge_description, '')
LEFT JOIN staging.shipment s ON s.shipment_id = c.shipment_id;

-- Excluded lines are not part of the review queue.
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
LEFT JOIN staging.shipment s ON s.shipment_id = c.shipment_id
WHERE (m.charge_type IS NULL OR m.method = 'none')
  AND NOT coalesce(upper(trim(s.carrier_name)) = 'FACTORING RATE CHECK' OR lower(trim(s.client_name)) = 'factoring look up', false)
GROUP BY 1, 2, 3, 4
ORDER BY abs(sum(c.charge_value)) DESC;
