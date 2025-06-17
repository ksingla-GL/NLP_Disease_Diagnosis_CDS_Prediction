-- This query runs against physionet‑data.mimiciii_clinical in BigQuery
WITH
-- 1) Pull admissions (to get admittime/dischtime and DRG)
  adm AS (
    SELECT
      subject_id,
      hadm_id,
      admittime,
      dischtime,
      drg_type,
      drg_code
    FROM
      `physionet-data.mimiciii_clinical.admissions`
  ),

-- 2) Pull ICD diagnoses and labels
  dx AS (
    SELECT
      di.subject_id,
      di.hadm_id,
      dd.long_title AS diag_label
    FROM
      `physionet-data.mimiciii_clinical.diagnoses_icd` di
    JOIN
      `physionet-data.mimiciii_clinical.d_icd_diagnoses` dd
    USING (icd9_code)
  ),

-- 3) Pull DRG description (the “HCFA:” part)
  drg_labels AS (
    SELECT
      drg_code,
      description AS drg_label
    FROM
      `physionet-data.mimiciii_clinical.drgcodes`
  ),

 -- 4) Assemble per‑admission lists
  per_adm AS (
    SELECT
      a.subject_id,
      a.hadm_id,
      FORMAT_TIMESTAMP("%F %H:%M:%S", a.admittime)  AS admit_ts,
      FORMAT_TIMESTAMP("%F %H:%M:%S", a.dischtime)  AS discharge_ts,
      -- comma‑separated diagnosis labels
      STRING_AGG(DISTINCT dx.diag_label, "," ORDER BY dx.diag_label) AS diag_list,
      -- comma‑separated DRG labels prefixed “HCFA:”
      CONCAT(
        "HCFA:",
        STRING_AGG(DISTINCT dr.drg_label, "," ORDER BY dr.drg_label)
      ) AS drg_list
    FROM
      adm a
    LEFT JOIN
      dx
    USING (subject_id, hadm_id)
    LEFT JOIN
      drg_labels dr
    ON
      a.drg_code = dr.drg_code
    GROUP BY
      subject_id, hadm_id, admit_ts, discharge_ts
  )

SELECT
  -- now put it all together with semicolons
  CONCAT(
    CAST(subject_id  AS STRING),  ";",
    CAST(hadm_id     AS STRING),  ";",
    admit_ts,                     ";",
    discharge_ts,                 ";",
    diag_list,                    ";",
    drg_list
  ) AS line
FROM
  per_adm
ORDER BY
  subject_id
;
-- This query runs against physionet‑data.mimiciii_clinical in BigQuery
WITH
-- 1) Pull admissions (to get admittime/dischtime and DRG)
  adm AS (
    SELECT
      subject_id,
      hadm_id,
      admittime,
      dischtime,
      drg_type,
      drg_code
    FROM
      `physionet-data.mimiciii_clinical.admissions`
  ),

-- 2) Pull ICD diagnoses and labels
  dx AS (
    SELECT
      di.subject_id,
      di.hadm_id,
      dd.long_title AS diag_label
    FROM
      `physionet-data.mimiciii_clinical.diagnoses_icd` di
    JOIN
      `physionet-data.mimiciii_clinical.d_icd_diagnoses` dd
    USING (icd9_code)
  ),

-- 3) Pull DRG description (the “HCFA:” part)
  drg_labels AS (
    SELECT
      drg_code,
      description AS drg_label
    FROM
      `physionet-data.mimiciii_clinical.drgcodes`
  ),

 -- 4) Assemble per‑admission lists
  per_adm AS (
    SELECT
      a.subject_id,
      a.hadm_id,
      FORMAT_TIMESTAMP("%F %H:%M:%S", a.admittime)  AS admit_ts,
      FORMAT_TIMESTAMP("%F %H:%M:%S", a.dischtime)  AS discharge_ts,
      -- comma‑separated diagnosis labels
      STRING_AGG(DISTINCT dx.diag_label, "," ORDER BY dx.diag_label) AS diag_list,
      -- comma‑separated DRG labels prefixed “HCFA:”
      CONCAT(
        "HCFA:",
        STRING_AGG(DISTINCT dr.drg_label, "," ORDER BY dr.drg_label)
      ) AS drg_list
    FROM
      adm a
    LEFT JOIN
      dx
    USING (subject_id, hadm_id)
    LEFT JOIN
      drg_labels dr
    ON
      a.drg_code = dr.drg_code
    GROUP BY
      subject_id, hadm_id, admit_ts, discharge_ts
  )

SELECT
  -- now put it all together with semicolons
  CONCAT(
    CAST(subject_id  AS STRING),  ";",
    CAST(hadm_id     AS STRING),  ";",
    admit_ts,                     ";",
    discharge_ts,                 ";",
    diag_list,                    ";",
    drg_list
  ) AS line
FROM
  per_adm
ORDER BY
  subject_id
;
