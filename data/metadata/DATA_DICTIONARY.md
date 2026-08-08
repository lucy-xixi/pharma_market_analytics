# Data Dictionary & Metadata Documentation

Enterprise metadata catalog for the **Digital Industry Analytics Platform with AI-assisted Insights**.

Canonical machine-readable source: [`data_dictionary.csv`](data_dictionary.csv)

---

## Catalog overview

| Dataset | Stage | Grain | Description |
|---------|-------|-------|-------------|
| `digital_industry_indicators` | raw | year × industry × region | Primary multi-source panel of digital economy, investment, growth, employment, and R&D indicators |
| `technology_investment_trends` | raw | year | Thematic technology investment series (AI, cloud, IoT) |
| `digital_employment_macro` | raw | year | Macro digital labor-market indicators |
| `industry_panel_clean` | processed | year × industry × region | Governed analytical store after quality validation, cleaning, and indicator enrichment |

---

## Field definitions

Each row in `data_dictionary.csv` follows this schema:

| Attribute | Meaning |
|-----------|---------|
| `dataset_name` | Logical dataset / table name |
| `column_name` | Field name in the dataset |
| `description` | Technical description of the column contents |
| `data_type` | Logical type (`integer`, `float`, `string`) |
| `business_meaning` | Why the field matters for analysis and decisions |
| `unit` | Measurement unit or scale (when applicable) |
| `nullable` | Whether nulls are expected at that pipeline stage |
| `source_stage` | `raw` intake or `processed` analytical store |

---

## Primary panel (`digital_industry_indicators`)

| Column | Type | Business meaning |
|--------|------|------------------|
| `year` | integer | Time dimension for trend analysis and forecasting |
| `industry` | string | Business segment for peer benchmarking and ranking |
| `region` | string | Spatial dimension for regional comparison |
| `digital_index` | float | Proxy for digitalization level (0–100) |
| `technology_investment` | float | Digital transformation spending (USD billions) |
| `revenue_growth` | float | Primary growth outcome (%) |
| `employment` | float | Labor scale (thousands of persons) |
| `rd_ratio` | float | Innovation intensity (% of revenue) |

**Business keys:** `(year, industry, region)` — must be unique after preparation.

---

## Supplementary sources

### `technology_investment_trends`

| Column | Type | Business meaning |
|--------|------|------------------|
| `year` | integer | Time key |
| `ai_investment_bn` | float | AI capability investment (USD bn) |
| `cloud_investment_bn` | float | Cloud infrastructure / services investment (USD bn) |
| `iot_investment_bn` | float | IoT deployment investment (USD bn) |

### `digital_employment_macro`

| Column | Type | Business meaning |
|--------|------|------------------|
| `year` | integer | Time key |
| `digital_jobs_million` | float | Digital-economy job stock (millions) |
| `skills_gap_index` | float | Workforce readiness risk (higher = larger gap) |
| `remote_work_share` | float | Share of digital roles performed remotely (%) |

---

## Analytical store (`industry_panel_clean`)

Inherits cleaned primary-panel fields and adds engineered indicators:

| Column | Type | Business meaning |
|--------|------|------------------|
| `employment_growth` | float | YoY labor expansion (%); DDI component |
| `*_norm` | float | Min-max normalized components (0–1) for index construction |
| `digital_development_index` | float | Weighted composite digital development KPI (0–100) |
| `*_yoy` | float | Short-term momentum for digital index, investment, and growth |

### Digital Development Index inputs

| Component column | Default weight |
|------------------|----------------|
| `technology_investment` | 0.30 |
| `digital_index` | 0.25 |
| `rd_ratio` | 0.25 |
| `employment_growth` | 0.20 |

---

## Usage

```python
from src.data_dictionary import load_data_dictionary, get_dataset_dictionary

dd = load_data_dictionary()
primary = get_dataset_dictionary("digital_industry_indicators")
print(primary[["column_name", "data_type", "business_meaning"]])
```

Pipeline and dashboard code should treat this catalog as the reference for column semantics, quality rules, and stakeholder documentation.
