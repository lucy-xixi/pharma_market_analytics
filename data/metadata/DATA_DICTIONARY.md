# Pharmaceutical sales data dictionary

The checked-in panel is `data/raw/pharma_sales_monthly_synthetic.csv`.

Every generated row has `data_type=synthetic_demo` and `data_source=synthetic_demo`. The product codes (`PX-CARD-01` and the other `PX-*` codes) and region labels (`East China`, `North China`, `South China`, `Southwest China`) are simulator labels. Differences between them are planted by `scripts/generate_pharma_data.py` and are not measurements of a real pharmaceutical market.

| Column | Meaning | Rule |
| --- | --- | --- |
| month | Month start, `YYYY-MM-01` | Unique together with product and region |
| product | Fictional product code | Not a marketed SKU |
| region | Fictional region label | Not a measured market |
| sales_amount | Simulated sales value | 模拟金额，未指定真实币种。Non-negative |
| sales_volume | Simulated quantity | 模拟数量，非真实销售包装单位或处方量。Non-negative |
| data_source | Providing process | `synthetic_demo` for the generator |
| data_type | Provenance | `synthetic_demo` or, for a future authorized feed, `real` |

The generator formula and the worked example are in `data/metadata/PHARMA_DATA_NOTES.md`. The machine-readable field list is `data/metadata/pharma_data_dictionary.csv`.
