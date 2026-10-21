# Pharmaceutical sales panel — rules and limitations

This note specifies the synthetic demo. It is not a description of a real drug market.

## Contract

| Item | Rule |
|------|------|
| Grain | One row per `month × product × region` |
| `month` | Month-start date (`YYYY-MM-01`), monthly frequency `MS` |
| `product` | Product identifier |
| `region` | Region identifier |
| `sales_amount` | Sales value, non-negative |
| `sales_volume` | Sales quantity, non-negative |
| `data_source` | Providing process. The demo uses `synthetic_demo` |
| `data_type` | `synthetic_demo` or `real`. The demo emits only `synthetic_demo` |
| Business key | `month + product + region` |

Machine-readable fields: [`pharma_data_dictionary.csv`](pharma_data_dictionary.csv).
Numeric simulator settings: [`config/pharma_settings.yaml`](../../config/pharma_settings.yaml).

The shipped demo grid is January 2021 through December 2024 (48 months), five fictional product codes (`PX-CARD-01`, `PX-ENDO-02`, `PX-DIAB-03`, `PX-RESP-04`, `PX-ANTI-05`), and four region labels (`East China`, `North China`, `South China`, `Southwest China`). That is 960 rows. The length is there so later time-series checks have several seasonal cycles. It is not a claim about how long any real brand has been on the market.

## How the demo file is produced

Command, from the repository root:

```bash
python scripts/generate_pharma_data.py
```

The script reads `config/pharma_settings.yaml` only. It writes `data/raw/pharma_sales_monthly_synthetic.csv`. It does not read `digital_industry_indicators.csv`, `technology_investment_trends.csv`, `digital_employment_macro.csv`, or `crude_oil_weekly_synthetic.csv`.

Default seed is `42`. The same seed and the same YAML reproduce the same file.

Random number order, before the file is sorted:

1. Product list order in the YAML.
2. Region list order in the YAML.
3. Month ascending.
4. One `numpy.random.Generator.normal(0, noise_scale)` draw per row (`noise_scale` is `0.03`).

The CSV is then sorted by `month`, `product`, `region`. Sort order is not the random-number order.

For each row, with `t = 0` on the first month:

```text
seasonal = 1 + season_amplitude * cos(2 * pi * (calendar_month - peak_month) / 12)
level    = base_volume * region_multiplier * (1 + monthly_trend * t) * seasonal
volume   = round(max(0, level * (1 + noise)), volume_decimals)
price    = unit_price * (1 + price_drift_per_month * t)
amount   = round(volume * price, amount_decimals)
```

`round` is Python 3 half-to-even rounding. Volume is floored at zero before rounding so a large negative noise draw cannot create a negative quantity. Amount uses the rounded volume. The generator builds every month in the range, so the demo file has no missing months and no duplicate keys.

Worked check with noise fixed at 0 (this is not the random first row of the file):

- `PX-CARD-01`, `East China`, `2021-01-01`, `t = 0`
- seasonal = `1 + 0.06 * cos(0) = 1.06`
- level = `12000 * 1.20 * 1.06 = 15264`
- volume = `15264.00`
- amount = `15264 * 86.0 = 1312704.00`

## What the planted pattern is

The YAML knobs create smooth differences on purpose:

- Each product has its own base volume, unit price, trend, seasonal amplitude, peak month, and price drift.
- Each region has a volume multiplier.
- Noise is independent Gaussian noise on the volume level.

Those patterns exist so later cleaning, growth, and forecast code has something regular to compute. A high seasonal amplitude on `PX-RESP-04`, or a larger East China multiplier, is a simulator setting. It is not evidence of respiratory seasonality or of regional demand.

## Limitations

- No patient, prescription, hospital, pharmacy, tender, or inventory records are included.
- Product codes are fictional. They are not national drug codes or brand names.
- Region names are grouping labels. Multipliers are not population, GDP, or market-share estimates.
- Price is a deterministic drift around a constant. It is not a tender price or an invoice price.
- There is no channel, pack, hospital tier, competitor, stock-out, or returns process. The only non-negativity rule is the mathematical floor at zero.
- The demo does not contain duplicate keys, missing months, or negative sales. Those cases belong in later quality tests, not in this generated file.
- `data_type=real` is reserved for a future authorized source. This generator refuses to write `real`.
- Any average, growth rate, or forecast computed from this file describes the simulator. It must not be presented as a pharmaceutical market finding, a real market share, or a commercial opportunity.
- The contract check does not fill gaps or clip values. Forecasting lives in `src/pharma_forecast.py`. It trains only for evaluation: rolling validation chooses the model, and the final test months are not used for that choice. Missing months are not filled with zero.
- Duplicate-key retention, negative-value exclusion, and gap reporting are demo conventions in `src/pharma_quality.py`. They are not universal rules for real pharmaceutical sales.

## Reproduce the worked example

```python
from src.pharma_schema import simulate_sales_observation

volume, amount = simulate_sales_observation(
    base_volume=12000,
    region_multiplier=1.2,
    t=0,
    month=1,
    monthly_trend=0.004,
    season_amplitude=0.06,
    peak_month=1,
    unit_price=86.0,
    price_drift_per_month=0.001,
    noise=0.0,
)
assert (volume, amount) == (15264.0, 1312704.0)
```
