# Age composition biases on-chain realization measures — replication package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20759419.svg)](https://doi.org/10.5281/zenodo.20759419)

Replication code and data for **"Age composition biases on-chain realization
measures: Evidence from Bitcoin."**

**One-line result.** The pooled on-chain PGR/PLR ratio built from UTXO cost
basis is confounded by coin-age composition. Gain and loss coins have different
age distributions and realization rates fall steeply with age, so the pooled
ratio can lie outside the range of every within-age ratio. Standardizing to a
common age distribution removes the apparent 2022 bear-market effect
(pooled 2.62 → 0.92–0.93; 87% of the pooled gap is age composition) and reveals
a modest asymmetry in 2024 that the pooled ratio (exactly 1.00) conceals. The
pooled statistic describes coin-age composition, not investor behavior.

## Quickstart

```
git clone https://github.com/canyonjang/btc-disposition.git
cd btc-disposition
pip install -r requirements.txt
python reproduce.py                  # baseline five-year ratios + Tables S1–S4
python age_buckets.py                # age-bucket ratios (Table 2) + TOST
python compute_maintable.py          # PGR/PLR/N main table (Table 1)
python hazard_logit.py               # realization logit (Table 4, Figure "hazard")
python age_std_v2.py                 # age-standardized ratio + Kitagawa (Table 3)
python monthly_blockboot_minlag.py   # monthly series (Figure "monthly"), block bootstrap, min-lag
python robust_count_std.py           # count-weighted standardization (Table S7)
```

No external services or credentials are required — all inputs are bundled in
`./data`. A full run of all scripts takes roughly 1–2 hours on a laptop;
`reproduce.py` alone takes a few minutes. Random seeds are fixed, so confidence
intervals reproduce exactly.

## What each script prints

| Script | Regenerates |
| --- | --- |
| `reproduce.py` | Baseline five-year STH PGR/PLR (both price series); Tables S1 (2022 robustness), S2/S3 (age-threshold sensitivity), S4 (churn-filter coverage) |
| `age_buckets.py` | Age-conditional PGR/PLR in four buckets (Table 2); TOST equivalence tests; writes `age_bucket_results.csv` |
| `compute_maintable.py` | Annual PGR, PLR, PGR−PLR, ratio, gain/loss exposure (Table 1); writes `main_results_table.csv` |
| `hazard_logit.py` | Coin-day realization logit — unconditional, age-bucket, continuous log-age, month/day fixed effects (Table 4); writes `hazard_logit_results.csv` |
| `age_std_v2.py` | Age-standardized disposition ratio DR* on the gain, loss, and trimmed-total age distributions; Kitagawa within-age/composition decomposition (Table 3); writes `age_std_results.csv` |
| `monthly_blockboot_minlag.py` | Monthly pooled-vs-standardized series (Figure, monthly); calendar-time block bootstrap (2022); minimum-holding sensitivity; writes `monthly_std_series.csv` |
| `robust_count_std.py` | Count-weighted age standardization, all years (Table S7); writes `age_std_count.csv` |

Headline five-year output from `reproduce.py` (95% cohort-bootstrap CI; seed fixed):

| Year | CMC | Bitstamp |
| --- | --- | --- |
| 2021 | 0.89 [0.71, 1.12] | 0.90 [0.72, 1.13] |
| 2022 | 2.62 [1.95, 3.48] | 2.93 [2.25, 3.89] |
| 2023 | 0.84 [0.66, 1.08] | 0.81 [0.62, 1.05] |
| 2024 | 1.00 [0.80, 1.27] | 1.02 [0.81, 1.26] |
| 2025 | 0.82 [0.64, 1.04] | 0.82 [0.67, 1.03] |

Age-standardized ratios from `age_std_v2.py` (gain-mix target; the pooled 2022
ratio of 2.62 becomes 0.93):

| Year | Pooled | DR* (gain mix) | DR* (trimmed) |
| --- | --- | --- | --- |
| 2021 | 0.89 | 1.14 [0.97, 1.34] | 1.14 |
| 2022 | 2.62 | 0.93 [0.72, 1.19] | 0.92 |
| 2023 | 0.84 | 1.05 [0.86, 1.28] | 1.05 |
| 2024 | 1.00 | 1.20 [1.03, 1.40] | 1.19 |
| 2025 | 0.82 | 0.97 [0.77, 1.19] | 1.05 |

## Repository layout

```
reproduce.py                 baseline pipeline (reads ./data, prints all baseline numbers)
age_buckets.py               age-bucket decomposition + TOST (imported by the scripts below)
compute_maintable.py         PGR/PLR/N main results table
hazard_logit.py              coin-day grouped-binomial realization logit
age_std_v2.py                age standardization + Kitagawa decomposition
monthly_blockboot_minlag.py  monthly series, block bootstrap, min-holding sensitivity
robust_count_std.py          count-weighted standardization robustness
data/
  prices_cmc.xlsx            daily BTC/USD close (CoinMarketCap-style)
  prices_bitstamp.csv        daily BTC/USD close (Bitstamp; read with skiprows=1)
  matrix_<year>.csv.gz       realized (create_d × spend_d) BTC volume: create_d,spend_d,n,btc
  snapshot_<year>.csv.gz     year-start unspent UTXO volume:           create_d,n_alive,btc_alive
  newdaily_<year>.csv.gz     daily new creation:                       create_d,n_new,btc_new
requirements.txt             Python dependencies
LICENSE                      MIT (covers code)
LICENSE-data                 CC BY 4.0 (covers ./data)
CITATION.cff                 citation metadata
```

## Data provenance

All behavioural data are aggregated derivatives of the public BigQuery dataset
`bigquery-public-data.crypto_bitcoin`, aggregated by creation day and spend day
— no individual addresses or transactions are stored. The creation floor is
2018-08-23, the common start of continuous daily price coverage. 
The bundled files cover 2021–2025 and are the sole inputs to every script
in this repository. The BigQuery extraction queries that regenerate the
bundled files from scratch are provided in `sql/`.

## Method in one paragraph

For every spent output we recover its creation day (cost basis = that day's
BTC/USD price) and its spend day directly from the UTXO graph, **without address
clustering** (the unit is the coin, not the investor). On value-weighted
short-term-holder supply (coins younger than 155 days), excluding same-day
spends and ±5%-deadband returns, we compute Odean's (1998) PGR and PLR. Because
gain and loss coins occupy different age distributions and realization rates
fall steeply with age, the pooled ratio is an age-composition average; we
therefore evaluate PGR and PLR on a common age distribution (the
**age-standardized disposition ratio**) and decompose the pooled gain−loss gap
into within-age and composition components. Intervals use a 2000-replication
cohort bootstrap (seed fixed); a calendar-time block bootstrap is reported for
the pivotal year.

## License

- **Code**: MIT — see `LICENSE`.
- **Data** (`./data`): CC BY 4.0 — see `LICENSE-data`.

## Citation

See `CITATION.cff`. Please cite both the paper and this replication package. The
archived version is available at <https://doi.org/10.5281/zenodo.20759419>.
