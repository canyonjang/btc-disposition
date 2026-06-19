# Disposition effect on the Bitcoin blockchain — replication package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20759420.svg)](https://doi.org/10.5281/zenodo.20759420)

Replication code and data for **"Identifying the Disposition Effect on the
Bitcoin Blockchain: Real but Regime-Specific."**

**One-line result.** A clustering-free, proportional PGR/PLR built from UTXO
cost basis on active short-term-holder supply shows that the on-chain
disposition signal is largely a measurement artifact: it is severely confounded
without an active-supply restriction, and once properly measured a robust
disposition effect survives **only in the 2022 bear market** (PGR/PLR ≈ 2.6–2.9),
while the three bull markets and the post-ETF years are statistically
indistinguishable from one.

## Quickstart

```bash
git clone https://github.com/canyonjang/btc-disposition.git
cd btc-disposition-frl
pip install -r requirements.txt
python reproduce.py          # regenerates all estimates and table numbers
python figures/make_figures.py   # regenerates Figure 1 and Figure 2
```

No external services or credentials are required — all inputs are bundled in
`./data`. A full run takes a few minutes on a laptop.

## What `reproduce.py` prints

* **Figure 1** — five-year STH PGR/PLR under the CoinMarketCap and Bitstamp price series
* **Table S1** — 2022 robustness (deadband, weighting, holding-period, crisis-window, strict specs)
* **Tables S2/S3** — STH age-threshold sensitivity (90–365 days), each price series
* **Table S4** — share of STH spend volume removed by the churn filters

Headline five-year output (95% cohort-bootstrap CI; seed fixed):

| Year (regime)      | CMC                | Bitstamp           |
|--------------------|--------------------|--------------------|
| 2021 (bull)        | 0.89 [0.71, 1.12]  | 0.90 [0.72, 1.13]  |
| 2022 (bear)        | 2.62 [1.95, 3.48]  | 2.93 [2.25, 3.89]  |
| 2023 (bull)        | 0.84 [0.66, 1.08]  | 0.81 [0.62, 1.05]  |
| 2024 (bull, ETF)   | 1.00 [0.80, 1.27]  | 1.02 [0.81, 1.26]  |
| 2025 (mixed, ETF)  | 0.82 [0.64, 1.04]  | 0.82 [0.67, 1.03]  |

## Repository layout

```
reproduce.py            end-to-end pipeline (reads ./data, prints all numbers)
figures/make_figures.py generates Figure 1 (forest) and Figure 2 (ladder)
figures/*.pdf           the figures as submitted (vector)
data/
  prices_cmc.xlsx       daily BTC/USD close (CoinMarketCap-style)
  prices_bitstamp.csv   daily BTC/USD close (Bitstamp; read with skiprows=1)
  matrix_<year>.csv.gz    realized (create_d × spend_d) BTC volume: create_d,spend_d,n,btc
  snapshot_<year>.csv.gz  year-start unspent UTXO volume:           create_d,n_alive,btc_alive
  newdaily_<year>.csv.gz  daily new creation:                       create_d,n_new,btc_new
requirements.txt        Python dependencies
LICENSE                 MIT (covers code)
LICENSE-data            CC BY 4.0 (covers ./data)
CITATION.cff            citation metadata
```

## Data provenance

All behavioural data are aggregated derivatives of the public BigQuery dataset
`bigquery-public-data.crypto_bitcoin`, produced by the queries in the paper's
Supplementary Material (Section S4). They are aggregated by creation day and
spend day — no individual addresses or transactions are stored. The creation
floor is 2018-08-23, the common start of continuous daily price coverage.
To regenerate the `./data` files from scratch, run the S4 queries and export the
columns shown above.

## Method in one paragraph

For every spent output we recover its creation day (cost basis = that day's
BTC/USD price) and its spend day, directly from the UTXO graph and **without
address clustering** (the unit is the coin, not the investor). We compute
Odean's (1998) proportion of gains realized (PGR) and proportion of losses
realized (PLR) on value-weighted short-term-holder supply (coins younger than
155 days), exclude same-day spends and ±5%-deadband returns to suppress
change-output churn, and bound the ratio with a 500-replication cohort
bootstrap. A ratio above one signals a coin-level realization asymmetry
consistent with a disposition effect.

## License

* **Code** (`reproduce.py`, `figures/make_figures.py`): MIT — see `LICENSE`.
* **Data** (`./data`): CC BY 4.0 — see `LICENSE-data`.

## Citation

See `CITATION.cff`. Please cite both the paper and this replication package.
The archived version of this package is available at
[https://doi.org/10.5281/zenodo.20759420](https://doi.org/10.5281/zenodo.20759420).
