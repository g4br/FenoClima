# FenoClima

The code presented here follows a GO HORSE methodology — it is a playground / work-in-progress and does not yet have operational maturity.

## Municipal-level phenology and climate pipeline:
- Time-series collection from Google Earth Engine / BigQuery,
- Phenological metric extraction (TIMESAT adaptation),
- Climate indicators,
- Hybrid neural network to predict productivity (deviation from the technological trend).

## Repository structure
- `big_query-gee.py` — NDVI calculation and extraction at the municipal level + soy mask (MapBiomas) + cloud mask + soil type in GEE (BigQuery).
- `timesat.py` — Detection of phenological metrics from the NDVI time series.
- `clima_timesat.py` — Calculation of climate parameters aligned to the phenological stage, dates, and municipality.
- `FenoClima.py` — Orchestrates the feature flow and trains the hybrid DL network.
- `diagram.png` — Neural network architecture diagram.
- `terminal-print.txt` — Console output at the end of training.
- `resultado-dados-test.csv` — Model output for the test dataset [records not seen by the neural network].
- `DOCUMENTATION.md` — Full technical documentation with methodology Mermaid diagram.

## [ Extra ] GEE Code
code: https://code.earthengine.google.com/c078670b4d00f5614b5bf91921ca15eb
app: https://ee-gabrielluanrodrigues.projects.earthengine.app/view/ndvi-mapbiomas-nuvens-mask
