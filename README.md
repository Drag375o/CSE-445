# CSE-445
Quantifying Temperature-Driven Electricity Demand and Its Potential Climate Impact Using Machine Learning


Temperature and Electricity Demand in Spain (2015–2018)
Estimating how Spanish national electricity demand responds to temperature, and what uniform
warming of +1 °C to +3 °C would do to seasonal load and CO₂ emissions.

Four years of hourly data, 35,028 usable hours after cleaning. Trained on 2015–2017, tested on 2018.

---

## Main result

Warming redistributes demand between seasons rather than changing the annual total.

| Scenario | Summer | Winter | Annual mean | Annual 95% CI |
|---|---|---|---|---|
| +1 °C | +166 MW | −140 MW | +0.2 MW | [−49.8, +47.3] |
| +2 °C | +337 MW | −276 MW | +10.7 MW | [−88.7, +104.3] |
| +3 °C | +512 MW | −404 MW | +32.2 MW | [−116.1, +174.6] |

Summer and winter effects exclude zero at every warming level. Spring and autumn never do.
The annual mean contains zero at every level. Under +3 °C, gross seasonal movement is 1,111 MW
against a net annual change of 32 MW — about 35 times larger.

This matters because electricity infrastructure is sized against peak load, not annual energy.
Reporting only the annual figure produces a null result and hides the finding.

### Other findings

- Demand is minimised at **19.0 °C**. Cooling sensitivity is +104.6 MW/°C above 18 °C;
  heating is +78.7 MW/°C below it.
- Temperature sensitivity depends heavily on time of day, and asymmetrically. Heating runs from
  **3.2 MW/°C overnight to 326.0 MW/°C in the afternoon**; cooling stays between 119 and 206 MW/°C.
  Electric heating follows building occupancy, cooling follows thermostats.
- The cooling response steepens to **+353.5 MW/°C above 30 °C**, roughly triple the 22–26 °C band.
- The **marginal** emission factor is **0.207 kg CO₂/kWh**, 12% below the fleet average of 0.236,
  because roughly 48% of the marginal response comes from reservoir hydro. Two independent
  estimation routes agree to three decimal places.
- Under +3 °C, emissions shift by roughly +233,000 t in summer and −180,000 t in winter. The net
  annual figure inherits the annual confidence interval and is **not** distinguishable from zero.

---

## The methodological point

An accurate forecasting model is not automatically a model of anything.

A random forest given demand lags reached **MAE 355 MW, R² 0.984** — but 90.7% of its feature
importance sat on the demand of the previous hour, and 0.27% on temperature. It cannot answer a
question about temperature, because the demand history it depends on was itself recorded under
the temperatures the counterfactual proposes to change.

So a second model was built with no demand history at all: weather and calendar features only,
with heating and cooling degree-days. It reaches **MAE 2,097 MW, R² 0.635**. That is the model
used for every scenario number above. The accuracy loss is the price of being able to ask the
question at all.

---

## Repository layout

```
FINAL
├── project_final_last_v1.ipynb      # the analysis, 68 code cells
├── Spain_Energy_Project_Report.pdf
├── Spain_Energy_Project_Report.docx

```

---

## Data

Both datasets are loaded from public URLs in Cell 1. Nothing needs downloading by hand.

| Dataset | Shape | Source |
|---|---|---|
| Weather features, 5 cities, hourly | 178,396 × 17 | `vitaliy-sharandin/energy-consumption-weather-hourly-spain` (Hugging Face) |
| Energy dataset, national, hourly | 35,064 × 29 | `vitaliy-sharandin/energy-consumption-hourly-spain` (Hugging Face) |

Cities: Madrid, Barcelona, Valencia, Seville, Bilbao. These are combined into one
population-weighted national series using 2018 municipal populations from the Spanish National
Statistics Institute (INE), weighting Madrid at 46.9% down to Bilbao at 5.0%.

Target variable: `total load actual` (realised national demand, MW).

### Cleaning worth knowing about

The weather file is not one row per hour. Only 32,514 timestamps had the expected five rows;
2,114 had six, and some had up to ten. Investigation showed the duplication came entirely from
the categorical weather labels — the source recorded both "clear" and "few clouds" for the same
hour and wrote the row twice. Across all 2,798 duplicated groups the numerical measurements were
identical in every single one, so collapsing each city-hour to one row is safe. That check is in
Cells 9–14 and it determined the cleaning rule; dropping exact duplicates alone would not have
worked, since only 42 rows were exact duplicates.

---

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
joblib
```

Written and run in Google Colab. Cell 68 imports `google.colab.drive` to save artifacts — if you
run this locally, that cell needs editing (see Known issues).

---

## Running it

Top to bottom, in order. Cell order is execution order and later cells depend on earlier ones.

Runtime is a few minutes on Colab's free tier. The slowest steps are the 300-refit bootstrap in
Cell 61 (31.8 s) and the learning curves in Cell 62, which refit each model twenty times.

Two things to be aware of if you modify anything:

- **Cell 44's feature builder takes a `temp_shift` argument and recomputes the degree-day columns
  from the shifted temperature.** If you shift temperature without recomputing `cdd` and `hdd`,
  every scenario result becomes meaningless. Cell 49 has an explicit check that exactly three
  columns differ between baseline and shifted matrices.
- **The train/test split is chronological, not random** (Cell 33). A random split lets the model
  see adjacent hours in both sets and produces an optimistic score that would not survive
  deployment.

---

## Notebook structure

The notebook has 68 code cells, numbered 1–68 in execution order. Markdown cells are not counted.
This is the numbering used in the notebook headings and in the report.

| Cells | What happens |
|---|---|
| 1–6 | Load both datasets; inspect shapes, missing values, timestamp formats, duplicates |
| 7–17 | Weather cleaning: city names, UTC parsing, duplicate diagnosis, collapse to city-hour |
| 18–21 | Electricity cleaning; Kelvin → Celsius; drop categorical weather columns |
| 22–23 | Population-weighted national weather series; merge on UTC timestamp |
| 24–26 | Feature engineering: local-time calendar, season, demand lags, rolling statistics |
| 27–32 | Exploratory analysis |
| 33 | Chronological train/test split at 1 January 2018 |
| 34–37 | Model A: persistence baseline, linear regression, random forest, gradient boosting |
| 38–41 | Error diagnostics and largest residuals |
| 42–43 | Feature importance — the result that forces Model B |
| 44–46 | Model B: degree-day builder, random forest, interpretable linear model |
| 47–48 | Permutation importance; temperature response curve |
| 49–50 | Warming scenarios; seasonal and hourly breakdown |
| 51 | Block bootstrap over test days — superseded by Cell 61 |
| 52–54 | Extrapolation check, calendar and balance-point sensitivity, linear cross-check |
| 55–57 | Emission factors from IPCC data; marginal factor two ways |
| 58–59 | Piecewise degree bands; time-of-day interactions |
| 60–61 | Primary model selection; bootstrap over training days |
| 62 | Learning curves under TimeSeriesSplit |
| 63–65 | Classify seasons by confidence interval; the redistribution result |
| 66–67 | CO₂ impact and emission-factor sensitivity |
| 68 | Save data, models and results summary |

Cell 51 is kept rather than deleted. It resamples which test hours get averaged, which for a
linear model captures almost nothing — the per-hour effect is a fixed function of the coefficients,
so within winter the intervals come out at about ±0.1%. That looks like precision and isn't.
Cell 61 replaces it by resampling training days and refitting, which measures uncertainty in the
coefficients. Keeping both makes the difference visible.

---

## Saved outputs

Cell 68 writes to `/content/drive/MyDrive/spain_energy_project`:

**Models**

| File | Contents |
|---|---|
| `PRIMARY_MODEL_interaction.pkl` | The model behind every reported scenario result |
| `PRIMARY_REF_columns.pkl` | Reference column list, needed to reuse the primary model |
| `phase8_response_rf.pkl` | Response random forest (robustness check) |
| `phase8_response_linear.pkl` | Single-slope linear response model |
| `phase7_linear.pkl`, `phase7_random_forest.pkl`, `phase7_gradient_boosting.pkl` | Forecasting models (benchmark only) |

**Data**

| File | Contents |
|---|---|
| `merged_clean.csv` | The merged, cleaned dataset |
| `results_summary.json` | Model comparison, primary model scores, seasonal effects |

To reuse the primary model without re-running the pipeline you need both
`PRIMARY_MODEL_interaction.pkl` and `PRIMARY_REF_columns.pkl` — the column list is what lets you
rebuild a design matrix the model will accept.

---

## Known issues

- **Model filenames still say `phase7` and `phase8`.** The notebook was reorganised to cell-based
  numbering but the save paths in Cell 68 were not renamed. `phase7_*` are the forecasting models
  from Cells 34–37; `phase8_*` are the response models from Cells 44–46.
- **The Drive folder contains files Cell 68 does not write.** `model_A_nolag.pkl`,
  `model_B_interact.pkl`, `model_B2_lagged.pkl`, `results_log.txt`, `merged_features.csv` and
  `feature_sets.json` appear in the directory listing but are leftovers from earlier runs. Clear
  the folder before a fresh run to avoid confusion about which artifacts are current.
- **Cell 68 is Colab-specific.** It mounts Google Drive. Running locally means replacing
  `project_dir` with a local path and removing the `drive.mount` call.
- **Two probable data errors are left in the dataset.** On 6 May 2018 demand drops to 19,964 MW at
  10:00 and jumps to 32,090 MW at 11:00 — a 12,000 MW swing in one hour on a Sunday morning, which
  is not physically plausible for a national system and looks like a transposed or mis-recorded
  pair of readings. They were not removed, so every reported test metric includes them.
- **Two piecewise band coefficients in Cell 58 have impossible signs** (cooling 18–22 °C at
  −29.2 MW/°C, heating below 6 °C at −35.6 MW/°C). These are collinearity artefacts from bands that
  overlap the month dummies and hold few hours. The "steepening factor: -12.1x" printed by that
  cell follows from dividing by the negative band and should be ignored; compare the 22–26 and
  above-30 bands instead.
- **The Cell 62 learning curve for the linear model spikes** to about 12,800 MW validation MAE at
  ~2,100 training hours. A block that short covers under three months, so some one-hot dummy is
  present in validation and nearly absent in training, leaving its coefficient unconstrained. It is
  an encoding artefact, not instability at the sizes actually used.
- **The per-source marginal decomposition does not close.** Slopes sum to 0.619 rather than 1.0
  (Cell 57), leaving ~38% unaccounted for. Cross-border interconnector flows, absent from this
  dataset, are the likely explanation.

---

## Caveats on interpretation

- The scenarios are **uniform temperature shifts, not climate projections**. Real warming is uneven
  across seasons and stronger in extremes than in means.
- The response is estimated from a **fixed 2015–2018 system**. Air conditioning penetration and
  heating electrification are held implicitly at their 2015–2018 values, and both are likely to
  rise on the timescale where +3 °C is relevant.
- **Specification choice moves the answer more than model fit can resolve.** The four response
  specifications differ by up to a factor of three in seasonal estimates while differing by 0.012
  in test R². The primary model was chosen on interpretability, extrapolation behaviour and
  mechanism, which is a judgement rather than a measurement.
- **Five cities do not represent Spain.** Galicia, Asturias, Aragón, Murcia, Castilla-La Mancha and
  the islands have no station in the set.
- **Public holidays are not in the feature set**, which inflates residual variance.
- **Plant efficiencies in Cell 55 carry no published range.** The IPCC fuel carbon contents come
  with low and high bounds and those are propagated; the efficiencies are single typical values held
  fixed. For gas the efficiency assumption is the larger lever — the IPCC range spans ±3.6% while
  moving CCGT efficiency from 0.50 to 0.45 moves the factor about +11%.
- **Only direct combustion emissions are counted.** Lifecycle emissions from nuclear, hydro, wind
  and solar are treated as zero, as are biogenic emissions from biomass and waste.

---

## Sources

**Data**

- Weather: `vitaliy-sharandin/energy-consumption-weather-hourly-spain` (Hugging Face)
- Energy: `vitaliy-sharandin/energy-consumption-hourly-spain` (Hugging Face)
- Populations: Instituto Nacional de Estadística,
  <https://www.ine.es/aplicaciones/piramides/piramides.htm?L=1#_munTab>

**Emission factors**

- IPCC (2006). *2006 IPCC Guidelines for National Greenhouse Gas Inventories*, Volume 2 (Energy),
  Chapter 2: Stationary Combustion, Table 2.2. Fuel carbon contents in kg CO₂/TJ with lower and
  upper bounds.
  <https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/2_Volume2/V2_2_Ch2_Stationary_Combustion.pdf>
- Plant thermal efficiencies are typical net efficiencies by technology, attributed in the notebook
  to the EU BREF for Large Combustion Plants and IEA technology data. **A specific document, table
  and year still need to be added** — no particular table is cited and no range is attached.

**Method**

The differencing approach used to estimate the marginal emission factor (Cells 56–57) is standard
in the marginal-emission-factor literature, but the notebook cites no source for it. One should be
added. The works usually cited are Hawkes (2010, *Energy Policy*) and Siler-Evans, Azevedo &
Morgan (2012, *Environmental Science & Technology*); **both need verifying against the published
articles before citation.**

The accompanying report contains a fuller literature review, which was written after the analysis
and whose citations are flagged as unverified throughout.

---
