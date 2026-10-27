# Sentinel — Hybrid Fast-Slow Streaming Anomaly Detector

Sentinel combines a robust statistical branch with a PyTorch bottleneck
autoencoder and optional asynchronous fine-tuning. Its reusable implementation
is in `backend\sentinel.py`:

- `SentinelConfig` configures the rolling window, dynamic percentile/EMA
  threshold, background retraining cadence, and autoencoder dimensions.
- `SentinelDetector(config, warmup_data_df)` fits the autoencoder and calibrates
  normal-score baselines before streaming starts.
- `process_step(timestamp, value)` returns a typed `AnomalyResult` with score,
  label, and readable evidence. Bad observations are logged and return a safe
  non-anomalous result.
- `save_state()` and `load_state()` persist model weights, rolling buffers,
  normal training windows, thresholds, and EMA state. Load state files only
  from trusted sources because they contain pickled state.
- `stop()` joins the daemon retraining worker cleanly.

The detector computes rolling mean, standard deviation, MAD, first-order
velocity, autocorrelation, and cyclical hour/day-of-week features. Anomalies are
kept out of the score baseline, and extreme values are clipped before entering
the rolling context. An alert requires a score above the adaptive threshold and
a point-magnitude trigger; values at or below `baseline_tolerance` (default
`0.05`) are suppressed and logged with the `suppressed_baseline_return` tag.
Magnitude is calibrated from the training baseline's 99.9th percentile, with
isolated moderate spikes requiring a cluster of six exceedances in the rolling
window. Extreme point spikes and strong hourly-baseline alarms remain
independently detectable. A zero-MAD training baseline falls back to its
standard deviation instead of a near-zero scale.
The threshold uses the configured score-history percentile with an EMA
adjustment and does not decay below the active baseline calibration. The original
`IntegratedHybridDetector` API remains available from `backend\detector.py` as
a multivariate adapter over the same scalar Sentinel engine.
When parseable timestamps are supplied with warm-up values, Sentinel also fits
a robust hour-of-day reference profile. Its calibrated residual score catches
time-dependent deviations such as a flat regime that would otherwise look
plausible to a global value baseline. A strong seasonal alarm is not diluted by
the autoencoder score and is excluded from automatic baseline adaptation, so
the detector does not learn a time-localized anomaly as normal. The dashboard
passes its CSV warm-up timestamps into this profile automatically; direct API
callers can pass a mapping with `timestamp` and `value` sequences or a dataframe.
Autoencoder training and warm-up should happen before production traffic; steady
state inference is the low-latency path. Fitting trains on the initial 80% of
the supplied warm-up prefix and calibrates reconstruction errors and thresholds
on a held-out final 20%, avoiding optimistic in-sample calibration. Each live
point is compared with the preceding window, so the observation being scored
does not dilute its own statistical baseline. The detector defaults to one PyTorch
CPU thread (`torch_num_threads=1`) to reduce contention and inference latency;
this setting applies to PyTorch process-wide.
Training runs for up to 30 epochs by default and uses the held-out portion for
validation-based early stopping (five epochs without improvement). The dashboard
calibrates from the first 20% of the selected dataset (at least 60 rows); the
benchmark also reserves 20% by default. This provides enough repeated temporal
coverage to calibrate the hourly baseline for daily patterns. The dashboard and default benchmark use only CSVs in the nearest
workspace `dataset` folder. Other files (including archive datasets) are not
selected unless a user explicitly configures a custom dataset or uploads one.

The dashboard's **Detection sensitivity** slider updates the score blend live.
By default, the autoencoder contributes 60% and the statistical branch 40%;
moving the slider changes the autoencoder share from 0% to 100%. The statistical
point gate is applied in statistical and hybrid modes; the low-baseline
suppression remains active in every mode.
At 100% ML the score is pure autoencoder; at 0% it is pure statistical scoring.
The calibration threshold is recomputed from held-out score components whenever
the blend changes, so the slider changes both the live score and its matching
decision threshold rather than leaving a stale threshold from the old mode.

To supply a custom stream, use **Choose CSV** and **Train & stream** in the
dashboard. Upload a CSV with `timestamp,value` columns (or headerless
timestamp/value rows), containing at least 61 finite observations. Sentinel
trains on the first 20% (at least 60 rows) and immediately switches the active
stream to that dataset. For unattended runs, set `SENTINEL_DATASET_PATH` to a
CSV file or a folder of CSV files before starting the backend. The event
timeline and live explanation panel show the active mode, branch scores,
reconstruction error, point deviation, calibrated point threshold, and the
decision threshold for each reported anomaly.

No unlabeled detector can promise perfect accuracy for arbitrary future data:
the initial prefix is treated as normal, so it should be representative and
free of the events you want to detect. Validate thresholds on labeled examples
for each custom data source before production use.

By default, a daemon worker is signaled every 1,000 streaming steps. It snapshots
up to 5,000 high-confidence normal windows (`anomaly_score < 0.2`), fine-tunes
a shadow autoencoder for two epochs, then swaps weights and reconstruction
calibration under a short model lock. It trains on a representative sample of
up to 64 windows per job to avoid competing with the stream. Inference only
holds that lock for the forward pass and never waits for the training epochs. Set
`enable_background_retraining=False` to disable it. Call `detector.stop()` when
the detector is no longer needed; the dashboard server does so during lifespan
shutdown.

Sentinel also checks a rolling candidate window against its learned historical
center and variance. It recalibrates after a sustained, stable center shift or
variance change is confirmed (300 observations by default). A variance drop to
70% or less of the learned variance can trigger adaptation; short spikes and
unstable windows do not. The anomaly-score threshold uses the active baseline's calibrated score
distribution and an adaptive percentile/EMA, with a configurable
`threshold_floor` of `0.05`; there is no fixed `0.7` score cutoff. A baseline
change recalibrates the threshold from the new data instead of retaining the
previous dataset's calibrated limit. Adaptation recalibrates feature/statistical
baselines and autoencoder error scale without training in `process_step()`. For
reviewed normal data, `update_normal_baseline()` performs a full offline
retraining.

Automatic drift adaptation is intentionally disabled when the detector has a
timestamp-conditioned hourly profile. In that mode, repeated deviations from
the expected hour-of-day pattern may be a sustained anomaly; learning them
automatically would erase the event during replay. To approve a new seasonal
normal, explicitly call `update_normal_baseline()` with reviewed data.

Example:

```python
from sentinel import AnomalyResult, SentinelConfig, SentinelDetector

detector = SentinelDetector(
    SentinelConfig(),
    warmup_data_df=normal_values,  # normal-only history
)
result: AnomalyResult = detector.process_step("2026-10-26T12:00:00", 0.42)
print(result.is_anomalous, result.anomaly_score, result.evidence)
detector.save_state("sentinel-state.pt")
```

When you have reviewed a new normal dataset, atomically replace the current
baseline outside the streaming request path:

```python
detector.update_normal_baseline(new_normal_dataframe)
```

The dataset must represent normal operation; do not retrain on an unlabeled
stream or on an unresolved anomaly period. Recalibration fits a replacement
model before swapping state, so a training failure leaves the active detector
unchanged. Offline fitting samples up to 2,048 representative windows to bound
startup time and memory; maximum training epochs, early-stopping patience, and
the retraining interval are configurable.

## Install and run the live dashboard

From `sentinel_final_bundle`:

```powershell
python -m pip install -r requirements.txt
cd backend
uvicorn main:app --reload
```

Open `frontend\index.html`. The dashboard trains on the initial dataset prefix
and streams subsequent CSV observations to the same `SentinelDetector`
implementation used by the benchmark.

## Run the tests

From the bundle root:

```powershell
python tests\test_sentinel.py
python -m unittest discover -s tests -t . -v
```

Tests cover normal streaming, point spikes, suppression of contextual-only
and zero-return alerts, live score-weight controls, safe
handling of invalid values, persistence, guarded adaptation to sustained
center/variance changes, per-dataset threshold recalibration, and a 20,000-point randomized stream. The 500,000-point
benchmark processes every post-warm-up observation and reports the processed
row count and inference latency. Hot-swap testing verifies that the background
worker updates model weights while streaming inference remains under 5 ms.

Run the dataset-specific model evaluation and generate its Markdown report:

```powershell
python -m unittest tests.test_model_accuracy -v
```

It evaluates only CSV files under the workspace `dataset` folder and writes
`tests\model_accuracy_report.md`. The provided target CSV has no row-level
labels; the report therefore identifies its April 11 reference window as a
task-derived proxy and clearly marks accuracy, precision, recall, and F1 as
proxy metrics rather than independently verified ground truth.

## Load the workspace dataset

The benchmark and dashboard use every `*.csv` file recursively found in the
nearest ancestor `dataset` folder containing CSVs by default. They do not fall
back to an archive or synthetic dataset. This supports a workspace-level dataset
alongside the bundle as well as a dataset packaged inside the bundle. CSVs may have a
`timestamp,value` header or contain rows directly in the format
`2014-04-02 07:15:00, 20.0`; surrounding spaces are accepted. The dashboard
loads the finite values from the selected CSV(s) to calibrate its detector, then streams the CSV
timestamps and values through a single-signal chart. It stops at the end of the
dataset rather than inventing or repeating observations. Each anomaly entry
shows the source timestamp, dataset row, value, score, and evidence. Dashboard
playback advances at 20 observations per second (50 ms between points). After a
full pass, **Replay dataset** starts a fresh stream with a newly calibrated
detector so previously detected anomalies remain visible on replay.

Run from `backend` to process those files:

```powershell
python run_nab_benchmark.py
```

Without labels, each file is used to fit an initial baseline from its first
20% of observations (at least 60 rows), then the remaining points are streamed
through Sentinel. The report includes file/row counts, anomaly episodes, and
latency; event precision/recall/F1 require ground-truth labels. You can point
`--data-dir` at another folder containing the same CSV format.

For NAB event-level evaluation, supply a local NAB clone's labels JSON:

```powershell
cd backend
python run_nab_benchmark.py `
  --data-dir C:\datasets\NAB\data `
  --labels C:\datasets\NAB\labels\combined_labels.json
```

The runner uses pandas' C CSV parser for NAB `timestamp,value` files, maps the
JSON label timestamps to rows, and creates configurable pre-event windows (10%
of that file by default).
It reserves a normal-only prefix for fitting, reducing that prefix if necessary
to avoid overlap with the first labeled window. Files without labeled events
are included to count false positives. It reports event precision/recall/F1, detection delay from each labeled onset
(in samples and, for parseable timestamps, seconds), and inference latency.
`--train-fraction`, `--window-fraction`,
`--threshold-percentile`, and `--epochs` can be tuned; the runner defaults to
the 99.95th percentile for precision; lower percentiles can be selected to
favor recall.

This is an event-level NAB evaluation, not the official NAB scoring application;
the selected pre-event window fraction and warm-up protocol should match the
benchmark rules used by your challenge before comparing final scores.

## Optional synthetic stress test

The synthetic stress-test generator is retained as an optional benchmark mode;
it is not used as the default training or dashboard dataset. To generate a
seeded, non-repeating 500,000-point NAB-like stream with correlated random
noise, slow random drift, and injected point spikes, level shifts, contextual
changes, dropouts, and variance changes:

```powershell
python run_nab_benchmark.py --synthetic-length --seed 42
```

The synthetic-length flag defaults to 500,000 when explicitly enabled without a
size; use `--synthetic-length 100000` to choose a smaller run.
The runner confirms every post-warm-up point was processed and reports
event-level precision/recall/F1 and detection delay from the injected onset.
Synthetic results are stress-test metrics, not official NAB benchmark results;
rerun the benchmark after detector changes rather than relying on old scores.

The synthetic score is a stress-test result, not an official NAB score. The
provided bundle does not contain the challenge's NAB data or hidden labels, so
those must be supplied using `--data-dir` and `--labels` for actual benchmark
results.
