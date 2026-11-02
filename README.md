# AI Time-Series Forecasting (geoskill-ai-time-series-forecast)

> LSTM time-series forecasting network (torch+CUDA, default) + classical linear/polynomial/AR(p) interpretable baselines: multi-step extrapolation + holdout MAE/RMSE validation + per-pixel raster output

---

## 1. Overview

Performs single-layer LSTM multi-step extrapolation on remote sensing time series (per-pixel NDVI/temperature/backscatter), validates on a holdout period (MAE/RMSE), and outputs prediction rasters for each future step plus a per-pixel validation RMSE map. **Core model**: single-layer LSTM (many-to-one + recursive extrapolation), trained/inferred on torch + CUDA by default; pre-trained weights `ts_lstm_weights.pt` are bundled with the skill (if missing, they are automatically trained on the GPU at first run and saved to disk). Three interpretable classical baselines (`--method linear|poly|ar`) are also provided for comparison and for GPU-free environments. Each model family has unit tests against hand-computed references for fitting accuracy, AR coefficient recovery, and holdout MAE/RMSE.

## 3. Quick Start

```bash
pip install -r requirements.txt
python geoskill-ai-time-series-forecast.py --bbox 116 39 117 40 --synthetic --output-dir ./out
```

## 4. CLI Parameters

Run `python <skill>.py --help` for the full list. Common parameters:

| Parameter | Type | Description |
|---|---|---|
| `--bbox` | `float[4]` | WGS84 bounding box `min_lon min_lat max_lon max_lat` |
| `--input` | `path` | Local input file (GeoJSON/GeoTIFF/etc.) |
| `--output-dir` | `path` | Output directory (default `./output`) |
| `--synthetic` | `flag` | Use synthetic data instead of real input |
| `--quiet` | `flag` | Suppress non-essential stdout |

## 5. Input / Output

| File | Format | Description |
|---|---|---|
| `forecast.tif` | GeoTIFF | Predictions for the future horizon steps (one band per step) |
| `validation_rmse.tif` | GeoTIFF | Per-pixel holdout validation RMSE |
| `forecast_report.json` | JSON | Global/center-pixel predictions and error metrics |
| `output-manifest.json` | JSON | Run manifest (input/output/QA/exit code) |


## 6. Technical Principle

(see SKILL.md for details)

## 8. License

MIT License. See [`LICENSE`](./LICENSE) for full text.

---

> 中文完整版：[references/SKILL.zh-CN.md](references/SKILL.zh-CN.md)
