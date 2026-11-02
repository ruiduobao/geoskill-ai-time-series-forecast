---
name: geoskill-ai-time-series-forecast
description: 'LSTM 时序预测网络（torch+CUDA，默认） + 经典线性/多项式/AR(p) 解释基线：多步外推 + 留出 MAE/RMSE 验证 + 逐像元栅格输出'
---

# AI时序预测 | AI Time Series Forecast

Performs multi-step extrapolation of remote sensing time series (per-pixel NDVI/temperature/backscatter) with a single-layer LSTM, validates on a held-out period (MAE/RMSE), and outputs forecast rasters for each future step plus a per-pixel validation RMSE map.

**Core model**: single-layer LSTM (many-to-one + recursive extrapolation), trained/inferred by default on torch + CUDA; the skill ships with pretrained weights `ts_lstm_weights.pt` (automatically trained on GPU and persisted at first run when missing). Three interpretable classical baselines (`--method linear|poly|ar`) are also retained for comparison and GPU-free environments. The fitting accuracy of each model class, AR coefficient recovery, and held-out MAE/RMSE all have unit tests against hand-computed baselines.

## Dependencies / 依赖

```bash
pip install numpy rasterio scipy torch --index-url https://download.pytorch.org/whl/cu121
```

Classical baseline methods (`--method linear|poly|ar`) do not require torch.

## Usage / 使用方法

### Example 1 (Synthetic Data, Offline)

```bash
python geoskill-ai-time-series-forecast.py --bbox 116.0 39.0 117.0 40.0 --synthetic --output-dir ./out
```

### Example 2: Synthetic Time-Series Forecast (Offline)

```bash
python geoskill-ai-time-series-forecast.py --bbox 116.0 39.0 117.0 40.0 --synthetic --method linear --horizon 4 --output-dir ./out
```

### Example 3: Polynomial Trend Extrapolation

```bash
python geoskill-ai-time-series-forecast.py --bbox 116.0 39.0 117.0 40.0 --synthetic --method poly --degree 2 --horizon 6 --output-dir ./out
```

### Example 4: Autoregressive Model

```bash
python geoskill-ai-time-series-forecast.py --input series.tif --method ar --order 3 --horizon 3 --output-dir ./out
```

## Output / 输出

| File | Format | Description |
|---|---|---|
| `forecast.tif` | GeoTIFF | Forecasts for the next horizon steps (one band per step) |
| `validation_rmse.tif` | GeoTIFF | Per-pixel held-out validation RMSE |
| `forecast_report.json` | JSON | Global/center-pixel forecasts and error metrics |
| `output-manifest.json` | JSON | Run manifest (inputs/outputs/QA/exit code) |

## Data Source / 数据源 / Source

Local multi-band GeoTIFF (bands = time steps), or --synthetic (NDVI-like cube of trend + annual cycle + noise).

## Privacy / 隐私声明 / Privacy

- Runs offline by default; `--synthetic` mode requires no network at all.
- All processing is done locally; no user data is ever uploaded.

## License / License

MIT

---

> 中文完整版：需要时再读 [references/SKILL.zh-CN.md](references/SKILL.zh-CN.md)，不要默认全量加载。
