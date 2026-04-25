# Triple-Barrier Label Tuning Results — IS 2020-2023

Tuning run of 2026-04-24 on bigred200, on a per-instrument basis, using:

- IS window: 2020-01-01 .. 2023-12-31 (2024+ untouched OOS)
- Bars: 15-min OHLCV from front-month Algoseek TAQ (1038 trading days × 4 instruments)
- Total bars per instrument: ~92,800 (after ATR warmup + tail)
- Grid: `k_up, k_dn ∈ {1.0, 1.25, 1.5, 1.75, 2.0, 2.5}`, `T ∈ {4, 6, 8, 12, 16, 24}`,
  `atr_window ∈ {20, 40, 60}` → 648 combos per instrument
- Cost assumptions (round-trip, instrument points): ES=0.50, NQ=1.50, RTY=0.30, YM=3.00
- Pipeline scripts: `build_bars_range.py` → `tune_labels.py` → `analyze_tuning.py`
- Result CSVs: `/N/project/ksb-finance-backtesting/data/label_tuning/results/`
  - `{INSTR}_tune_2020_2023.csv` (648 rows × 28 cols per instrument)
  - `{INSTR}_stability_2020_2023.csv` (top-10 combos × 4 years per instrument)

---

## Top combos by `balance_score × |label_forward_return_corr|`

```
┌────────────┬──────┬──────┬─────┬───────┬─────────┬───────┬────────────┬──────────┐
│ Instrument │ k_up │ k_dn │  T  │ atr_w │ balance │ corr  │ mean_pts ± │ pts/cost │
├────────────┼──────┼──────┼─────┼───────┼─────────┼───────┼────────────┼──────────┤
│ ES         │ 1.25 │ 1.25 │ 8   │ 60    │ 0.993   │ 0.762 │ ±7.28      │ 14.5×    │
│ NQ         │ 1.00 │ 1.00 │ 6   │ 60    │ 0.990   │ 0.810 │ ±23.7      │ 15.8×    │
│ RTY        │ 1.00 │ 1.00 │ 6   │ 60    │ 0.992   │ 0.803 │ ±3.80      │ 12.7×    │
│ YM         │ 1.25 │ 1.25 │ 8   │ 60    │ 0.995   │ 0.745 │ ±55.6      │ 18.5×    │
└────────────┴──────┴──────┴─────┴───────┴─────────┴───────┴────────────┴──────────┘
```

Column meanings:

- **balance** — entropy of label distribution `{-1, 0, +1}` over `log(3)`. 1.0 = perfect 33%/33%/33%; we want ~1.
- **corr** — pearson correlation between integer label and realized log return over the labeling horizon. High = labels faithfully encode return direction.
- **mean_pts ±** — average barrier distance in instrument points (= k × mean ATR by construction). For label=+1 it's positive, for label=-1 it's negative; symmetric barriers (k_up=k_dn) make these mirror images.
- **pts/cost** — `|mean_pts| / round_trip_cost_pts`. Tradeability ratio; >2 comfortable, all here >12.

Interpretation:
- All four converge on **atr_window=60** (longest in grid) and **symmetric barriers** with **T=6-8** (1.5-2h horizon at 15-min bars).
- **NQ has highest label-return correlation (0.81)** — strongest signal in the label itself. ES at 0.76 and RTY at 0.80 are also strong; YM at 0.745 is the weakest of the four.
- All `pts/cost` ratios well above tradeability threshold; labeling layer is not the bottleneck.

---

## Net economics per top combo (gross → net after cost)

```
┌────────────┬───────────┬────────────┬──────┬─────────┬──────────┬───────┬────────┬────────────────┐
│ Instrument │ Gross win │ Gross loss │ Cost │ Net win │ Net loss │ $/win │ $/loss │ Breakeven rate │
├────────────┼───────────┼────────────┼──────┼─────────┼──────────┼───────┼────────┼────────────────┤
│ ES         │ 7.27      │ 7.29       │ 0.50 │ 6.77    │ 7.79     │ $339  │ $390   │ 53.5%          │
│ NQ         │ 23.66     │ 23.78      │ 1.50 │ 22.16   │ 25.28    │ $443  │ $506   │ 53.3%          │
│ RTY        │ 3.80      │ 3.79       │ 0.30 │ 3.50    │ 4.09     │ $175  │ $205   │ 53.9%          │
│ YM         │ 55.61     │ 55.62      │ 3.00 │ 52.61   │ 58.62    │ $263  │ $293   │ 52.7%          │
└────────────┴───────────┴────────────┴──────┴─────────┴──────────┴───────┴────────┴────────────────┘
```

Conversion: ES=$50/pt, NQ=$20/pt, RTY=$50/pt, YM=$5/pt.

Key takeaways:
- **Cost asymmetry**: cost adds to losses, subtracts from wins → net_loss > net_win, breakeven ~53% across all four.
- **All breakeven rates cluster at 52.7-53.9%** — the cost/barrier ratio is roughly constant across instruments.
- **At 56% directional accuracy** (mid-range for modern intraday ML), expected per-contract net edge:
  - ES: ~$19/trade
  - NQ: ~$25/trade
  - RTY: ~$8/trade  ← thinnest, most fragile to accuracy degradation
  - YM: ~$18/trade

---

## Per-year stability (worst-year-balance combos)

| Instrument | Worst-year combo | bal_min | corr_min | comment |
|---|---|---|---|---|
| ES  | (k=2.0, T=16, atr=60)  | 0.991 | 0.734 | wider barriers more stable than rank-1 |
| NQ  | (k=1.75/1.5, T=12, atr=60) | 0.991 | 0.774 | mild asymmetric edge in 2020 |
| RTY | (k=1.75/1.5, T=12, atr=60) | 0.988 | 0.778 | |
| YM  | (k=2.0, T=16, atr=60)  | 0.989 | 0.739 | |

Across all four instruments and all 4 IS years, the stability-best combos hold:
- **balance_score ≥ 0.988 in every year**, including COVID 2020
- **corr 0.73-0.89 across all years** — no regime collapse

---

## Time-expired class (label=0) characteristics

| Instrument | mean_ret_pts_zero | frac_zero |
|---|---|---|
| ES  | +0.47 | 27.8% |
| NQ  | +1.06 | 26.6% |
| RTY | +0.18 | 27.4% |
| YM  | +2.92 | 28.2% |

Time-expired bars have a small positive drift across all four instruments — consistent with the 2020-2023 equity bull regime. ~28% of bars are zero-labels. Probably not worth trading at the label level (drift ≈ cost), but the model can still leverage the structure of when these occur.

---

## Decisions to lock

For V1 modeling, use these per-instrument label parameters (the balance × |corr|-best combos):

```yaml
ES:  { k_up: 1.25, k_dn: 1.25, T: 8, atr_window: 60 }
NQ:  { k_up: 1.00, k_dn: 1.00, T: 6, atr_window: 60 }
RTY: { k_up: 1.00, k_dn: 1.00, T: 6, atr_window: 60 }
YM:  { k_up: 1.25, k_dn: 1.25, T: 8, atr_window: 60 }
```

The wider stability-best combos remain available as fallback options (V2) if regime shift hurts the primary.

---

## Open questions / next directions

- **Trade management uplift**: meta-labeling + dynamic stop placement can shift effective net win/loss meaningfully — the 53% breakeven is a *floor*, not a target.
- **Alternative labelers worth comparing**: efficiency-ratio-based labels (Kaufman), drawdown-aware labels, signed-return regression target. May offer different tradeoffs in label noise vs predictability.
- **Wider T grid**: current grid tops at T=24 (6h horizon). Worth probing T={32, 48} (8-12h) to see if longer horizons offer better corr at the cost of fewer trades.
- **Verify cost assumptions** with actual IBKR commission + measured slippage on small live trades; especially RTY/YM where cost estimates are lower-confidence.
