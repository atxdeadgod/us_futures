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

---

## Daily volume + variance share by ET hour (IS 2020-2023)

Computed on the OHLCV-only bars for each instrument (1038 trading days) — used to
decide whether the closing rotation hour (15:00-16:00 ET) and the post-NYSE-close
window (16:00-17:00 ET) should be labeled. Both turned out to be major windows;
together they justify implementing halt-truncated forward windows rather than
dropping pre-halt bars.

```
--- ES ---
 h_et  vol_pct  var_pct  session
    0     0.41     1.37       ASIA
    1     0.50     1.63       ASIA
    2     0.80     2.17       ASIA
    3     1.55     3.32         EU
    4     1.42     3.01         EU
    5     1.16     2.03         EU
    6     1.16     2.03         EU
    7     1.45     2.70         EU
    8     2.68     6.55         EU
    9     9.12     7.52        RTH
   10    16.52    11.26        RTH
   11    11.73     7.81        RTH
   12     8.53     5.74        RTH
   13     7.70     5.59        RTH
   14     8.53     5.96        RTH
   15    10.54     7.33  RTH-CLOSE
   16    12.24     7.93        ETH
   17     0.29     0.31         EU
   18     0.42     1.68       ASIA
   19     0.51     2.76       ASIA
   20     0.82     5.91       ASIA
   21     0.79     2.40       ASIA
   22     0.65     1.79       ASIA
   23     0.48     1.20       ASIA
  >> 15:00-16:00 ET (closing rotation): vol=10.54%, var=7.33%

--- NQ ---
 h_et  vol_pct  var_pct  session
    0     0.59     1.13       ASIA
    1     0.71     1.35       ASIA
    2     1.05     1.86       ASIA
    3     1.80     3.14         EU
    4     1.71     2.61         EU
    5     1.35     1.72         EU
    6     1.33     1.91         EU
    7     1.68     2.35         EU
    8     3.02     6.67         EU
    9    10.28     9.03        RTH
   10    17.71    13.58        RTH
   11    12.23     8.63        RTH
   12     9.05     6.15        RTH
   13     7.84     5.51        RTH
   14     8.30     6.15        RTH
   15     9.30     6.73  RTH-CLOSE
   16     7.14     7.77        ETH
   17     0.22     0.27         EU
   18     0.52     1.71       ASIA
   19     0.65     2.31       ASIA
   20     1.00     4.77       ASIA
   21     1.00     2.09       ASIA
   22     0.85     1.55       ASIA
   23     0.66     1.02       ASIA
  >> 15:00-16:00 ET (closing rotation): vol=9.30%, var=6.73%

--- RTY ---
 h_et  vol_pct  var_pct  session
    0     0.27     0.98       ASIA
    1     0.33     1.23       ASIA
    2     0.55     1.73       ASIA
    3     1.13     2.93         EU
    4     1.11     2.56         EU
    5     0.95     1.86         EU
    6     0.98     1.80         EU
    7     1.30     2.85         EU
    8     2.59     5.90         EU
    9    11.85    12.06        RTH
   10    17.77    15.91        RTH
   11    11.45     8.56        RTH
   12     8.28     5.82        RTH
   13     7.19     5.28        RTH
   14     7.82     5.85        RTH
   15     9.54     6.09  RTH-CLOSE
   16    13.90     5.69        ETH
   17     0.36     0.34         EU
   18     0.42     1.27       ASIA
   19     0.42     2.30       ASIA
   20     0.58     4.64       ASIA
   21     0.50     1.91       ASIA
   22     0.41     1.51       ASIA
   23     0.31     0.95       ASIA
  >> 15:00-16:00 ET (closing rotation): vol=9.54%, var=6.09%

--- YM ---
 h_et  vol_pct  var_pct  session
    0     0.71     1.43       ASIA
    1     0.90     1.69       ASIA
    2     1.49     2.16       ASIA
    3     2.81     3.35         EU
    4     2.60     3.10         EU
    5     2.01     2.06         EU
    6     2.01     2.04         EU
    7     2.36     3.10         EU
    8     3.55     5.63         EU
    9    10.97     9.33        RTH
   10    16.23    11.43        RTH
   11    10.87     7.52        RTH
   12     7.66     5.50        RTH
   13     6.73     5.69        RTH
   14     7.39     5.45        RTH
   15     8.64     7.20  RTH-CLOSE
   16     7.11     7.60        ETH
   17     0.25     0.34         EU
   18     0.62     1.64       ASIA
   19     0.85     2.67       ASIA
   20     1.26     5.63       ASIA
   21     1.18     2.46       ASIA
   22     1.01     1.78       ASIA
   23     0.78     1.18       ASIA
  >> 15:00-16:00 ET (closing rotation): vol=8.64%, var=7.20%
```

Key takeaways feeding the halt-truncation decision:
- **15:00-16:00 ET (closing rotation)** is **8-11% of daily volume / 6-7% of variance** across all four instruments — comparable to mid-RTH hours; cannot be dropped.
- **16:00-17:00 ET (post-NYSE close)** is **7-14% of daily volume / 6-8% of variance**: RTY hour 16 is its single largest hour; ES hour 16 is third-largest. Originally bucketed as low-priority "ETH" — the data corrects that view.
- **Hour 10 ET (10:00-11:00) is the volume + variance peak** for all four (16-18% volume; 11-16% variance) — the post-NYSE-open hour is the most informative window of the day.
- **17:00-18:00 ET halt** is genuinely empty (hour 17 ET = 0.2-0.4% volume; what's there is residual transition-bar noise).
- **20:00-21:00 ET (Asian session start)** has elevated *variance* (4.6-5.9%) on relatively low *volume* (0.5-1.3%) — variance/volume ratio is highest here. Asian-policy / news-release driven moves on thin liquidity.
