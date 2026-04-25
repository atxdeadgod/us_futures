# V1 Labeling Architecture — FROZEN

Status: **locked 2026-04-25**. This document captures the final V1 labeling
architecture and per-instrument parameters, the empirical evidence that drove
each decision, and the reasoning at each fork in the road. Future work
(features, model training) builds on top of these labels.

> Companion document: `LABEL_TUNING_RESULTS.md` (the prior calendar-ATR baseline
> that this work superseded — kept for historical reference).

---

## Locked Architecture

```yaml
# Same across all 4 instruments
atr_mode:           time_conditional      # TC-ATR (NOT calendar)
partition_minutes:  15                    # 15-min bar-of-day buckets
halt_aware:         true                  # detect 17:00-18:00 ET halt
halt_mode:          truncate              # shrink forward window, don't drop
min_effective_T:    5                     # drops hour 16 ET (post-close burst)
bar_minutes:        15                    # 15-min OHLCV bars
```

## Per-instrument parameters

```yaml
ES:
  k_up: 1.25
  k_dn: 1.25
  T: 8                    # 2h horizon
  lookback_days: 150
  # corr 0.838, balance 0.994, pts/cost 14.57

NQ:
  k_up: 1.25
  k_dn: 1.25
  T: 8                    # 2h horizon
  lookback_days: 180
  # corr 0.824, balance 0.988, pts/cost 20.23

RTY:
  k_up: 1.00
  k_dn: 1.00
  T: 4                    # 1h horizon
  lookback_days: 180
  # corr 0.820, balance 1.000, pts/cost 13.10

YM:
  k_up: 1.25
  k_dn: 1.25
  T: 8                    # 2h horizon
  lookback_days: 150
  # corr 0.837, balance 0.993, pts/cost 18.06
```

ES, NQ, YM converged on the same `(k=1.25, T=8)` operating point. Only RTY uses
tighter, shorter `(k=1.0, T=4)` labels — its lower per-bar volume and noisier
flow mean the model gets cleanest signal from a tighter, faster horizon.

---

## Why each architectural decision

### 1. Time-conditional ATR (vs calendar)

The original calendar-ATR labeler (`LABEL_TUNING_RESULTS.md`) was producing
**degenerate labels in opposite directions across the 23h CME session**:

- **RTH hours**: barriers too tight (calendar ATR averaged in overnight calm) →
  98% of bars hit a barrier in 2h → label was effectively binary +1/-1, no
  zero class
- **ASIA hours**: barriers too wide (calendar ATR averaged in RTH motion) →
  76% of bars time-expired → labels carried no signal

Per-hour realized variance varies 5-10× across the day, but ATR over a 60-bar
calendar window is a hour-blind average — it can't track these intraday shifts.

**TC-ATR fixes this** by partitioning ATR by bar-of-day (in US/Eastern). The
ATR for 14:00 ET is the rolling mean of TR over past 14:00 ET bars only.
Result: barriers are sized to each hour's actual move-distribution.

### 2. 15-min partition granularity (vs 30-min)

We empirically tested three configurations on the per-hour balance/corr:

| Config | balance | corr | days_used |
|---|---|---|---|
| 15-min × 30-day lookback (A) | 0.96-0.99 | 0.84 (RTH) | 27,239 |
| 30-min × 30-day lookback (B) | 0.93-0.96 | 0.82 (RTH) | 27,630 |
| 15-min × 60-day lookback (C) | 0.97-0.99 | **0.88** (RTH) | 26,434 |

**Result: C > A > B uniformly across all instruments and sessions.** 30-min
partitioning loses precision even at boundaries that align cleanly with 30-min
marks. Lock 15-min partitioning and gain stability via longer lookback instead.

### 3. Halt-aware with `halt_mode='truncate'`

CME halt is 17:00-18:00 ET. Bars whose forward T-window crosses the halt would
otherwise produce labels using gapped (post-halt) prices — labels become noise
because the apparent barrier-hit was at a vacuum-gap price you couldn't fill at.

Initial fix: drop these bars entirely. But this lost the 15:00-16:00 ET
(closing rotation) hour, which volume/variance analysis showed is a major
window (8-11% of daily volume, 6-7% of variance).

**Truncate mode keeps these bars with shortened effective horizon** (T=8 → T=5
for the 15:45 ET bar). Labels are slightly weaker (corr drops from 0.879 to
0.811 across the truncated sub-bars) but well above tradeability thresholds.

### 4. `min_effective_T=5`

Drops bars whose effective T after halt-truncation falls below 5 (~75 min
horizon). Empirically validated:

- 15:00-15:45 ET (effective T = 5..8): labels healthy (balance 0.99, corr 0.81-0.84)
- 16:00 ET bar (effective T = 4): degenerate (balance 0.27, corr 0.55, frac_zero 89%)
- 16:15+ ET bars: degenerate

The structural cause: 16:00 ET bar has a high TR (close burst captured in the
bar's OHLC) but the forward 4 bars are post-burst calm. TC-ATR sizes the
barrier to the burst → forward motion can't hit it → 89% time-expire.

This is a fundamental mismatch between bar-vol and forward-vol that no barrier
adjustment fixes. **Hour 16 ET is unlabelable in this framework.** Drop it.

Also reinforced by liquidity argument: we don't actively monitor positions
post-NYSE-close, so we shouldn't ENTER trades there anyway.

### 5. Per-instrument lookback (Option B)

We tested `lookback_days ∈ {30, 45, 60, 90, 120, 150, 180}`. Headline:

```
                  60→90    90→120   120→150  150→180
ES corr Δ        +0.025    +0.008    +0.004    +0.003
NQ corr Δ        +0.008    +0.003    +0.001    +0.011  ← jumps at 180
RTY corr Δ       +0.010    +0.005    +0.005    +0.005  ← still linear
YM corr Δ        +0.029    +0.011    +0.006    +0.003
```

Two empirical findings:

1. **NQ at 180 unlocks an operating-point shift** — from (k=1.0, T=4) to
   (k=1.25, T=8) — same pattern ES/YM showed at 90. Setting NQ at 120 would
   miss this.
2. **RTY is still gaining linearly** at 180. Going further (240+) might help
   but at significant data cost.

**Per-instrument lookback (Option B) maximizes corr where it matters**:
- ES 150 — at the knee where gains taper
- YM 150 — same as ES
- NQ 180 — captures the operating-point shift
- RTY 180 — continued linear gains; capped at 180 to bound data loss

The "complexity" of per-instrument lookback is a single YAML key per
instrument — trivial. Architecture stays uniform across instruments; only the
estimation window varies.

---

## Empirical comparison: V1 (locked) vs old calendar-ATR baseline

| Instrument | Old (cal) corr | New (TC) corr | Δ corr | balance | pts/cost |
|---|---|---|---|---|---|
| ES  | 0.762 | **0.838** | **+0.076** | 0.994 | 14.57× |
| NQ  | 0.810 | 0.824 | +0.014 | 0.988 | 20.23× |
| RTY | 0.803 | 0.820 | +0.017 | 1.000 | 13.10× |
| YM  | 0.745 | **0.837** | **+0.092** | 0.993 | 18.06× |

ES and YM saw substantial corr improvements (+0.076, +0.092). These were the
most "broken" labels under calendar ATR — their wider k and longer T amplified
the regime-mismatch problem. NQ and RTY were less affected by calendar bias
but still gained modestly.

In ML terms, +0.076 corr translates to roughly 3-4 percentage points of model
accuracy in the high-corr regime — material for a 53% breakeven target.

---

## Net economics (after round-trip cost)

| Instrument | Gross win pts | Gross loss pts | Cost pts | Net win | Net loss | $/win | $/loss | Breakeven rate |
|---|---|---|---|---|---|---|---|---|
| ES  | 8.59 | -8.64 | 0.50 | 8.09 | 9.14 | $405 | $457 | **53.1%** |
| NQ  | 35.31 | -30.18 | 1.50 | 33.81 | 31.68 | $676 | $634 | **48.4%** ← edge! |
| RTY | 3.83 | -3.91 | 0.30 | 3.53 | 4.21 | $176 | $210 | **54.4%** |
| YM  | 54.37 | -54.46 | 3.00 | 51.37 | 57.46 | $257 | $287 | **52.8%** |

NQ now has an *asymmetric* operating point at lookback=180 (k_up=1.5, k_dn=1.25
were among the top combos): wider upside barriers than downside. This brings
the breakeven rate below 50% — in NQ's locked combo, the barriers themselves
already produce positive expected value.

---

## What's frozen vs what's still open

**Frozen** (no further iteration in V1):
- TC-ATR with 15-min partition + per-instrument lookback
- Halt-aware truncate with min_effective_T=5
- Per-instrument (k_up, k_dn, T) and lookback_days as locked above
- Drop hour 16 ET entirely (post-close labels degenerate by structure)

**Deferred to V1.5/V2**:
- Hour 16 ET specifically: needs a different label type (signed return target)
  to capture the post-close burst signal
- Hour 0 ET (Tokyo open) shows TC-ATR over-shrinkage; might benefit from a
  surgical T override
- Cost-aware sample weighting for training (label sparsity at low-volume hours
  isn't compensated yet)
- Live cost gating at inference (overnight spread can be 1.5-3× RTH)

---

## OOS integrity reminder

- IS window: 2020-01-01 → 2023-12-31 (locked)
- OOS window: 2024-01-01 → present (NEVER touched during labeling work)

All architectural decisions, parameter tuning, and validation diagnostics ran
on IS only. The OOS window is reserved for the final model evaluation.

---

## Next steps

With labels locked, the modeling stack becomes:

1. **Wide-start feature library** with calendar + TC variants of vol/spread/OFI
   features (tognn_us approach: ~300-800 candidates → ILP-selected ~50-80 winners)
2. **LightGBM primary classifier** trained on the locked labels, walk-forward
   purged + embargoed CV inside the IS window
3. **Meta-labeling** (AFML Ch. 3) for trade gating
4. **Trade manager** state machine with TC-ATR-sized stops
5. **Backtest engine + cost model** including per-bar realistic slippage
6. **IBKR live integration**

The labeling rework is finished. Modeling work begins.
