# Phase 1 — Directional ES Intraday Strategy

Status: draft, 2026-04-23. Narrow-start first milestone before expanding to the 30-contract vision in [`DESIGN.md`](DESIGN.md).

---

## 1. Scope

- **Single instrument traded**: ES (E-mini S&P 500 front-month continuous, with roll management)
- **Strategy class**: directional, predictive forecast of short-horizon ES forward return; NOT index-basis arbitrage
- **Horizon**: 15-min forward log return primary target; 5/30/60-min as multi-task auxiliary targets for regularization
- **Rebalance**: every 15 min in RTH (26 scoring windows/day), typical 5–20 actual position changes/day after filters
- **Capital scale**: $10–50M (IBKR PM broker, ~4.5× effective leverage assumed)
- **Target net Sharpe**: 1.5–2.5 after realistic costs
- **Target max DD**: <6% intraday, <10% rolling monthly

---

## 2. Feature set — wide candidate pool → ILP-selected survivors

**Strategy**: mirror the equity-book flow (`tognn_us`: 864 dashboard variants → ILP → 77 selected). Start wide, let dashboard IC + ILP with correlation constraint do the selection discipline.

- **Base features**: ~60 across 6 tiers below (informational inputs only — no cross-sectional panel construction since we predict ES alone; features span asset classes as *predictors*, not as a cross-section to rank)
- **Variant expansion**: parameter grids per applicable feature over `{windows: 5, 10, 15, 20, 30, 60}`, `{lags: 1, 3, 5, 10}`, `{depths: 1, 3, 5, 10}`, transforms `{raw, rolling_z, ema3, slope_N, rank_over_window}`
- **Candidate pool**: ~300–800 feature×variant combinations
- **Dashboard pass**: IC / t / ICIR / hit-rate at 5 / 15 / 30 / 60-min horizons (direct analog of `feature_dashboard_v2_sector_beta.csv`)
- **ILP selection**: ρ < 0.45 correlation constraint, select top `~40–80` by max_abs_t (reuse `stage2_ilp_selection.py` adapted)
- **Transform strategy for single instrument**: since cross-sectional rank is undefined (N=1), replace Gauss-rank with **rolling time-series z-score** or **rolling quantile rank** over a 252-bar window; decide by ablation

### Tier 1 — ES own microstructure (highest expected IC)

| Feature | Source class | Refresh |
|---|---|---|
| Deep OFI (L1–L10 signed ΔQ, distance-weighted) | new `DeepOFI` class | 1 min |
| Aggressor imbalance (signed trade $) | `AggressorSideVolume`, `DollarImbalance` | 1 min |
| Microprice drift | `MicropriceDrift` | 1 min |
| Top-of-book imbalance | `OrderImbalance` | 1 min |
| Spread width + acceleration | `SpreadRelBps`, `SpreadAcceleration` | 1 min |
| Quote-to-trade ratio | `QuoteToTrade` | 1 min |
| Large-trade volume share | `LargeTradeVolumeShare` | 1 min |
| Quote movement directionality | `QuoteMovementDirectionality` | 1 min |

### Tier 2 — ES own time-series / vol
- Log returns at 1, 3, 5, 15, 30 min (5)
- Return autocorrelation (10-bar window)
- Realized vol (1-min, 5-min windows)
- Parkinson range vol
- Intraday momentum flag (first-30-min return sign)
- VWAP deviation (price vs session VWAP)
- Trend strength

### Tier 3 — Direct cross-asset lead-lag (high marginal value)

Each is a return at specified lag predicting forward ES return:

| Predictor | Lags (min) | Rationale |
|---|---|---|
| NQ | 1, 3, 5 | Often leads ES in tech-driven days |
| RTY | 1, 3 | Breadth signal |
| YM | 1, 3 | Mega-cap / defensive signal |
| ZN (10Y) | 1, 5 | Rates-equity correlation, regime-dependent |
| VX (front) | 1, 5 | Vol regime pulse |
| 6E | 1, 5 | DXY proxy (inverse) |
| NQ–ES beta-adjusted spread change | 1, 5 | Pure tech-vs-broad signal |

~15 features.

### Tier 4 — Cross-sectional equity features (information about market state)

| Feature | Why it predicts ES |
|---|---|
| Sector dispersion (std of 11 sector ETF returns, last 5 min) | Regime confidence gate: low dispersion → lead-lag stronger |
| Top-5 mega-cap avg return (MSFT, AAPL, GOOGL, NVDA, AMZN) | Mega-caps drive ES directly; distinct signal from NQ |
| Breadth: % of 11 sectors with positive 5-min return | Classic regime indicator; extremes mean-revert |
| Leader-laggard spread: max sector return − min | Rotation intensity |
| Cross-sectional momentum: first PC of sector return vector | Broad-market directional signal |
| Risk-on/off proxy: XLY − XLP return spread | Pure risk appetite |

~6 features. All computable from `us-equity-1sec-taq` (2020–2024 on drive).

### Tier 5 — Vol / options regime
- VIX level + 20-day z-score
- VIX/VIX3M ratio (term structure slope)
- VX front − VX 2nd spread
- VVIX level
- Optional (paid vendor): dealer-gamma flip distance

~5 features.

### Tier 6 — Temporal / event
- Minute-of-day cyclical encoding (sin/cos) (2)
- Day-of-week dummies (4)
- Distance (minutes) to next scheduled macro release
- Binary flags: ±15-min FOMC / NFP / CPI / EIA / ISM
- Settlement-window flag (15:45–16:00 ET)
- Post-FOMC 2-hour window

~10 features.

### Deliberately excluded
European/Asian indices intraday during US RTH; individual commodity futures (CL/GC/NG); intraday credit spreads (HYG/LQD); single stocks beyond mega-caps; kitchen-sink correlations.

---

## 3. Labels, model, and meta-labeling

### 3.1 Triple-barrier labels (primary model training)

Following Lopez de Prado (AFML Ch. 3). For each candidate entry bar `t`, define three barriers:

- **Upper** (profit): `price_t × (1 + k_up × ATR_t / price_t)`
- **Lower** (stop): `price_t × (1 − k_dn × ATR_t / price_t)`
- **Vertical** (time): `t + T` bars

Label = first barrier touched, yielding one of `{+1, −1, 0}` (upper / lower / time-expired). This replaces hand-designed return-threshold labels and naturally encodes the trading horizon + risk asymmetry.

Starting barrier params (placeholders, to be tuned):
- `k_up` ≈ 1.5 (in ATR units)
- `k_dn` ≈ 1.0
- `T` ≈ 12 bars (3 hours on 15-min bars)
- `ATR_t` computed on 20-bar rolling window

### 3.2 Primary model

**Baseline**: Ridge regression, ternary (sign of triple-barrier label) or continuous forward return at 15-min horizon. Multi-task variant predicts auxiliary 5/30/60-min horizons for regularization. L2 penalty tuned on walk-forward CV.

**Stretch**: LightGBM single-task classifier on ternary labels; handles time-of-day + event flags as categoricals.

### 3.3 Meta-labeling (secondary model)

Per AFML Ch. 3: on top of the primary model, train a second binary classifier predicting `should_take_signal ∈ {0, 1}` given the primary signal, its confidence, and the feature set. Purpose:

- Improves precision by filtering low-conviction primary signals
- Natural home for bet-size scaling (predicted probability → position size multiplier)
- Cleanly isolates direction decision (primary) from trade-or-skip decision (meta)

Output: final signal = `primary_sign × meta_prob × vol_scaler`, thresholded against trade-admission gates.

### 3.4 Validation

- **Walk-forward with purging + embargo** (AFML Ch. 7): train on expanding window, predict next month, purge overlapping labels and embargo `T` bars around each fold boundary to prevent label leakage
- **2025 is holdout — untouched during any hyperparameter selection**
- Year-by-year IC calibration (2020/21/22/23/24) per feature to flag single-regime features
- ILP-based ρ < 0.45 feature selection (reuse `stage2_ilp_selection` from `tognn_us` adapted)
- Dashboard v2 on ES bars — futures analog of the 864-feature equity dashboard (`tognn_us/data/feature_dashboard_v2_sector_beta.csv`)

### 3.5 Hyperparameter tuning

- **Bayesian optimization via Optuna** (not random search) over primary + meta + barrier params, using walk-forward objective with embargo
- Fixed search budget per refit; report parameter stability across folds as a confidence diagnostic
- Metrics to optimize (in priority order):
  1. OOS Sharpe net of modeled costs
  2. Realized R per winning trade (not just hit rate)
  3. Max drawdown bounded
  4. Parameter stability across folds

---

## 4. Trade management layer

~50% of realized Sharpe lives here. Structure follows triple-barrier exits (inherited from label construction) + a small number of pre-trade filters and post-entry overrides.

### 4.1 Pre-trade filters (entry gates)

| Gate | Rule |
|---|---|
| Signal threshold | Enter only if \|meta_prob × primary_sign\| > k_entry; tuned walk-forward |
| Cost gate | Skip if expected edge × notional < 3 × expected round-trip cost |
| Regime gate | Suspend if VIX > 30, or macro event ±15 min, or spread > 2 × normal |
| Slope-confirmation filter (optional) | Enter only if multi-window (10/20/30) slopes agree with signal direction — inherited from legacy framework as a whipsaw defence; **ablation-test against no-filter variant** before including |
| Trade-frequency cap | Max N round-trips/day (N tuned) |

### 4.2 Post-entry rules

| Rule | Behaviour |
|---|---|
| Triple-barrier exits | Upper (profit), lower (stop), vertical (time) from the label spec — first touch wins, exit on touch |
| Trailing lower barrier (optional) | Ratchet the lower barrier only in favour direction (never loosen); ATR-based step |
| Opposing-signal override | If new primary signal is opposite and conviction > k_override, exit regardless of barrier state and P&L |
| EOD flatten | Flat by 15:45 ET if in loss; flat by 16:00 ET unconditional (ES CME close is 17:00 ET — policy chooses earlier flatten) |

### 4.3 Sizing

- `position = notional_cap × meta_prob × primary_sign / σ_target`
- `σ_target` = rolling 20-bar realized vol × vol_scale
- Per-trade cap, daily gross cap, per-instrument gross cap

### 4.4 Book-level risk

| Rule | Threshold |
|---|---|
| Daily drawdown kill-switch | Hard shut-off at −2% book DD; resume next day |
| Weekly drawdown flag | At −5% rolling weekly DD, cut sizing in half; manual review |
| Slippage monitor | Realized vs model fills; if slippage > budget for 5 consecutive days, widen entry threshold |
| PnL attribution | Decompose daily PnL: signal edge, cost, slippage, timing; reviewed daily |

### 4.5 Design principles (carried from review)

- **Start minimal, add only what ablation-tests justify** — every parameter is a potential overfit
- **Measure realized R per winner**, not just hit rate; optimize `realized_R × hit_rate − (1 − hit_rate) − costs`
- **Primary model handles direction + horizon; trade manager handles principled risk** — don't let trade-management compensate for weak signal
- **Live-backtest parity**: simulate slicing, slippage, partial fills, rejected orders in backtest with same order logic as live

---

## 5. Execution

- **Broker**: IBKR (Portfolio Margin), TWS API via `ib_insync`
- **Order routing**: Adaptive Algo for 15-min slicing where size warrants; direct limits for small clips
- **Data**: Algoseek for backtest; IBKR real-time for live
- **Cost model**: realistic = 1.5–3× naive half-spread + exchange fees + clearing; calibrate on paper-trade
- **Latency budget**: ~50ms end-to-end via SmartRouter; fine for 15-min rebalance

---

## 6. Data requirements (all available)

| Tier | Source | Status |
|---|---|---|
| 1, 2 | Algoseek ES MBP-10 + TAQ (2020–2025) | On Expansion drive |
| 3 | Algoseek NQ/RTY/YM/ZN/VX/6E (depth + TAQ) | On drive |
| 4 | Algoseek us-equity-1sec-taq (2020–2024) | On drive |
| 5 | VIX/VIX3M/VVIX (Cboe daily free; intraday via IBKR subscription) | Need subscription |
| 6 | Macro calendar (FOMC/BLS/EIA schedules) | Public, trivial ingest |

---

## 7. Build order (6–8 weeks)

1. **Data ingestion** (week 1): Algoseek ES depth + TAQ → polars → 15-min bars + 1-min microstructure features; contract-roll logic for continuous front-month series
2. **Feature engineering — wide generation** (weeks 2–3): all 6 tiers via `feature-factory` with parameter grids; add custom classes (`DeepOFI`, etc.); incremental/cached computation so live updates are cheap
3. **Triple-barrier labeling** (week 2, parallel): label generator, ATR compute, purge + embargo utilities
4. **Dashboard v2 on ES** (week 3): IC/t/icir/hit-rate at 5/15/30/60-min horizons across all candidate features (~300–800)
5. **ILP feature selection** (week 3): ρ<0.45 constraint, top 40–80 selected — reuse `stage2_ilp_selection` from `tognn_us` adapted
6. **Primary model baseline** (week 4): LightGBM (baseline) + Ridge (ablation) on triple-barrier labels, walk-forward backtest with Optuna tuning, aim ≥1.0 net Sharpe as floor
7. **Meta-labeling model** (week 5): binary classifier, bet-size gate, precision-recall analysis on 2024
8. **Trade manager module** (weeks 5–6): triple-barrier exits + filters + overrides + sizing + kill-switches; unit-tested state machine, thread-safe
9. **Cost realism & validation** (week 7): 2025 OOS (first time it's touched), slippage calibration, DD kill-switch testing, live-backtest parity check
10. **Paper trade + live readiness** (week 8): IBKR integration via `ib_insync`, live parity check vs backtest, go-live checklist

---

## 8. Success criteria for advancing to Phase 2

- Net Sharpe ≥ 1.5 on 2025 OOS
- Max DD < 8%
- Live paper-trade results within ±0.3 Sharpe of backtest after ≥1 month
- Per-feature IC stability across 2020–2024 years

---

## 9. Strategy evolution path

Each version proves its own edge before adding the next. Moving from directional → arbitrage is where the Sharpe step-up comes from (dollar-neutral construction reduces vol); this path earns it in stages.

| Version | Scope | Expected net Sharpe | What it proves |
|---|---|---|---|
| **V1 (this doc)** | Directional ES single-instrument | 1.5–2.5 | Predictive quality of the feature + label + model stack; trade-management discipline |
| **V2** | Same per-instrument directional model replicated on NQ, RTY, YM (4 independent equity-index books); optional correlation-aware book sizing | 1.8–2.8 | Model architecture transfers beyond ES; simple diversification gain |
| **V3** | **Arbitrage / dollar-neutral structures**: index basis (ES vs SPY/sector basket), sector rotation long-short with ES hedge, pair trades on index family | 2.5–3.5 | Vol reduction from dollar-neutral construction; Sharpe lift from removing market beta |
| **V4** | Full Book A (30-contract per-instrument cross-asset) + Book B (Treasury flies, crush, crack, SOFR packs, WTI-Brent) per [`DESIGN.md`](DESIGN.md) | 3–4 combined | Full diversified book; production scale |

Each step is a go/no-go decision: if V1 lands at <1.0 net Sharpe, we diagnose before adding V2 complexity; if V3's arbitrage layer doesn't add Sharpe net of execution complexity, we stay at V2.
