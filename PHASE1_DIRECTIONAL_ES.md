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

## 2. Feature set (~50–60 features, 6 tiers)

Informational inputs only — no cross-sectional panel construction since we predict ES alone. Features span asset classes as *predictors*, not as a cross-section to rank.

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

## 3. Model

**Baseline**: Ridge regression, multi-task (auxiliary 5/30/60-min horizons), with L2 penalty tuned on walk-forward CV. Monthly refit, 3-month embargoed holdout.

**Stretch**: LightGBM single-task on 15-min horizon, with categorical handling for time-of-day and event flags.

**Validation**:
- Expanding-window walk-forward, monthly refits
- Year-by-year IC calibration (2020/21/22/23/24) per feature to flag single-regime features
- ILP-based ρ<0.45 feature selection (reuse `stage2_ilp_selection` from `tognn_us` adapted)
- Dashboard v2 on ES bars (futures analog of equity 864-feature dashboard)

---

## 4. Trade management layer

The signal is multiplied by this layer; ~50% of realized Sharpe lives here.

| Component | Rule |
|---|---|
| Signal threshold | Trade only if \|ŷ\| > k·σ(ŷ); k tuned on walk-forward |
| Vol-scaled sizing | position ∝ ŷ / σ², capped at per-trade and daily gross limits |
| ATR-based stop | Flatten on 2×ATR adverse; profit-take at 1.5–2×ATR |
| Time stop | Auto-flatten at signal horizon (15–30 min) regardless of P&L |
| Trailing stop | Ratchet stop on favourable moves; never loosen |
| Target ratchet | On target-cross: lock a tighter stop, shrink subsequent target |
| Regime gate | Suspend if VIX > 30, or macro event ±15 min, or spread > 2× normal |
| Cost gate | Skip trade if \|ŷ\| × notional < 3× expected (half-spread + fees) |
| Opposing-signal cut | Exit position immediately on opposing signal when in loss |
| Slope-confirmation filter | Only enter if multi-window (10/20/30) slopes agree with signal direction |
| Daily DD kill-switch | Hard shut-off at −2% book DD |
| Trade frequency cap | Max ~20 round-trips/day (tune) |
| EOD flattening | Flat by 15:45 ET if in loss; flat by 16:00 ET unconditional |
| Live slippage tracker | Realized vs model fills; widen thresholds if slippage > budget |
| PnL attribution | Decompose daily PnL: signal edge, cost, slippage, timing |

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

1. **Data ingestion** (week 1): Algoseek ES depth + TAQ → polars → 15-min bars + 1-min microstructure features
2. **Feature engineering** (weeks 2–3): Tiers 1 + 2 + 3 via `feature-factory`; add `DeepOFI` class
3. **Dashboard v2 on ES** (week 3): IC/t/icir at 5/15/30/60-min horizons
4. **Model baseline** (week 4): Ridge multi-task + walk-forward backtest, aim ≥1.0 net Sharpe as floor
5. **Add Tiers 4/5/6** (week 5): cross-sectional equity features, vol regime, temporal; re-dashboard, re-train
6. **Trade management** (weeks 5–6): ATR stops, trailing, regime gate, cost gate, slope confirmation
7. **Cost realism & validation** (week 7): 2025 OOS, slippage calibration, DD kill-switch testing
8. **Paper trade + live readiness** (week 8): IBKR integration, live parity check, go-live checklist

---

## 8. Success criteria for advancing to Phase 2

- Net Sharpe ≥ 1.5 on 2025 OOS
- Max DD < 8%
- Live paper-trade results within ±0.3 Sharpe of backtest after ≥1 month
- Per-feature IC stability across 2020–2024 years

---

## 9. Expansion path

**Phase 2**: same per-instrument directional model replicated on NQ, RTY, YM (4 equity index models running independently); optional correlation-aware book-level sizing.

**Phase 3**: extend to 30-contract cross-asset per-instrument book per [`DESIGN.md`](DESIGN.md).

**Phase 4**: add Book B spreads.
