# TODO — Deferred items

Running list of features / modules consciously deferred from V1 build. Each entry
notes the tier, why it's deferred, and the trigger for enabling.

V1 focus is **microstructure + pattern + regime features** for single-contract ES.
Macro and regression-based items can plug in later without pipeline rework.

For what's already wired, see `BUILD_PIPELINE.md` § "V1 feature inventory".
For pipeline architecture, see `REFACTOR.md`.

---

## Deferred from V1 — feature work

### Macro calendar + event features (T6.05, T6.06, T6.08)

- **Scope**: Distance to next macro release (minutes), ±15-min event-window flag
  per event type (FOMC/NFP/CPI/EIA/ISM), post-FOMC 2-hr window flag.
- **Why deferred**: V1 is a microstructure-driven strategy. Macro events add
  exogenous regime information but aren't core to the 15-min signal.
- **Trigger to enable**: When we observe V1 OOS Sharpe degradation around macro
  release windows (visible in PnL attribution by hour-of-day).
- **Work required**: macro-calendar CSV (scrape BLS/Fed/EIA), `src/features/events.py`
  with `distance_to_next_release`, `event_window_flag(kind, width_min)`.

### L2 cancel proxy levels 2-5 (T1.43 extension)

- **Scope**: Currently `bars_cancel.py` emits L1-only cancel proxy
  (`net_bid_decrement_no_trade_L1`, `net_ask_decrement_no_trade_L1`). FEATURES.md
  schema calls for L1-L5.
- **Why deferred**: L2-L5 attribution requires handling level-promotion/demotion
  when inner prices change (cancel at L2 today was L3 yesterday).
- **Trigger**: If ILP retains the L1 cancel proxy as material, extend to L2-L5.

### T1.38 BookResilience (genuinely needs per-event tracking)

- **Scope**: Depth state at +N seconds AFTER each large trade — does the book
  refill, or stay thin? Per-trade state + lookforward.
- **Why deferred**: Requires interleaved trade + depth event stream with
  per-event N-second lookforward. Our current bar-builders aggregate per 5-sec
  windows, not per-event.
- **Trigger**: V1.5 if other large-trade-related features (T1.23, T1.47) carry
  signal — likely BookResilience is correlated.
- **Work required**: New `src/data/bars_resilience.py` per-trade processor + new
  bar-builder pass.

### bar_cross regression features (T3.10 extensions)

- **Scope**: `factor_model_residual`, `idiosyncratic_volatility`,
  `cointegration_zscore`, `ou_half_life`, `residual_band_position`,
  `residual_dispersion`, `residual_shock_repair`.
- **Why deferred**: Each requires rolling OLS / state-space estimation. Nontrivial
  implementation, marginal ex-ante signal value vs our microstructure features.
- **Trigger**: V2+ if we find ES-vs-peer cross-sectional residual alpha.

### l2_cross regression features

- **Scope**: `hedge_ratio`, `rolling_beta`, `half_life_mean_reversion`,
  `information_share` (Hasbrouck).
- **Why deferred**: Same as above — rolling regression machinery needed.
- **Trigger**: V2 when we implement pairs trading (Book B spreads).

### Tier 4 — Cross-sectional equity features (V1.5)

- **Scope**: Sector ETF dispersion, breadth, mega-cap composite, XLY-XLP
  risk-on/off, etc.
- **Why deferred**: Requires equity 1-second TAQ ingest for 11 SPDR sector ETFs
  + top-5 mega-caps as a parallel bar-builder pipeline.
- **Trigger**: After V1 OOS Sharpe meets floor (≥1.0 net). If V1 suggests ES
  intraday is sensitive to sector rotation beyond what NQ already captures,
  enable Tier 4.

### Tier 5b — Term structure (CL/NG/ZN curve)

- **Scope**: BasisSlope, BasisMomentum, CurveCurvature, RollYield for commodity/
  rates contracts. Parked because ES basis is pure cost-of-carry (no directional
  info on outright ES).
- **Trigger**: V3+ when we build Book B (curve trades in rates/energy).

### Intraday GEX via tick_option_price (T5 upgrade)

- **Scope**: Upgrade from EOD SPX-chain GEX to intraday GEX using WRDS
  `optionm.tick_option_price_YYYY` (available 2006-2023).
- **Why deferred**: Different schema (`securityid` not `secid`), coverage ends
  2023. Current daily GEX is sufficient for V1.
- **Trigger**: If V1 GEX features show clear IC, upgrade to intraday. Also
  consider SpotGamma / Menthor Q subscription at that point ($500/mo vs our
  current $0).

### Cross-asset GEX expansion (V1.5)

- **Scope**: NDX → NQ, RUT → RTY, DJX → YM (chains already downloaded).
- **Why deferred**: V1 is single-contract ES; cross-asset GEX needs the
  multi-instrument production run.
- **Trigger**: When `build_options_panel.py` is run for instruments other than ES.
  Wiring is trivial (extend `TARGET_TO_OPTIONS_TICKER`).

---

## Deferred — feature-selection methodology (V1.5)

IC dashboard + ILP ρ<0.45 selection captures CORRELATION with the target but
not CAUSATION. Two features can be near-identically correlated with the label
yet have very different causal structures (common-cause vs lead-lag vs
collider-bias). Add a causation layer before final feature lock-in:

- **Lead-lag IC scan**: for each candidate feature, also score it at lags
  k = -3, -1, 0, +1, +3. Features with peak IC at *lagged* values
  (X(t-k) → label(t)) carry directional info. Features with peak IC at
  k=0 only may be coincidental / common-cause-driven.
- **Granger causality** for the top-IC features: does past X help predict
  Y beyond past Y alone? Standard implementation: pairwise F-test on a
  VAR(p) lag selection. Available in statsmodels.
- **Ablation studies** during model training: train with/without specific
  features, measure OOS delta. The "true alpha" features survive ablation.
- **Permutation feature importance** (post-training): randomize each
  feature in turn, measure prediction degradation. Better than tree-based
  feature importance because it captures actual prediction usage, not
  split frequency.
- **Stability across regimes**: causally-driven features should retain IC
  across different vol regimes (COVID 2020 vs calm 2023). Spurious-
  correlation features tend to break across regime shifts.

Trigger: after V1 model trains and we have OOS performance. If V1 ships and
works, the causation layer becomes a V1.5 robustness investment to make the
model more durable against regime change.

---

## Production / pipeline housekeeping

### Multi-instrument single-panel build

- **Scope**: Currently only ES has a `features/single/ES_2024.parquet`. To
  enable cross-sectional (Phase 4), need single panels for at least NQ + a few
  macro instruments (preferably all 30).
- **Trigger**: Once V1 ES single-contract model trains and we want to add
  cross-sectional features.
- **Work**: Re-run `build_futures_panel.sbatch` and `build_single_panel.sbatch`
  with `INSTR=NQ`, `INSTR=GC`, etc. (each is a 5-year array job).

### Bar-builder consolidation (`build_bars.py --mode`)

- **Status**: Intentionally deferred per `REFACTOR.md`. Per-bar-type scripts
  work, are independent, have different SLURM resource needs.
- **Trigger**: Skip until script-count proliferation causes real friction.

### Caching / `_BUILD_INFO.json` sidecars

- **Status**: Intentionally deferred per `REFACTOR.md`. Full rebuilds are
  ~5-10 min/year per stage so cost is bounded.
- **Trigger**: Add when iteration cadence on a stage exceeds ~5/day.

### Variant expansion grid

- **Scope**: PHASE1 §2 defines `{window: 3/10/30/60}`, `{lag: 1/3/10}`,
  `{depth: 1/3/10}`, transforms. Need a config-driven expander that takes our
  feature functions + grid → emits ~400-600 feature columns.
- **Status**: Currently we hand-wire window grids in
  `single_contract.attach_base_microstructure_features` (e.g.,
  `rv_windows=(20, 60, 120)`). Adequate for V1.
- **Trigger**: When the model wants ~600 features and hand-wiring is tedious.

### Dashboard pass + ILP feature selection

- **Scope**: `save_dashboard_v2.py` analog computing IC + t-stat per feature ×
  horizon; `stage2_ilp_selection.py` analog with ρ<0.45 constraint. Port from
  `tognn_us`.
- **Trigger**: After V1 panel builds for multiple years and we want to prune
  the ~427-col feature set down to a tighter signal-rich subset for modeling.

---

## Research ideas (parking lot)

### GEX as indicator-multiplier rather than standalone feature

- **Hypothesis**: GEX may have low standalone IC but materially modulate the
  predictive power of other features (e.g., flow becomes more / less
  trade-able based on dealer-positioning regime).
- **Test**: Train two LightGBM models — one with raw GEX, one with GEX-feature
  interactions (multiplicative cross-terms with top-IC microstructure features).
  Compare OOS Sharpe.
- **Trigger**: After V1 baseline model gives us a feature-importance ranking,
  pick the top 5 microstructure features and add GEX-multiplied variants.
