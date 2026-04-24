# TODO — Deferred items

Running list of features / modules consciously deferred from V1 build. Each entry
notes the tier, why it's deferred, and the trigger for enabling.

V1 focus is **microstructure + pattern + regime features**. Macro and regression-
based items can plug in later without pipeline rework.

---

## Deferred from V1 — plug in later

### Macro calendar + event features (T6.05, T6.06, T6.08)

- **Scope**: Distance to next macro release (minutes), ±15-min event-window flag
  per event type (FOMC/NFP/CPI/EIA/ISM), post-FOMC 2-hr window flag.
- **Why deferred**: V1 is a microstructure-driven strategy. Macro events add
  exogenous regime information but aren't core to the 15-min signal.
- **Trigger to enable**: When we observe V1 OOS Sharpe degradation around macro
  release windows (visible in PnL attribution by hour-of-day). Macro calendar is
  the first place to look for the fix.
- **Work required**: macro-calendar CSV (scrape BLS/Fed/EIA), `src/features/events.py`
  with `distance_to_next_release`, `event_window_flag(kind, width_min)`.

### L2 cancel proxy levels 2-5 (T1.43 extension)

- **Scope**: Currently `bars_cancel.py` emits only `net_bid_decrement_no_trade_L1`
  and `_ask_L1`. FEATURES.md schema calls for L1-L5.
- **Why deferred**: L2-L5 attribution requires handling level-promotion/demotion
  when inner prices change (cancel at L2 today was L3 yesterday).
- **Trigger**: If ILP survives V1 and selects the L1 cancel proxy as material,
  extend to L2-L5.

### bar-level event aggregates — Phase E bar schema (T1.24, T1.25, T1.28, T1.29)

- **Scope**: `quote_to_trade_ratio`, `quote_movement_directionality`,
  `side_conditioned_liquidity_shift`, `liquidity_migration`.
- **Why deferred**: Need per-quote (not per-bar-close) directional accounting
  kept during bar-build. Our current 5-sec builder emits summary stats, not
  per-event change logs.
- **Work required**: Add `quote_update_count`, `quote_up_count`, `quote_down_count`
  to 5-sec bar schema (Phase E). Then these features become simple ratios.

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

---

## Implementation hygiene tasks

- **Per-bar application of engines**: `engines.py` has VPIN/Hawkes/CVD/etc. as
  primitives. Need a thin per-bar wrapper (one-liner) that emits them as named
  feature columns once we have the 5-sec bar frame ready.
- **Variant expansion grid**: PHASE1 §2 defines `{window: 3/10/30/60}`,
  `{lag: 1/3/10}`, `{depth: 1/3/10}`, transforms. Need a config-driven expander
  that takes our feature functions + grid → emits ~400-600 feature columns.
- **Dashboard pass**: `save_dashboard_v2.py` analog that computes IC + t-stat
  per feature × horizon. Port from `tognn_us`.
- **ILP selection**: `stage2_ilp_selection.py` analog with ρ<0.45 constraint.
  Port from `tognn_us`.

---

## Housekeeping / follow-ups from prior commits

- Algoseek phase 2 (job `6913007`) monitor completion; re-run feature-sync if
  any new files landed that need reprocessing.
- SPX chain job `6913304` pulls NDX/QQQ/RUT/IWM/DJX/DIA on top of already-done
  SPX/SPY. Confirm all 8 ticker-years complete.
- VX futures (from Algoseek) — we have it but haven't built the VX-specific
  5-sec bar pipeline yet. Needed for T5.01-T5.07.
