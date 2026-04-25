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

---

## V1 build chain (in flight or imminent — 2026-04-25)

V1 trades **4 equity-index futures** (ES, NQ, RTY, YM) but the directional
prediction model needs **macro context** from the full 30-contract universe:
DXY (USD strength) ↑ → equities up; gold + bonds bid + DXY ↓ → equities sell
(risk-off composite). Without intermarket signal, equity-index directional
prediction has no chance. So we build OHLCV bars for all 30 contracts;
deeper L2 enrichment only for the 4 we actually trade.

Four tracks chained sequentially via SLURM `afterok`:

- **Track 0 (VIX sync)**: re-fetch VIX curves from `s3://vix-futures/taq/{YYYY}/{date}/`.
  Original sync produced empty day-dirs (cause unidentified — possibly a
  thread-pool race that swallowed errors). Bucket layout confirmed correct;
  re-run via `scripts/sync_vix.py` + `sync_vix.sbatch`. ~30-60min, ~7.5GB.
- **Track A (Phase A bars — ALL 30 contracts)**: Wraps
  `src/data/bars_5sec.build_5sec_bars_core` per (instrument, day) → downsample
  to 15m. Output: `bars_phase_a/{INSTR}/15m/{INSTR}_{YYYYMMDD}_15m.parquet`
  with OHLCV + aggressor split + L1 close + spread sub-bar + CVD.
  Universe (30): ES NQ RTY YM (equity indices, V1 trading); 6A 6B 6C 6E 6J
  (FX, USD pairs); BZ CL HO NG RB (energy); GC HG PA PL SI (metals); SR3 TN
  ZB ZF ZN ZT (rates / curve); ZC ZL ZM ZS ZW (ags). ~10-14h wall, 30 array
  tasks (one per instrument).
- **Track B (Phase B L2 bars — only 4 trading contracts)**: Reads Phase A bars
  + depth events, calls `src/data/depth_snap.attach_book_snapshot` → adds 60
  L2 columns (bid_px_L1..L10, ask_px_L1..L10, sizes, orders) + `book_ts_close`.
  Output: `bars_phase_ab/{INSTR}/15m/...`. Restricted to ES/NQ/RTY/YM since
  cross-asset macro features only need OHLCV-level data on the other 26.
  ~16-20h.
- **Track C (GEX features)**: Runs `compute_daily_gex_profile` per ticker × year
  on SPX/SPY chain parquets. Output: `gex_features/{TICKER}_gex_profile_{YEAR}.parquet`.
  V2 wires NDX→NQ, RUT→RTY, DJX→YM (chains already downloaded). ~30-60min.

After all four complete, `src/features/panel.py` orchestrates the COMPLETE
feature library:
  - Per-trading-instrument single-instrument features (TC variants, overnight,
    smoothed)
  - Within-equity-index cross-sectional features (`bar_cross.py`, `l2_cross.py`):
    dispersion, breadth, leader-laggard, pairwise z-scores
  - Cross-asset MACRO features using all 30 OHLCV bars:
    - Synthetic DXY composite from 6E/6J/6B/6C (~92% DXY weight coverage)
    - Risk-on/off composite: gold (GC) + bonds (ZN, ZB) vs equities
    - Rolling correlations of trading instruments vs each macro family
    - Rates curve slope (2s5s, 5s10s, 10s30s from ZT/ZF/ZN/ZB)
    - Energy / commodity divergence flags
  - GEX features (SPX/SPY → ES/NQ-adjacent)
  - VX features (when sync completes)

Output: per-trading-instrument feature panel parquet with both single and
cross-asset features. Then IC dashboard + ILP feature selection.

## Post-VIX work (after Track 0 finishes)

- **VX bar builder** — VX raw TAQ → 15-min bars for VX1, VX2, VX3 (front-three
  monthly contracts). Mirrors `bars_5sec` + `bars_downsample` + roll handling.
  Roll convention: monthly (VX has VXF/G/H/J/K/M/N/Q/U/V/X/Z monthly cycle).
- **VX panel module**: take wide VX1/VX2/VX3 bars + apply `src/features/vx.py`
  to emit term-structure features (calendar spread, ratio, curvature,
  spread z-score, VX-mid z-score, vx-OFI-weighted).
- **Attach VX features to ES bars**: VX trades on a different exchange (CFE)
  with overlapping but not identical session hours. Use `engines.asof_strict_backward`
  to attach the most recent VX bar to each ES bar.

## Post-build cleanup (after V1 panel.py works end-to-end)

- **Delete bars_phase_a parquets** once Phase B is stable (Phase B is a strict
  superset). Keeping both wastes ~50% of disk.
- **Index by (instrument, year) parquet partitioning** for the IC/dashboard
  pipeline — per-day reads are slow at scale.

## Open V1.5 / V2 work that this build unlocks

- L1.x quote-derived features beyond `attach_book_snapshot`'s close-time
  state (bid/ask velocity, midprice diffusion). Need event-stream access.
- Cancel proxy at L2-L5 from depth event deltas (already TODO above).
- Cross-asset GEX: NDX/QQQ for NQ, RUT/IWM for RTY, DJX/DIA for YM
  (already downloaded; just wire into `attach_gex_features` per instrument).
- Tier 5b term-structure (CL/NG/ZN curves) for V3 multi-asset extension.
