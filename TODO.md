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

## VIX data — BLOCKED for V1 (2026-04-25 investigation)

VIX data is currently inaccessible. Investigation:

- **Original `s3://vix-futures/` bucket**: list works, but
  `HeadObject`/`GetObject` returns 403 Forbidden using the `plt-de-dev` AWS
  profile (despite `--request-payer requester`). Bucket-level permissions
  barrier on individual objects.
- **Mirror at `s3://plt-de-dev/lake/US/raw/Algoseek/s3/vix-futures/`**: list
  works; every VIX object has `StorageClass = GLACIER`. `GetObject` fails
  with `InvalidObjectState`. `aws s3 cp` skips with explicit message:
  "Object is of storage class GLACIER. Unable to perform download operations
  on GLACIER objects. You must restore the object."

To unblock VIX in the future:
- **Option 1**: issue `aws s3api restore-object` requests on the mirror,
  Bulk tier (5-12h per object). ~$0.0025/request × ~15,600 files (6 yrs ×
  260 days × ~10 expiries) = ~$40 in requests + temporary restore storage
  (~$2/GB·month). Total ~$50-100 for the full 6-year curve.
- **Option 2**: obtain IAM credentials with direct `GetObject` on
  `s3://vix-futures/` (verify with the data-platform team whether a profile
  with these permissions exists). Bypasses the Glacier issue if the source
  bucket files are in Standard storage.

V1 ships without VX features. The panel.py architecture already supports
slotting VX features in once the data is unblocked — only the VX bar
builder + VX panel module need to be added (see "Post-VIX work" below).

## Post-VIX work (when VIX data is unblocked)

- **VX bar builder** — VX raw TAQ → 15-min bars for VX1, VX2, VX3 (front-three
  monthly contracts). Mirrors `bars_5sec` + `bars_downsample` + roll handling.
  Roll convention: monthly (VX has VXF/G/H/J/K/M/N/Q/U/V/X/Z monthly cycle).
- **VX panel module**: take wide VX1/VX2/VX3 bars + apply `src/features/vx.py`
  to emit term-structure features (calendar spread, ratio, curvature,
  spread z-score, VX-mid z-score, vx-OFI-weighted).
- **Attach VX features to ES bars**: VX trades on a different exchange (CFE)
  with overlapping but not identical session hours. Use
  `engines.asof_strict_backward` to attach the most recent VX bar to each ES bar.

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

---

## Panel.py architecture (decided 2026-04-25)

Locked decisions for the cross-asset-aware feature builder we'll build after
the bar chain finishes:

- **Full TS features on all 30 contracts** (not just trading 4). Each contract
  gets TC z-score, MAD z-score, rolling quantile rank for each base value
  (return, abs_return, log_volume, OFI, CVD_change, spread_z, vol_surprise).
  The 26 non-trading contracts contribute through their TS features as inputs
  to the trading-contract panels (e.g., GC's TC z-score of OFI as a feature
  for predicting ES). Storage / compute cheap; ILP does the pruning.
- **Cross-sectional Gauss-Rank features** (NEW) — every base value gets ranked
  two ways at each ts:
  - Across the full 30-contract universe (gives "vs market" position)
  - Within asset class (equity-indices / rates / FX / energy / metals / ags)
    so equity-index relative ranks aren't diluted by gold/oil context
  Gauss-Rank: rank → quantile → inverse normal CDF → bounded standard-normal
  values; outlier-tame, symmetric, ideal for tree models.
- **NO threshold-engineered features.** Pre-engineered binary "significant
  move" flags or excess-magnitude features lock in arbitrary thresholds the
  model can't override; they throw away smooth distributional information.
  LightGBM does its own thresholding via tree splits — give it smooth
  continuous distributional features and let it find the splits.
- **L2 deep features only on 4 trading contracts** (Phase B output).
- **Composite features** (continuous, not binary): synthetic DXY (weighted
  6E/6J/6B/6C), rates curve slopes (2s5s/5s10s/10s30s), risk-on/off
  composite as a continuous z-weighted score, cross-asset rolling correlations.

---

## Feature-selection methodology — beyond IC (V1.5)

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

## V1 build chain — still in flight (job IDs 6915995 → 6915996 → 6915997 → 6915998)

- 6915995 (VIX sync, array 0-5 by year) — pending, no deps
- 6915996 (Phase A bars, array 0-29 across 30 contracts) — held on afterok:VIX
- 6915997 (Phase B L2, array 0-3 ES/NQ/RTY/YM only) — held on afterok:A
- 6915998 (GEX features, single node) — held on afterok:B

While bars build, develop `src/features/cross_asset_macro.py` (TC + MAD +
quantile rank + Gauss-Rank + composites) and `src/features/panel.py`
(orchestrator) so they're ready when bars finish.


- Also I have a suspicion that the  GEX features might not be useful on its own rather we should use it as an indicator variable ; perhaps multiply it with correct features to make it more useful. 
