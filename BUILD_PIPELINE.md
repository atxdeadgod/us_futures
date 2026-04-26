# Feature-Panel Build Pipeline

**Purpose**: one place to look up *how to build the full V1 feature panel from
scratch, end-to-end, for all 4 trading instruments and the 26 macro-feed
contracts*.
**Scope**: V1 = ES/NQ/RTY/YM with the **full feature set** (Phase A + Phase A+B
+ 5-sec sub-bar engines + Phase E + VX + GEX) plus 26 macro-feed contracts
with bar-derived features only — together feeding cross-sectional features
back onto each trading target.
**Companion docs**: `FEATURES.md` (the catalog of what each feature means),
`LABELING_V1_SUMMARY.md` (the triple-barrier label spec), and
`REFACTOR.md` (the 4-phase architecture this pipeline follows).

---

## TL;DR — rebuild end-to-end (master sequence)

Assuming raw Algoseek mirror is on bigred at
`/N/project/ksb-finance-backtesting/data/algoseek_futures/{taq,depth,vix}/...`
and the SPX/NDX/RUT/DJX/SPY options chains are in
`/N/project/.../spx_options_chain/`. Each `INSTR=X` env override targets one
instrument; the sbatches are array=0-4 over years 2020-2024 unless noted.

> **Cluster note**: every sbatch is portable across bigred and quartz (same
> account, shared filesystems, absolute python paths). Submit on whichever
> cluster has capacity. Cross-cluster SLURM dependencies don't work, so keep
> a chain (futures → options → single) on one cluster.

```bash
########################################################################
# Phase 1 — Bar builds (raw Algoseek → per-day parquets)
# All independent within Phase 1; can run in parallel.
########################################################################

# (1a) Phase A bars: 30 instruments × 15-min OHLCV+aggressor+L1+CVD+spread sub-bar
sbatch scripts/build_phase_a_bars.sbatch                                       # array=0-29, ~6h

# (1b) Phase A+B bars: 4 trading instruments + L1-L10 book snapshot
INSTR=ES  sbatch scripts/build_l2_bars.sbatch                                   # array=0-4, ~45min
INSTR=NQ  sbatch scripts/build_l2_bars.sbatch
INSTR=RTY sbatch scripts/build_l2_bars.sbatch
INSTR=YM  sbatch scripts/build_l2_bars.sbatch
# (Or single sbatch if its array covers all 4 — check the .sbatch header.)

# (1c) VX bars: VX1/VX2/VX3 (front-3 monthly VIX futures)
sbatch scripts/build_vx_bars.sbatch                                            # single task, ~75min

# (1d) 5-sec bars: ALL 4 trading instruments (needed for VPIN + Hawkes)
INSTR=ES  sbatch scripts/build_5sec_bars.sbatch                                 # ~60min each
INSTR=NQ  sbatch scripts/build_5sec_bars.sbatch
INSTR=RTY sbatch scripts/build_5sec_bars.sbatch
INSTR=YM  sbatch scripts/build_5sec_bars.sbatch

# (1e) Phase E bars: ALL 4 trading instruments (exec quality + cancel proxy + quote dir)
INSTR=ES  sbatch scripts/build_phase_e_bars.sbatch                              # ~3-4h each
INSTR=NQ  sbatch scripts/build_phase_e_bars.sbatch
INSTR=RTY sbatch scripts/build_phase_e_bars.sbatch
INSTR=YM  sbatch scripts/build_phase_e_bars.sbatch

########################################################################
# Phase 2a — GEX daily profile (options chain → per-day GEX parquet)
# Run after SPX/NDX/RUT/DJX chains are downloaded.
########################################################################

TICKERS=SPX,SPY,NDX,RUT,DJX sbatch scripts/build_gex_features.sbatch           # ~15-30min total

########################################################################
# Phase 2b — Per-source feature panels (per instrument, per year)
# Submit chained: futures must complete before options/single for that instrument.
########################################################################

# Dispatch script (run from any cluster); captures job IDs for chaining
declare -A FUT_JOBS OPT_JOBS
ALL_INSTRUMENTS=(ES NQ RTY YM 6A 6B 6C 6E 6J BZ CL HO NG RB \
                 GC HG PA PL SI SR3 TN ZB ZF ZN ZT ZC ZL ZM ZS ZW)
TRADING=(ES NQ RTY YM)

# 30 futures panel jobs (one sbatch per instrument, array=0-4 over years)
for INSTR in "${ALL_INSTRUMENTS[@]}"; do
    JID=$(INSTR=$INSTR sbatch --parsable scripts/build_futures_panel.sbatch)
    FUT_JOBS[$INSTR]=$JID
done

# 4 options panel jobs (only for trading 4) — chained on each instrument's futures
for INSTR in "${TRADING[@]}"; do
    JID=$(INSTR=$INSTR sbatch --parsable \
        --dependency=afterok:${FUT_JOBS[$INSTR]} \
        scripts/build_options_panel.sbatch)
    OPT_JOBS[$INSTR]=$JID
done

########################################################################
# Phase 3 — Single-contract merged panel + V1 labels (per instrument)
# Joins futures + options on ts, applies V1 triple-barrier labels (trading 4 only).
########################################################################

for INSTR in "${ALL_INSTRUMENTS[@]}"; do
    DEPS="afterok:${FUT_JOBS[$INSTR]}"
    if [[ -n "${OPT_JOBS[$INSTR]:-}" ]]; then
        DEPS="${DEPS}:${OPT_JOBS[$INSTR]}"
    fi
    INSTR=$INSTR sbatch --dependency=$DEPS scripts/build_single_panel.sbatch
done

########################################################################
# Phase 4 — Cross-sectional panel (only after ALL 30 single panels exist)
# Per-target: joins all 30 single panels on ts → CS ranks + composites + interactions.
########################################################################

sbatch scripts/build_cross_panel.sbatch    # array=0-19, 4 targets × 5 years, ~30min
```

**Outputs**:
- `/N/.../bars_phase_a/{INSTR}/15m/...`           (Phase 1a)
- `/N/.../bars_phase_ab/{INSTR}/15m/...`          (Phase 1b)
- `/N/.../bars_phase_a/VX{1,2,3}/15m/...`         (Phase 1c)
- `/N/.../bars_5sec/{INSTR}/5s/...`               (Phase 1d)
- `/N/.../bars_phase_e/{INSTR}/15m/...`           (Phase 1e)
- `/N/.../gex_features/{TICKER}_gex_profile_{YEAR}.parquet`  (Phase 2a)
- `/N/.../features/futures/{INSTR}_{YEAR}.parquet`  (Phase 2b)
- `/N/.../features/options/{INSTR}_{YEAR}.parquet`  (Phase 2b — trading 4 only)
- `/N/.../features/single/{INSTR}_{YEAR}.parquet`   (Phase 3)
- `/N/.../features/cross/{TARGET}_{YEAR}.parquet`   (Phase 4)

**Wall time estimate (cold start, with parallel cluster capacity)**:
- Phase 1: ~6h longest pole (Phase A bars 30-instrument array)
- Phase 2a: ~30 min
- Phase 2b: ~1-2h (ES is the long pole because of Hawkes)
- Phase 3: ~30 min
- Phase 4: ~30 min
- **Total ~8-10h** for a from-scratch full rebuild.

**Wall time for INCREMENTAL adds (e.g., extending sub-bar engines + Phase E to NQ/RTY/YM after they were ES-only)**:
- 5-sec bars × 3 instruments parallel: ~60 min
- Phase E × 3 instruments parallel: ~3-4h
- Re-run futures + single for those 3 instruments: ~30 min
- Total ~5h additional.

**Pre-flight smoke for a single instrument**: see "Smoke tests" below — always run this before submitting a multi-year array job.

---

## Stage map

```
PHASE 1: BARS (raw Algoseek → per-day parquets)

   raw Algoseek (taq/depth/vix)
        │
        ├──→ Phase A bars       (30 instruments × 15m)
        ├──→ Phase A+B bars     (4 trading instr × 15m + L1-L10 book)
        ├──→ VX1/VX2/VX3 bars   (front-3 monthly VX × 15m)
        └──→ 5-sec bars         (ES, for VPIN/Hawkes — V1.5)

PHASE 2: PER-SOURCE FEATURE PANELS
                                    │
   bars ─→ build_futures_panel.py ──┤  ┌──→ features/futures/{INSTR}_{YEAR}.parquet
                                    │  │     (per-instrument bar features + VX)
                                    │  │
   SPX/NDX options chain            │  │
        ├─→ GEX daily profile       │  │
        └─→ build_options_panel.py ─┤──┤──→ features/options/{INSTR}_{YEAR}.parquet
                                    │  │     (per-instrument GEX features)
                                    │  │
PHASE 3: SINGLE-CONTRACT MERGE      │  │
                                    │  │
        build_single_panel.py ←─────┼──┘
                 │
                 ├─→ join on ts (futures + options)
                 ├─→ apply V1 triple-barrier labels
                 └─→ features/single/{INSTR}_{YEAR}.parquet

PHASE 4: CROSS-SECTIONAL
                 │
        build_cross_panel.py ──→ joins all 30 single panels on ts
                 │             ├─→ Gauss-Rank universe + per-class
                 │             ├─→ synthetic DXY, rates curve, risk-on/off
                 │             └─→ rolling cross-asset correlations
                 │
                 └─→ features/cross/{TARGET}_{YEAR}.parquet
```

---

## Stage details

Each stage is **idempotent** (skip-if-exists) and **independent** within its
input dependencies. SLURM array tasks are independent days × instruments.

### S1. Phase A bars (30 instruments)
- **Reads**:  `algoseek_futures/taq/{ROOT}/{YYYY}/{YYYYMMDD}/{EXPIRY}.csv.gz`
- **Writes**: `bars_phase_a/{ROOT}/15m/{ROOT}_{YYYYMMDD}_15m.parquet`
- **Schema**: ts + OHLCV + aggressor split + L1 close + spread sub-bar + CVD dual-reset (29 cols).
- **Script**: `scripts/build_phase_a_bars.py` (CLI), `build_phase_a_bars.sbatch` (array=0-29 over 30 instruments).
- **Status**: ✅ done — files exist for all 30 instruments × 1097-1299 days.

### S2. Phase A+B bars (4 trading instruments — adds L1-L10 book snapshot)
- **Reads**: Phase A bars + `algoseek_futures/depth/{ROOT}/.../{EXPIRY}.csv.gz`
- **Writes**: `bars_phase_ab/{ROOT}/15m/{ROOT}_{YYYYMMDD}_15m.parquet`
- **Adds 60 cols**: bid_px/sz/ord_L{1..10}, ask_px/sz/ord_L{1..10}.
- **Script**: `scripts/build_l2_bars.py`, `build_l2_bars.sbatch` (array=0-3 over ES/NQ/RTY/YM).
- **Status**: ✅ done — 1295 days for each of ES/NQ/RTY/YM.

### S3. VX1/VX2/VX3 bars (3 slots — front-three monthly VIX futures)
- **Reads**: `algoseek_futures/vix/{YYYY}/{YYYYMMDD}/VX*.csv.gz`
- **Writes**: `bars_phase_a/VX{i}/15m/VX{i}_{YYYYMMDD}_15m.parquet` for i ∈ {1,2,3}
- **Roll**: front-N selected by parsed calendar order (`src/data/roll.parse_vx_expiry`).
- **Schema**: same Phase A schema (29 cols). Aggressor split is zero (VIX TAQ has only `TRADE`, no aggressor classification) — VX features use mid/spread only.
- **Script**: `scripts/build_vx_bars.py`, `build_vx_bars.sbatch` (single task, 3 slots × ~1300 days).
- **Status**: 🟡 in progress (job 6921146).

### S4. 5-sec bars (4 trading instruments — needed for VPIN + Hawkes engines)
- **Reads**: same as Phase A.
- **Writes**: `bars_5sec/{INSTR}/5s/{INSTR}_{YYYYMMDD}_5s.parquet` (~15K rows/day × 29 cols) for ES, NQ, RTY, YM.
- **Used by**: VPIN volume buckets (`engines.vpin_volume_buckets`), Hawkes intensity recursion (`engines.hawkes_intensity_recursive`); wired into `build_futures_panel.py` via `sub_bar_engines.attach_sub_bar_engine_features`.
- **Script**: `scripts/build_5sec_bars.py`, `build_5sec_bars.sbatch` (per-instrument single-task; override `INSTR` via env).
- **Submission**: 4 separate sbatch submissions, one per trading instrument; runs in parallel.
- **Status**: ES ✅ (1299 days). NQ/RTY/YM ⏳ pending (to be built post current multi-instrument single-panel run).

### S4b. Phase E bars (4 trading instruments — execution-quality + cancel-proxy + quote-event + quote-direction aggregates)
- **Reads**: raw TAQ + depth (Algoseek).
- **Writes**: `bars_phase_e/{INSTR}/15m/{INSTR}_{YYYYMMDD}_15m.parquet` (24 cols) for ES, NQ, RTY, YM.
- **Cols**:
  - Effective spread: eff_spread_{sum,weight,count,buy_sum,buy_weight,sell_sum,sell_weight} (T1.35-T1.37 prereqs)
  - Large trades: n_large_trades, large_trade_volume (T1.23 prereqs)
  - Hidden absorption: hidden_absorption_{volume,trades} (T1.47/T7.12 prereqs)
  - Cancel proxy: net_{bid,ask}_decrement_no_trade_L1 (T1.43 prereqs)
  - **Side-conditioned shifts (T1.28 prereqs)**: bid_sz_L1_delta_signed, ask_sz_L1_delta_signed, hit_bid_vol, lift_ask_vol
  - Quote count: quote_update_count (T1.24 prereq)
  - **Quote direction (T1.25 prereqs)**: bid_up_count, bid_down_count, ask_up_count, ask_down_count
- **Used by**: `single_contract.attach_phase_e_features` to derive: vwap_eff_spread + asymmetry (T1.35-T1.37), large_trade_volume_share (T1.23), hidden_absorption_ratio (T1.47/T7.12), cancel_to_trade_ratio (T1.43), quote_to_trade_ratio (T1.24), **quote_movement_directionality (T1.25)**, **side_cond_ask_resilience_buy / side_cond_bid_resilience_sell (T1.28)**.
- **Script**: `scripts/build_phase_e_bars.py`, `build_phase_e_bars.sbatch` (per-instrument single-task; override `INSTR` via env).
- **Submission**: 4 separate sbatch submissions, one per trading instrument; runs in parallel.
- **Status**: ES ✅ (1287 days; 8 days lacked upstream Algoseek depth files). NQ/RTY/YM ⏳ pending.

  Note: T1.29 liquidity_migration is NOT in Phase E — it's derived directly from Phase A+B's L1-L5 cols via bar-to-bar deltas in `attach_l2_deep_features`.

### S5. GEX daily profile (SPX, SPY, NDX, RUT, DJX)
- **Reads**: `spx_options_chain/{TICKER}_{YEAR}.parquet` (WRDS OptionMetrics).
- **Writes**: `gex_features/{TICKER}_gex_profile_{YEAR}.parquet`.
- **Cols**: total_gex, gex_sign, zero_gamma_strike, max_call_oi_strike, max_put_oi_strike, gex_0dte_share, gex_0dte_only, gex_without_0dte.
- **Script**: `scripts/build_gex_features.py`, `build_gex_features.sbatch` (env overrides: `TICKERS`, `START_YEAR`, `END_YEAR`).
- **Mapping** (`TARGET_TO_OPTIONS_TICKER` in `build_options_panel.py`):
  - ES → SPX
  - NQ → NDX
  - RTY → RUT
  - YM → DJX
- **Status**: ✅ all 5 tickers × 2020-2024 built (25 profiles).

### S6. Futures feature panel (Phase 2b — bar-derived per-instrument features)
- **Reads**: Phase A+B bars (or Phase A with `--no-l2-deep`) + VX1/VX2/VX3 bars (optional) + 5-sec bars (optional, for VPIN/Hawkes).
- **Writes**: `features/futures/{INSTR}_{YEAR}.parquet`.
- **Script**: `scripts/build_futures_panel.py`, `build_futures_panel.sbatch` (array=0-4, one year per task; override INSTR via env).
- **Pipeline** (orchestrated by `src/features/single_contract.build_per_instrument_features` + `attach_l2_deep_features` + `external_sources.attach_vx_features` + `sub_bar_engines.attach_sub_bar_engine_features`):

  ```
  load Phase A+B bars (year)             →  90 cols
  ├── attach_base_microstructure_features    +44
  ├── attach_session_flags + cyclic + overnight  +11
  ├── attach_pattern_features                +13
  ├── attach_engine_features                 +4
  ├── attach_l2_deep_features (depth=10)     +50
  ├── attach_ts_normalizations               +136
  ├── attach_ema_smoothed                    +16
  ├── attach_vx_features                     +20
  └── attach_sub_bar_engine_features         +6   (VPIN + Hawkes from 5-sec bars)
                                             ─────
                                             ~382 cols
  ```
- **Sub-bar engines**: requires 5-sec bars at `--bars-5s-root`. Disable with
  `--no-sub-bar-engines`. Tunable: `--vpin-bucket-size`, `--hawkes-hl-fast`, `--hawkes-hl-slow`.

### S7. Options feature panel (Phase 2b — GEX-derived per-target features)
- **Reads**: GEX daily profile parquet (`SPX_gex_profile_{YEAR}.parquet`) + target's bars (just ts + close).
- **Writes**: `features/options/{INSTR}_{YEAR}.parquet`.
- **Script**: `scripts/build_options_panel.py`, `build_options_panel.sbatch`.
- Output schema is just `ts` + GEX columns (≈15 cols) — the close column is dropped to avoid join collision in Phase 3.
- Only emits a panel for instruments with a configured options-chain mapping (`TARGET_TO_OPTIONS_TICKER`); for V1 that's just `ES → SPX`.

### S8. Single-contract merged panel (Phase 3)
- **Reads**: `features/futures/{INSTR}_{YEAR}.parquet` + `features/options/{INSTR}_{YEAR}.parquet` (optional).
- **Writes**: `features/single/{INSTR}_{YEAR}.parquet`.
- **Script**: `scripts/build_single_panel.py`, `build_single_panel.sbatch`.
- Left-joins futures + options on ts (futures is authoritative for duplicates), applies V1 triple-barrier labels, drops warmup/halt-truncated rows.
- Final 2024 ES panel: ~9,970 valid labeled rows × 396 cols.

### S9. Cross-sectional panel (Phase 4)
- **Reads**: `features/single/{INSTR}_{YEAR}.parquet` for all 30 instruments.
- **Writes**: `features/cross/{TARGET}_{YEAR}.parquet`.
- **Script**: `scripts/build_cross_panel.py`, `build_cross_panel.sbatch` (array=0-19 over 4 targets × 5 years).
- **Status**: ✅ done — 20 cross panels on disk (4 × 5).

**Pipeline inside the script**:

```
1. Probe parquet schemas via pq.read_schema (cheap metadata) for all 30 instruments
   → identify which CS_VALUE_COLS each panel actually has

2. Scan-load only ts + needed value cols for each panel via pl.scan_parquet().select().collect()
   → ~30× memory reduction vs read-then-narrow

3. Per-instrument dedupe on ts (.unique(subset=["ts"], keep="first"))
   → CRITICAL: without dedupe, sequential left-joins explode cartesian-style
     (24K rows → 30M rows after 16 joins, OOM at 30M)

4. build_wide_cross_asset_frame:
   master ts grid (sorted union of all per-instrument ts) +
   30 sequential left-joins onto master (one per instrument)
   → ~24K rows × 421 cols (14 BASE × 30 instr + ts)

5. attach_cross_sectional_ranks:
   For each base value × each scope {universe, asset-class}:
     attach_gauss_rank_cs → rank → quantile → inverse-CDF (bounded standard-normal)
   → +840 cols (14 × 30 × 2)

6. attach_cross_asset_composites:
   synthetic_dxy_logret, rates curve spreads (2s5s/5s10s/2s10s/10s30s/butterfly),
   risk-on/off composite, cross-asset rolling corrs
   → +22 cols

7. Filter wide frame to drop other-target {target}_* prefix cols and
   other-target rolling-corr cols (we keep only the running target's corrs)
   → -26 cols

8. Left-join filtered wide frame onto target's single panel on ts
   → +445 cols (target's full single panel from Phase 3, ~446 cols)

9. attach_regime_interactions:
   12-13 multiplicative interactions (regime × direction). Naming:
   ix_<regime>_x_<direction>. Pre-computed so ILP can rank them on
   standalone |IC| — tree models can't easily learn multiplicative
   interactions through splits.
   → +16 cols

→ ~1,718 cols × ~9-10K labeled rows per (target, year).
```

**SBATCH env vars** (in `build_cross_panel.sbatch`):
- `POLARS_MAX_THREADS=1` — bound polars thread pool to avoid per-thread buffer multiplication
- `POLARS_STREAMING_CHUNK_SIZE=10000` — control streaming engine memory usage
- `--mem=128G` — sufficient with the dedupe fix; was insufficient before

**Common gotcha**: per-instrument single panels can have duplicate ts rows
(from the futures+options join when options has multi-row-per-bar). The dedupe
step is essential — without it, the wide-join is a 30-way cartesian explosion.

---

## Smoke tests (run these before any SLURM job)

```bash
# Phase A — single instrument × single day
ssh bigred 'cd ~/us_futures && python -u scripts/build_phase_a_bars.py \
    --instrument ES --start 2024-03-01 --end 2024-03-01 \
    --out-root /tmp/phase_a_smoke'

# VX bars — one day, three slots
ssh bigred 'cd ~/us_futures && python -u scripts/build_vx_bars.py \
    --start 2024-03-01 --end 2024-03-01 \
    --out-root /tmp/vx_smoke \
    --algoseek-root /N/project/ksb-finance-backtesting/data/algoseek_futures'

# 5-sec bars — one day
ssh bigred 'cd ~/us_futures && python -u scripts/build_5sec_bars.py \
    --instrument ES --start 2024-03-01 --end 2024-03-01 \
    --out-root /tmp/bars_5s_smoke'

# Full ES single panel chain — one year
ssh bigred 'cd ~/us_futures && python -u scripts/build_futures_panel.py --instrument ES --year 2024 --out /tmp/es_smoke'
ssh bigred 'cd ~/us_futures && python -u scripts/build_options_panel.py --instrument ES --year 2024 --out /tmp/es_smoke'
ssh bigred 'cd ~/us_futures && python -u scripts/build_single_panel.py  --instrument ES --year 2024 --out /tmp/es_smoke'
# Cross panel needs single panels for all 30 instruments first; smoke after multi-instrument run
```

Local unit tests (always run before pushing changes):

```bash
python -m pytest tests/ -q
```

---

## V1 feature inventory

Two distinct panels. Trading instruments (ES/NQ/RTY/YM) get the FULL feature
set; macro-feed contracts get a leaner subset. Cross-sectional adds another
~850 features per target.

### Per-instrument single-panel size

| Panel | Size | Used for |
|---|---:|---|
| Trading 4 (ES/NQ/RTY/YM) full | **~446 cols × ~9-10K labeled rows** | Direct modeling target |
| Trading 4 partial (only ES has Phase E + 5-sec engines as of 2026-04-25 evening) | ~416 / ~446 | Mid-build state |
| Macro-feed 26 (no Phase A+B / no Phase E / no 5-sec) | **~280 cols × ~24K unlabeled rows** | Cross-sectional inputs only |

### Cross-sectional panel size

After `build_cross_panel.py` runs (per target). Validated on ES 2021 build
(job 8895087_4):

| Layer | Cols added | Cumulative |
|---|---:|---:|
| Wide frame (14 BASE × 30 instruments × 2 ranks + ts) | 421 | 421 |
| + CS Gauss-Rank universe + per-asset-class | +840 | 1,261 |
| + Cross-asset composites (DXY, rates curve, rolling corrs) | +22 | 1,283 |
| − Filter to target (drop other-target corrs) | −26 | 1,257 |
| + Merged with target's full single panel | +445 | 1,702 |
| + Regime × direction interactions | +16 | **1,718** |

**Total per cross panel: ~1,718 cols × ~9-10K labeled rows.**

Pre-ILP. ILP feature-selection then prunes to ~50-80 modeling features at
ρ<0.45 pairwise correlation.

### Cross-sectional feature inventory by group

| Group | Cols | Naming convention |
|---|---:|---|
| Single panel passthrough | 446 | (everything from Phase 3, see "Per-instrument single panel" above) |
| CS Gauss-Rank — universe scope (vs all 30) | 420 | `cs_universe_<INSTR>_<base_value>` |
| CS Gauss-Rank — within asset class scope | 420 | `cs_class_<CLASS>_<INSTR>_<base_value>` |
| Synthetic DXY | 1 | `synthetic_dxy_logret` |
| Rates curve spreads | 5 | `curve_2s5s`, `curve_5s10s`, `curve_2s10s`, `curve_10s30s`, `butterfly_2s5s10s` |
| Cross-asset rolling correlations (per-target) | ~4 | `corr_{TARGET}_vs_{gold,oil,ZN,DXY}_w60` |
| Other composites (risk-on/off etc.) | ~12 | varies |
| Regime × direction interactions | 16 | `ix_<regime>_x_<direction>` |

### Column-group breakdown (per labeled bar)

| Group | Count | Source | Tier coverage |
|---|---:|---|---|
| identity / labels | 10 | bar build + V1 labeling | — |
| base micro | 133 | `single_contract.attach_base_microstructure_features` + T1.29 + ema-smoothed | T1.30 (Amihud), T1.39 (implied vol share + skew), T1.21/22 (OFI/aggressor), T2.01-T2.29 (returns/vol/jumps/...), T1.29 (liquidity migration L1↔L2/L3 — bid+ask), session/cyclic/overnight |
| TC z-score normalizations | 72 | `attach_ts_normalizations` | 36 BASE_VALUE_COLS × 2 lookback days {30, 60} |
| MAD z-score normalizations | 72 | same | robust-to-outliers variant |
| L2 deep (composite) | 10 | `attach_l2_deep_features` | T1.10 dw_imbalance, T1.11 cum_imbalance, T1.14 depth_weighted_spread, T1.15 liq_adj_spread, T1.16 spread_acceleration, T1.20 HHI ×2, T1.31 deep_ofi ×2, T1.17 spread_z |
| L2 per-level (k=1..10/5) | 40 | same | T1.09 vol_imbalance × 10, T1.13 basic_spread × 10, T1.31 ofi_at × 10, T1.18/19 order_count/size_imbalance × 5 |
| patterns (T7.*) | 13 | `attach_pattern_features` | T7.01 absorption, T7.04-T7.06 breakouts/reversals/post-reversal flow, T7.07 spike_fade, T7.08 imbalance_persistence, T7.09 CVD_price_divergence, T7.10 range_compression |
| engine | 4 | `attach_engine_features` | §8.A fracdiff(d=0.4) on log(close), §8.F round_pin distance N={10,25,50} |
| sub-bar engines (5s) | 6 | `sub_bar_engines.attach_sub_bar_engine_features` | T1.40-T1.42 VPIN family, T1.44-T1.46 Hawkes family |
| Phase E (exec/cancel/quote-direction) | 11 | `attach_phase_e_features` (gated on `--bars-phase-e-root` data) | T1.23 large_trade_share, T1.24 quote_to_trade, T1.25 quote_movement_directionality, T1.28 side_cond_resilience ×2, T1.35-T1.37 vwap_eff_spread + asym, T1.43 cancel_to_trade, T1.47/T7.12 hidden_absorption_ratio |
| EMA-smoothed | 16 | `smoothed.py` | 8 base × spans {10, 30} causal EMA |
| session / cyclic / overnight | 11 | `tc_features` + `overnight` | hour_et + 4 session flags + sin/cos minute + 4 overnight aggregates |
| VX (vol-regime) | 20 | `external_sources.attach_vx_features` | T5.01-T5.07: vx1/2/3 mid + spread + zscore + calendar spread/ratio + term curvature |
| GEX (options) | 9 | `external_sources.attach_gex_for_target` | T5 dealer-gamma family: total_gex, gex_sign, distance_to_{zero_gamma_flip, max_call_oi, max_put_oi}, ... |

The exact column list is printed at the end of every `build_single_panel.py` run
under `[col-summary]`.

### Tier-by-tier coverage

| Tier | Coverage | Status |
|---|---|---|
| T1.01–T1.20 (L1/L2 from Phase A+B) | All implemented | ✅ |
| T1.21–T1.22 (bar-level OFI) | All implemented | ✅ |
| T1.23 LargeTradeVolumeShare | Phase E | ✅ |
| T1.24 QuoteToTradeRatio | Phase E | ✅ |
| T1.25 QuoteMovementDirectionality | Phase E + F | ✅ |
| T1.26–T1.27 (depth ratios + side-weighted spread) | Implemented | ✅ |
| T1.28 SideConditionedLiquidityShift | Phase E + F | ✅ |
| T1.29 LiquidityMigration | from L1-L5 deltas | ✅ |
| T1.30 AmihudIlliquidity | base micro | ✅ |
| T1.31–T1.34 (DeepOFI, depth aggregates) | L2 deep | ✅ |
| T1.35–T1.37 (effective spread family) | Phase E | ✅ |
| T1.38 BookResilience | needs per-event +N-sec lookforward | ❌ V1.5 |
| T1.39 ImpliedVolumeShare | base micro | ✅ |
| T1.40–T1.42 VPIN family | sub-bar engines | ✅ |
| T1.43 CancelToTradeRatio (L1 only) | Phase E | ✅ |
| T1.44–T1.46 Hawkes family | sub-bar engines | ✅ |
| T1.47 HiddenAbsorption | Phase E | ✅ |
| T2.01–T2.29 (returns/vol family) | base micro + EMA | ✅ |
| T3.* (cross-asset lead-lag) | wired in `cross_sectional.py`, not yet run | 🟡 V1.5 (orchestrator ready) |
| T5 (VX + GEX) | external_sources | ✅ |
| T6.* (calendar/temporal) | tc_features + overnight | ✅ partial (T6.05/06/08 macro events deferred) |
| T7.01–T7.12 (patterns) | patterns + Phase E hidden | ✅ |

### What's still NOT wired (single-contract)

| Feature | Reason | Path forward |
|---|---|---|
| **T1.38 BookResilience** | Needs depth state at +N seconds AFTER each large trade — per-event lookforward, not a per-bar aggregation. | New `src/data/bars_resilience.py` per-trade processor + new bar-builder pass. V1.5. |
| **T1.43 cancel proxy levels 2-5** | L1 wired; L2-L5 needs price-level promotion/demotion handling. | Extend `bars_cancel.py`. V1.5. |
| **Auto-d fracdiff** | Currently fixed `d=0.4`. | Calibrate per-instrument once a year via `engines.fracdiff_auto_d` (ADF). V1.5. |
| **T6.05/06/08 macro-event windows** | Need scraped macro calendar (BLS/Fed/EIA). | New `src/features/events.py` + CSV. V1.5. |
| **Cross-sectional features** | Phase 4 orchestrator (`build_cross_panel.py`) ready; needs single panels for >1 instrument. | Run multi-instrument production build. |
| **Tier 4 equity sector ETF features** | Needs new equity 1-sec TAQ ingest pipeline. | V2 — only if V1 OOS underperforms. |

### Current production state (2026-04-25 evening)

| Stage | Status |
|---|---|
| Phase A bars (30 instruments) | ✅ done |
| Phase A+B bars (ES/NQ/RTY/YM) | ✅ done |
| VX1/VX2/VX3 bars | ✅ done (1292 days × 3 slots) |
| 5-sec bars: ES | ✅ done (1299 days) |
| 5-sec bars: NQ/RTY/YM | ⏳ pending — to be submitted post-multi-instrument-build |
| Phase E bars: ES | ✅ done (1287 days; 8 missing-source-file days) |
| Phase E bars: NQ/RTY/YM | ⏳ pending — to be submitted post-multi-instrument-build |
| GEX profiles (SPX/SPY/NDX/RUT/DJX × 5 yr) | ✅ done (25 profiles) |
| Phase 2b futures panels (30 instruments × 5 yrs) | 🟡 in-flight on quartz — ~70/150 done |
| Phase 2b options panels (4 trading × 5 yrs) | 🟡 chained — pending futures completion |
| Phase 3 single panels (30 instruments × 5 yrs) | 🟡 chained — pending |
| Phase 4 cross-sectional panels (4 × 5 yrs) | ✅ done — 20 cross panels on disk (~1,718 cols × 9-10K rows each) |

### Next actions queue

All 4 phases (bars + per-source panels + single panels + cross panels) ✅ done
end-to-end for 2020-2024 across 30 instruments. The data foundation is ready.

Remaining V1 work:

1. **IC + t-stat dashboard** — for each of ~1,718 cross panel features, compute
   information coefficient (rank corr) vs the V1 triple-barrier label across
   all 4 trading targets, with t-statistic. Identify top features per target.
2. **ILP feature selection** — greedy MWIS at ρ<0.45 pairwise correlation
   constraint to prune 1,718 → ~50-80 model-input features.
3. **LightGBM training** — walk-forward CV with embargo + purge = label
   horizon T (8 bars for ES). Per-target 3-class classifier. Trained on
   IS = 2020-2023, evaluated on OOS = 2024.
4. **Backtest + Sharpe attribution**.
5. (Optional) **2025 OOS extension** — when 2025 raw Algoseek data lands,
   re-run all 9 phases for the 2025 year-task only.

---

## Common pitfalls

- **Time precision auto-detect**: Algoseek futures TAQ uses `HHMMSSnnnnnnnnn` (15 chars, ns), VIX TAQ uses `HHMMSSmmm` (9 chars, ms). `src/data/ingest._parse_ts` auto-detects from max string length AND coerces the result to `datetime[ns, UTC]` so all bars have a consistent ts dtype across data sources. (Without the final coercion, polars' `pl.duration(milliseconds=…)` promotes ns→μs and asof-joins between VX and futures bars fail with `SchemaError`.)
- **VIX has no aggressor classification**: only `TRADE` (no `TRADE AGRESSOR ON BUY/SELL`). VX bars will have `buys_qty=sells_qty=0`. VX features use mid/spread only — don't compute OFI from VX.
- **Polars CSV i64 inference on depth files**: depth's L{k}Price columns can be inferred as i64 from early null/zero rows, then crash on later fractional prices. `src/data/ingest.read_depth` forces `Float64` via `schema_overrides` — keep that.
- **`group_by_dynamic` with empty trade days**: zero-trade-day files (holidays, halts) make `build_5sec_bars_core` return zero rows. The bar builders treat that as soft-skip (return 0,0; log `[empty]`); they do NOT raise.
- **Front-N for VX**: liquidity-rank ordering breaks near roll. `roll.front_n` parses calendar expiry for VX and orders by `(year, month)` ascending.
- **`src.features.panel` is a backward-compat facade**: the original monolithic module was split into `single_contract.py` / `external_sources.py` / `cross_sectional.py` / `labeling.py`. New code should import from those directly; `panel.py` re-exports for legacy callers and may be removed in a future cleanup.
- **Stage-wise rebuilds**: each phase's output is a parquet. Rebuilding a single stage (e.g., re-running `build_futures_panel.py` after a feature change) reuses the options panel from disk; only the changed stage runs. Use this for fast feature iteration.
