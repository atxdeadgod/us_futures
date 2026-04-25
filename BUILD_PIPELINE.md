# Feature-Panel Build Pipeline

**Purpose**: one place to look up *how to build the V1 ES feature panel from scratch*.
**Scope**: V1 = single-contract ES, ~396 features per 15-min bar (with VX + GEX wired).
**Companion docs**: `FEATURES.md` (the catalog of what each feature means),
`LABELING_V1_SUMMARY.md` (the triple-barrier label spec), and
`REFACTOR.md` (the 4-phase architecture this pipeline follows).

---

## TL;DR — rebuild end-to-end

Assuming raw Algoseek mirror is already on bigred at
`/N/project/ksb-finance-backtesting/data/algoseek_futures/{taq,depth,vix}/...`:

```bash
# Phase 1: Bars (independent, can run in parallel)
ssh bigred 'cd ~/us_futures && sbatch scripts/build_phase_a_bars.sbatch'      # 30 instruments × 15m  (~6h, array of 30)
ssh bigred 'cd ~/us_futures && sbatch scripts/build_l2_bars.sbatch'           # ES/NQ/RTY/YM Phase A+B (~3h, array of 4)
ssh bigred 'cd ~/us_futures && sbatch scripts/build_vx_bars.sbatch'           # VX1/VX2/VX3 15m       (~75min)
ssh bigred 'cd ~/us_futures && sbatch scripts/build_5sec_bars.sbatch'         # ES 5-sec (for VPIN/Hawkes; ~60min)
ssh bigred 'cd ~/us_futures && sbatch scripts/build_phase_e_bars.sbatch'      # ES Phase E exec/cancel/quote-count (~2h)

# Phase 2a: GEX daily profile (depends on SPX options chain at /N/.../spx_options_chain/)
ssh bigred 'cd ~/us_futures && sbatch scripts/build_gex_features.sbatch'      # SPX+SPY × 5 yrs  (~15min, array of 10)

# Phase 2b: per-source feature panels (independent of each other)
ssh bigred 'cd ~/us_futures && sbatch scripts/build_futures_panel.sbatch'     # bar-derived + VX, ES × 5 yrs  (~30min, array of 5)
ssh bigred 'cd ~/us_futures && sbatch scripts/build_options_panel.sbatch'     # GEX-derived, ES × 5 yrs       (~10min, array of 5)

# Phase 3: single-contract merged panel + V1 labels
ssh bigred 'cd ~/us_futures && sbatch scripts/build_single_panel.sbatch'      # join futures+options + label, ES × 5 yrs (~10min, array of 5)

# Phase 4: cross-sectional features (only after building single panels for ALL 30 instruments)
ssh bigred 'cd ~/us_futures && sbatch scripts/build_cross_panel.sbatch'       # 4 trading instruments × 5 yrs (~30min, array of 20)
```

Outputs:
- `/N/.../features/futures/{INSTR}_{YEAR}.parquet`   (Phase 2b output, bar-derived)
- `/N/.../features/options/{INSTR}_{YEAR}.parquet`   (Phase 2b output, GEX)
- `/N/.../features/single/{INSTR}_{YEAR}.parquet`    (Phase 3 output, merged + labeled)
- `/N/.../features/cross/{TARGET}_{YEAR}.parquet`    (Phase 4 output, with CS features)

Local smoke (single year, single day) before any SLURM job — see "Smoke tests" below.

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

### S4. 5-sec bars (ES, for sub-bar engines)
- **Reads**: same as Phase A.
- **Writes**: `bars_5sec/ES/5s/ES_{YYYYMMDD}_5s.parquet` (~15K rows/day × 29 cols).
- **Used by**: VPIN volume buckets (`engines.vpin_volume_buckets`), Hawkes intensity recursion (`engines.hawkes_intensity_recursive`); wired into `build_futures_panel.py` via `sub_bar_engines.attach_sub_bar_engine_features`.
- **Script**: `scripts/build_5sec_bars.py`, `build_5sec_bars.sbatch`.
- **Status**: ✅ done (1299 days for ES).

### S4b. Phase E bars (ES, execution-quality + cancel-proxy + quote-event + quote-direction aggregates)
- **Reads**: raw TAQ + depth (Algoseek).
- **Writes**: `bars_phase_e/ES/15m/ES_{YYYYMMDD}_15m.parquet` (24 cols).
- **Cols**:
  - Effective spread: eff_spread_{sum,weight,count,buy_sum,buy_weight,sell_sum,sell_weight} (T1.35-T1.37 prereqs)
  - Large trades: n_large_trades, large_trade_volume (T1.23 prereqs)
  - Hidden absorption: hidden_absorption_{volume,trades} (T1.47/T7.12 prereqs)
  - Cancel proxy: net_{bid,ask}_decrement_no_trade_L1 (T1.43 prereqs)
  - **Side-conditioned shifts (T1.28 prereqs)**: bid_sz_L1_delta_signed, ask_sz_L1_delta_signed, hit_bid_vol, lift_ask_vol
  - Quote count: quote_update_count (T1.24 prereq)
  - **Quote direction (T1.25 prereqs)**: bid_up_count, bid_down_count, ask_up_count, ask_down_count
- **Used by**: `single_contract.attach_phase_e_features` to derive: vwap_eff_spread + asymmetry (T1.35-T1.37), large_trade_volume_share (T1.23), hidden_absorption_ratio (T1.47/T7.12), cancel_to_trade_ratio (T1.43), quote_to_trade_ratio (T1.24), **quote_movement_directionality (T1.25)**, **side_cond_ask_resilience_buy / side_cond_bid_resilience_sell (T1.28)**.
- **Script**: `scripts/build_phase_e_bars.py`, `build_phase_e_bars.sbatch`.

  Note: T1.29 liquidity_migration is NOT in Phase E — it's derived directly from Phase A+B's L1-L5 cols via bar-to-bar deltas in `attach_l2_deep_features`.

### S5. GEX daily profile (SPX, SPY)
- **Reads**: `spx_options_chain/{TICKER}_{YEAR}.parquet` (WRDS OptionMetrics).
- **Writes**: `gex_features/{TICKER}_gex_profile_{YEAR}.parquet`.
- **Cols**: total_gex, gex_sign, zero_gamma_strike, max_call_oi_strike, max_put_oi_strike, gex_0dte_share, gex_0dte_only, gex_without_0dte.
- **Script**: `scripts/build_gex_features.py`, `build_gex_features.sbatch`.
- **Status**: ✅ done — SPX + SPY × 2020-2024.

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
- Adds Gauss-Rank universe + per-asset-class ranks, synthetic DXY, rates curve, risk-on/off, cross-asset rolling correlations on top of target's single panel.

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

## Column inventory (post-VX, post-GEX, ~378 cols total)

| Group | Count | Source | Notes |
|---|---:|---|---|
| identity / labels | 10 | bar build + V1 labeling | ts, root, expiry, label, realized_ret, atr, halt_truncated, ... |
| base micro | ~44 | `panel.attach_base_microstructure_features` | 34 BASE_VALUE_COLS + helpers (vwap_deviation, rolling vol/trade-count helpers) |
| TC z-score normalizations | 68 | `panel.attach_ts_normalizations` | 34 BASE × {30, 60} day lookback |
| MAD z-score normalizations | 68 | same | robust-to-outliers variant |
| L2 deep (composite) | 10 | `panel.attach_l2_deep_features` | cum/dw imbalance, depth-weighted spread, HHI bid+ask, deep_ofi×2, spread_z |
| L2 per-level | 40 | same | volume_imbalance/basic_spread/ofi_at L1..L10 + order_count/size_imbalance L1..L5 |
| patterns (T7.*) | 13 | `panel.attach_pattern_features` | breakouts, reversals, CVD divergence, range compression, absorption, spike-fade |
| engine | 4 | `panel.attach_engine_features` | fracdiff(d=0.4) + round_pin {N=10,25,50} |
| EMA-smoothed | 16 | `src/features/smoothed.py` | 8 base × spans {10, 30} causal EMA |
| session / cyclic / overnight | 11 | `tc_features` + `overnight` | hour_et + 4 session flags + sin/cos minute + 4 overnight |
| VX | 7 | `panel.attach_vx_features` | vx1/2/3_mid, calendar_spread/ratio, term_curvature, vx1_zscore, vx_spread_z |
| GEX | 5 | `panel.attach_gex_for_target` | total_gex, gex_sign, distance_to_{zero_gamma_flip, max_call_oi, max_put_oi} |

The exact column list is printed at the end of every `build_es_panel.py` run
under `[col-summary]`.

---

## V1.5+ deferred features (not in current panel)

| Feature | Why deferred | What's needed |
|---|---|---|
| **VPIN** (volume buckets, Easley-LdP) | Needs sub-bar trade stream at feature time | After 5-sec bars finish: per-day VPIN per bucket → asof-join to 15-min bars |
| **Hawkes** (recursive λ_buy/λ_sell) | Same — needs actual Δt event stream | Same pattern |
| **Effective spread / large-trade share / hidden-liquidity rolling ratio** | Needs Phase E (`bars_exec.py`) cols | Run a Phase E bar build for ES |
| **Cancel-to-trade ratio** (T1.43) | Needs `bars_cancel.py` Phase | Same; cancel-proxy from MBP-10 snapshot deltas |
| **Quote dynamics** (T1.24/25/28/29) | Needs per-quote-event aggregation inside the bar | Extend `bars_5sec.build_5sec_bars_core` to emit quote-update counts |
| **Auto-d fracdiff** | Currently uses fixed d=0.4 | Calibrate per-instrument via `engines.fracdiff_auto_d` (ADF) once a year |
| **Cross-sectional 30-instrument ranks + macro composites** | Out of V1 scope (single-contract focus) | Already wired in `panel.attach_cross_sectional_ranks` etc.; just not invoked by `build_es_panel.py` |

---

## Common pitfalls

- **Time precision auto-detect**: Algoseek futures TAQ uses `HHMMSSnnnnnnnnn` (15 chars, ns), VIX TAQ uses `HHMMSSmmm` (9 chars, ms). `src/data/ingest._parse_ts` auto-detects from max string length AND coerces the result to `datetime[ns, UTC]` so all bars have a consistent ts dtype across data sources. (Without the final coercion, polars' `pl.duration(milliseconds=…)` promotes ns→μs and asof-joins between VX and futures bars fail with `SchemaError`.)
- **VIX has no aggressor classification**: only `TRADE` (no `TRADE AGRESSOR ON BUY/SELL`). VX bars will have `buys_qty=sells_qty=0`. VX features use mid/spread only — don't compute OFI from VX.
- **Polars CSV i64 inference on depth files**: depth's L{k}Price columns can be inferred as i64 from early null/zero rows, then crash on later fractional prices. `src/data/ingest.read_depth` forces `Float64` via `schema_overrides` — keep that.
- **`group_by_dynamic` with empty trade days**: zero-trade-day files (holidays, halts) make `build_5sec_bars_core` return zero rows. The bar builders treat that as soft-skip (return 0,0; log `[empty]`); they do NOT raise.
- **Front-N for VX**: liquidity-rank ordering breaks near roll. `roll.front_n` parses calendar expiry for VX and orders by `(year, month)` ascending.
- **`src.features.panel` is a backward-compat facade**: the original monolithic module was split into `single_contract.py` / `external_sources.py` / `cross_sectional.py` / `labeling.py`. New code should import from those directly; `panel.py` re-exports for legacy callers and may be removed in a future cleanup.
- **Stage-wise rebuilds**: each phase's output is a parquet. Rebuilding a single stage (e.g., re-running `build_futures_panel.py` after a feature change) reuses the options panel from disk; only the changed stage runs. Use this for fast feature iteration.
