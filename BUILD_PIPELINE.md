# Feature-Panel Build Pipeline

**Purpose**: one place to look up *how to build the V1 ES feature panel from scratch*.
**Scope**: V1 = single-contract ES, ~378 features per 15-min bar.
**Companion docs**: `FEATURES.md` (the catalog of what each feature means) and
`LABELING_V1_SUMMARY.md` (the triple-barrier label spec).

---

## TL;DR — rebuild end-to-end

Assuming raw Algoseek mirror is already on bigred at
`/N/project/ksb-finance-backtesting/data/algoseek_futures/{taq,depth,vix}/...`:

```bash
# 1. Bars (independent, can run in parallel)
ssh bigred 'cd ~/us_futures && sbatch scripts/build_phase_a_bars.sbatch'      # 30 instruments × 15m  (~6h, array of 30)
ssh bigred 'cd ~/us_futures && sbatch scripts/build_l2_bars.sbatch'           # ES/NQ/RTY/YM Phase A+B (~3h, array of 4)
ssh bigred 'cd ~/us_futures && sbatch scripts/build_vx_bars.sbatch'           # VX1/VX2/VX3 15m       (~75min)
ssh bigred 'cd ~/us_futures && sbatch scripts/build_5sec_bars.sbatch'         # ES 5-sec (for VPIN/Hawkes; ~60min)

# 2. GEX (depends on SPX options chain at /N/.../spx_options_chain/)
ssh bigred 'cd ~/us_futures && sbatch scripts/build_gex_features.sbatch'      # SPX+SPY × 5 yrs  (~15min, array of 10)

# 3. ES feature panel (depends on Phase A+B ES, VX1-3, GEX)
ssh bigred 'cd ~/us_futures && sbatch scripts/build_es_panel.sbatch'          # ES × 5 years (~30min, array of 5)
```

Output: `/N/project/ksb-finance-backtesting/data/features/ES_panel_{2020..2024}.parquet`

Local smoke (single year, single day) before any SLURM job — see "Smoke tests" below.

---

## Stage map

```
                 raw Algoseek (taq/depth/vix)
                          │
        ┌─────────────────┼──────────────────────────┐
        │                 │                          │
        ▼                 ▼                          ▼
  Phase A bars      Phase A+B bars             VX1/VX2/VX3 bars
  (30 instr)        (ES/NQ/RTY/YM)             (3 slots)
  15m parquets      15m parquets w/ L1-L10     15m parquets
        │                 │                          │
        │                 ▼                          │
        │           5-sec bars (ES)                  │
        │           (for VPIN/Hawkes — V1.5)         │
        │                                            │
        │                  SPX options chain         │
        │                          │                 │
        │                          ▼                 │
        │                  GEX daily profile         │
        │                          │                 │
        └─────────────────┬────────┴─────────────────┘
                          ▼
               ╔════════════════════════╗
               ║  build_es_panel.py     ║
               ║  (single-contract ES)  ║
               ╚════════════════════════╝
                          │
                          ▼
              ES_panel_{YEAR}.parquet
              (~378 cols, labeled, valid rows only)
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
- **Used by**: VPIN volume buckets, Hawkes intensity recursion, sub-bar realized moments. **Not currently consumed by `build_es_panel.py` — V1.5 wiring needed.**
- **Script**: `scripts/build_5sec_bars.py`, `build_5sec_bars.sbatch`.
- **Status**: 🟡 queued (job 6921629).

### S5. GEX daily profile (SPX, SPY)
- **Reads**: `spx_options_chain/{TICKER}_{YEAR}.parquet` (WRDS OptionMetrics).
- **Writes**: `gex_features/{TICKER}_gex_profile_{YEAR}.parquet`.
- **Cols**: total_gex, gex_sign, zero_gamma_strike, max_call_oi_strike, max_put_oi_strike, gex_0dte_share, gex_0dte_only, gex_without_0dte.
- **Script**: `scripts/build_gex_features.py`, `build_gex_features.sbatch`.
- **Status**: ✅ done — SPX + SPY × 2020-2024.

### S6. ES feature panel (the V1 entry point)
- **Reads**: Phase A+B ES + VX1/VX2/VX3 + GEX.
- **Writes**: `features/ES_panel_{YEAR}.parquet`.
- **Script**: `scripts/build_es_panel.py`, `build_es_panel.sbatch` (array=0-4, one year per task).
- **Pipeline inside the script** (orchestrated by `src/features/panel.py`):

  ```
  load Phase A+B ES bars (year)             →  90 cols
  ├── attach_base_microstructure_features    +44   (returns/flow/vol/moments/illiq/...)
  ├── attach_session_flags                   +5
  ├── attach_minute_of_day_cyclic            +2
  ├── attach_overnight_features              +4
  ├── attach_pattern_features                +13   (T7.* breakouts/divergences)
  ├── attach_engine_features                 +4    (fracdiff + 3 round-pin)
  ├── attach_l2_deep_features (depth=10)     +50   (10 deep + 40 per-level)
  ├── attach_vx_features (asof-join)         +7    (mid/zscore/calendar/curvature)
  ├── attach_gex_for_target                  +5
  ├── attach_ts_normalizations               +136  (34 BASE × 2 lookbacks × 2 z-types)
  ├── attach_ema_smoothed                    +16   (8 features × 2 spans)
  └── assemble_target_panel (V1 labels)      +8    (label, realized_ret, atr, halt flags)
                                             ─────
                                             ~378 cols
  ```

- **Flags**:
  - `--year YYYY` (required)
  - `--out PATH` (required, output directory)
  - `--label` apply triple-barrier labels
  - `--no-vx` / `--no-gex` / `--no-l2-deep` skip respective stage if its inputs aren't ready

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

# Full ES panel — one year
ssh bigred 'cd ~/us_futures && python -u scripts/build_es_panel.py \
    --year 2024 --out /tmp/es_panel_smoke --label'
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

- **Time precision auto-detect**: Algoseek futures TAQ uses `HHMMSSnnnnnnnnn` (15 chars, ns), VIX TAQ uses `HHMMSSmmm` (9 chars, ms). `src/data/ingest._parse_ts` auto-detects from max string length. Don't hard-code the format.
- **VIX has no aggressor classification**: only `TRADE` (no `TRADE AGRESSOR ON BUY/SELL`). VX bars will have `buys_qty=sells_qty=0`. VX features use mid/spread only — don't compute OFI from VX.
- **Polars CSV i64 inference on depth files**: depth's L{k}Price columns can be inferred as i64 from early null/zero rows, then crash on later fractional prices. `src/data/ingest.read_depth` forces `Float64` via `schema_overrides` — keep that.
- **`group_by_dynamic` with empty trade days**: zero-trade-day files (holidays, halts) make `build_5sec_bars_core` return zero rows. The bar builders treat that as soft-skip (return 0,0; log `[empty]`); they do NOT raise.
- **Front-N for VX**: liquidity-rank ordering breaks near roll. `roll.front_n` parses calendar expiry for VX and orders by `(year, month)` ascending.
- **No incremental panel rebuild**: `build_es_panel.py` rewrites the full panel parquet every run. If only one feature group changes, the entire pipeline re-runs (~5-10min/year). Plan around this when iterating on feature additions.
