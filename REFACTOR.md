# Build-Pipeline Refactor

**Status**: ✅ **executed** (commits 16ec935 → 5d41a1e, 2026-04-25). The
target architecture below is now the actual state of the repo. Two of six
steps were intentionally deferred — see "Migration plan" / "What shipped vs
deferred" sections at the bottom.

This doc remains in the repo as the architectural rationale; for operational
"how to build" guidance see `BUILD_PIPELINE.md`.

## Why

The current pipeline is functional but fragmented:

- 6 ingestion / build scripts (`build_phase_a_bars`, `build_l2_bars`,
  `build_5sec_bars`, `build_vx_bars`, `build_gex_features`, `build_es_panel`)
  with implicit ordering dependencies and no single source of truth for
  "what runs in what order."
- `panel.py` has 8 attach functions with cross-cutting concerns (per-instrument
  vs cross-asset vs target-only) that both make sense and obscure the mental
  model.
- VX is treated as a special case in `build_vx_bars.py` even though structurally
  it's just another contract with a monthly cycle.
- Cross-sectional wiring exists but is never actually invoked end-to-end.
- New contributors have to read 4 scripts + `panel.py` + the SLURM sbatches to
  understand "how do I rebuild the panel for ES."

This is fine when 1 person knows it cold. It scales poorly past that.

## Target architecture — four phases

```
Phase 1: BARS  (all contracts × all granularities)
   build_bars.py --mode {15m, 5s, l2, vx}  --instrument <X>  --year <Y>

       ↓ produces  bars/{mode}/{INSTR}/{INSTR}_{YYYYMMDD}_{horizon}.parquet

Phase 2: SINGLE-SOURCE FEATURE PANELS  (per data source, per instrument)
   build_futures_panel.py  --instrument <X> --year <Y>
   build_options_panel.py  --instrument <X> --year <Y>      # ES uses SPX, NQ uses NDX, etc.

       ↓ produces  features/{source}/{INSTR}_{YYYY}.parquet

Phase 3: SINGLE-CONTRACT MERGED PANEL
   build_single_panel.py --instrument <X> --year <Y>

       ↓ joins {futures, options} feature panels for one instrument
       ↓ produces  features/single/{INSTR}_{YYYY}.parquet
       ↓ when args.label, also applies V1 triple-barrier here

Phase 4: CROSS-SECTIONAL PANEL
   build_cross_panel.py --year <Y> --target <X>

       ↓ joins all 30 single-contract panels on ts
       ↓ computes Gauss-Rank universe + within-asset-class
       ↓ computes synthetic DXY, rates curve, risk-on/off composites
       ↓ writes target-specific:  features/cross/{TARGET}_{YYYY}.parquet
```

Four CLIs, four phases. Each phase has one orchestrator script. Each
orchestrator reads its inputs from the previous phase's output directory.

## Concrete file layout (after refactor)

### Keep (rename / move)

| Current | Refactored | Notes |
|---|---|---|
| `src/data/bars_5sec.py` | `src/data/bar_builders.py::build_5sec_core` | merge bar-builder primitives into one module |
| `src/data/bars_downsample.py` | `src/data/bar_builders.py::downsample` | same |
| `src/data/depth_snap.py` | `src/data/bar_builders.py::attach_book_snapshot` | same |
| `src/data/ingest.py` | unchanged | pure ingest stays separate |
| `src/data/roll.py` | unchanged | roll logic stays separate |
| `src/features/panel.py` | split into 3 modules — see below | currently does too much |

### Replace `src/features/panel.py` with three focused modules

| New module | Responsibility | Functions |
|---|---|---|
| `src/features/single_contract.py` | Per-instrument feature computation | `attach_base_microstructure_features`, `attach_pattern_features`, `attach_engine_features`, `attach_l2_deep_features`, `attach_ts_normalizations`, `attach_ema_smoothed`, `build_per_instrument_features` |
| `src/features/external_sources.py` | Features from non-bar sources | `attach_gex_for_target`, `attach_vx_features` (asof-join VX onto target) |
| `src/features/cross_sectional.py` | Multi-instrument operations | `build_wide_cross_asset_frame`, `attach_cross_sectional_ranks`, `attach_cross_asset_composites` |
| `src/features/labeling.py` | Final assembly + label apply | `assemble_target_panel` (currently in `panel.py`) |

### Replace 6 build scripts with 4 orchestrators

| Current | Refactored | Notes |
|---|---|---|
| `build_phase_a_bars.py` | `build_bars.py --mode 15m` | mode dispatches on the bar type |
| `build_l2_bars.py` | `build_bars.py --mode l2` | shares CLI / sbatch wrapper |
| `build_5sec_bars.py` | `build_bars.py --mode 5s` | same |
| `build_vx_bars.py` | `build_bars.py --mode vx` | VX is just `--instrument VX --mode 15m` with calendar-front-N roll |
| `build_gex_features.py` | `build_options_panel.py` | renamed to fit Phase 2 taxonomy |
| `build_es_panel.py` | becomes obsolete | replaced by `build_futures_panel` + `build_single_panel` |
| (new) | `build_futures_panel.py` | bar-derived features (per-instrument, per-year) |
| (new) | `build_single_panel.py` | joins futures+options panels for one instrument |
| (new) | `build_cross_panel.py` | Phase 4 |

### sbatch wrappers — one per orchestrator

```
scripts/sbatch/
  bars_15m.sbatch          # array=0-29 over instruments
  bars_l2.sbatch           # array=0-3 over trading instruments
  bars_5s.sbatch           # array=0-29 (or just trading 4 — TBD)
  bars_vx.sbatch           # single task
  futures_panel.sbatch     # array=year × instrument
  options_panel.sbatch     # array=year × ticker (SPX, SPY, ...)
  single_panel.sbatch      # array=year × instrument
  cross_panel.sbatch       # array=year (or year × target)
```

## Data layout (after refactor)

```
/N/project/ksb-finance-backtesting/data/
├── algoseek_futures/                   # raw, unchanged
├── spx_options_chain/                  # raw, unchanged
├── bars/
│   ├── 15m/{INSTR}/{INSTR}_{DATE}_15m.parquet
│   ├── l2/{INSTR}/{INSTR}_{DATE}_15m.parquet
│   ├── 5s/{INSTR}/{INSTR}_{DATE}_5s.parquet
│   └── vx/{VX1,VX2,VX3}/...
├── features/
│   ├── futures/{INSTR}_{YEAR}.parquet     # Phase 2 output
│   ├── options/{INSTR}_{YEAR}.parquet     # Phase 2 output
│   ├── single/{INSTR}_{YEAR}.parquet      # Phase 3 output
│   └── cross/{TARGET}_{YEAR}.parquet      # Phase 4 output
└── label_tuning/                          # unchanged
```

## Key design decisions to lock down before coding

1. **Which contracts get 5-sec bars?** Currently only ES is queued. For
   cross-sectional VPIN/Hawkes-derived signals we'd want at least the trading
   4 (ES/NQ/RTY/YM). For 30-instrument cross-sectional, full 30 is overkill —
   most macro features only need 15-min granularity. **Proposed**: 5s for
   trading 4, 15m for all 30.

2. **Options panel coverage**. GEX only meaningfully attaches to ES (via SPX
   chain) and NQ (via NDX chain). Other 28 contracts don't have an options
   panel. **Proposed**: `build_options_panel.py --instrument {ES,NQ}` only;
   `build_single_panel.py` left-joins options if a panel exists, else passes
   through.

3. **Phase 3 vs Phase 4 labeling**. Triple-barrier labels are per-target
   (different k_up/k_dn/T per instrument). Where do they apply?
   **Proposed**: in Phase 3 (`build_single_panel.py`) — labels live with the
   per-instrument data they're computed from. Phase 4 cross-sectional
   features just attach to already-labeled rows.

4. **Cross-sectional scope**. The current `panel.py` has both within-universe
   and within-asset-class Gauss-Rank, plus synthetic DXY / rates curve / risk-on
   composites. Should Phase 4 emit one wide cross-sectional panel (all
   contracts, joined) or N target-specific panels (one per trading instrument)?
   **Proposed**: target-specific. Each trading instrument gets its own
   cross-sectional panel that includes its CS features + macro composites.

5. **Caching / incremental builds**. Currently every panel rebuild rewrites
   the whole parquet. With 4 phases and 5 years × 30 instruments × 4 outputs,
   a full rebuild is non-trivial.
   **Proposed for V1.5+**: each phase output writes a `_BUILD_INFO.json`
   sidecar with input hash + git SHA; if the input is unchanged, skip the
   stage. Don't build now; defer.

## Migration plan (when we execute)

Do this incrementally so we don't break the working build:

1. **Add new structure side-by-side**. Create `src/features/single_contract.py`
   etc. as thin facades over the existing `panel.py` functions. Don't delete
   `panel.py` yet.
2. **Refactor `build_es_panel.py` → `build_futures_panel.py` + `build_single_panel.py`**.
   Validate by comparing output parquets to current `build_es_panel.py` byte-for-byte
   (modulo column reordering).
3. **Add `build_bars.py` with `--mode 15m`**. Validate against `build_phase_a_bars.py`
   on a smoke day. Then add `--mode l2`, `--mode 5s`, `--mode vx` one at a time.
4. **Switch sbatch wrappers** to call the new orchestrators. Old scripts kept
   as legacy until next quarter.
5. **Build Phase 4 (`build_cross_panel.py`)**. This is genuinely new work — the
   current pipeline doesn't actually emit a cross-sectional panel.
6. **Delete `panel.py` and old scripts** once the new ones are validated end-to-end.

Each step is independently revertable. No flag day.

## Out of scope (explicit non-goals)

- Rewriting the bar-builder primitives. `build_5sec_bars_core`, `attach_book_snapshot`,
  `cvd_with_dual_reset` — those work, no need to touch.
- Changing the on-disk schema of any existing parquet. Refactor is structural,
  not data-format.
- Replacing polars or rethinking the partitioning. Polars + per-day parquets is
  a working choice for our scale.
- Extending to more instruments / asset classes. We're cleaning up V1, not
  expanding scope.

## Effort estimate

Rough order of magnitude (1 person, focused):

- Structure refactor (steps 1–4): **2–3 days** of careful work + testing.
- Phase 4 cross-sectional (step 5): **1–2 days** — the wiring exists in
  `panel.py`; need an orchestrator and sbatch.
- Cleanup and delete legacy (step 6): **half a day**.

Total: ~5 days. Worth doing once V1 single-contract panels are stable and
before adding cross-sectional features in earnest.

---

## What shipped vs deferred (post-execution, 2026-04-25)

### ✅ Shipped

| Step | Commit | Detail |
|---|---|---|
| 1. Module split | 16ec935 | `src/features/panel.py` → `single_contract.py` / `external_sources.py` / `cross_sectional.py` / `labeling.py`; `panel.py` retained as a thin facade for backward compat. |
| 2. Script split | 16ec935 | `build_es_panel.py` → `build_futures_panel.py` + `build_options_panel.py` + `build_single_panel.py`. Legacy script removed. |
| 3. Phase 4 orchestrator | 16ec935 | `scripts/build_cross_panel.py` + `build_cross_panel.sbatch` (4 targets × 5 years). |
| 6. Cleanup | 16ec935 | Old `build_es_panel.py` and the older `build_features_panel.py` deleted. `panel.py` kept as shim (lighter than full removal; no behavioral cost). |

### 🟡 Intentionally deferred

| Step | Why | Trigger |
|---|---|---|
| 4. Bar-builder consolidation (`build_bars.py --mode`) | The 4 per-bar-type scripts work, are independent, and have different SLURM resource asks. Consolidating is cosmetic. | Skip until we feel real pain from script-count proliferation. |
| 5. Caching / `_BUILD_INFO.json` sidecars | Premature; full rebuilds are ~5-10 min/year per stage so cost is bounded. | Add when iteration cadence on a stage exceeds ~5/day. |

### Validation that the refactor preserves behavior

Smoke-tested end-to-end on bigred for ES 2024 immediately after each refactor commit:
- 16ec935 (post-split): 396-col futures panel matches pre-refactor 376 + 20 VX (which was newly wired same commit). Tests stayed at 249/249.
- 040afc2 / 5d41a1e (Phase E + F additions): 416 cols (no Phase E parquet) → 9,970 valid labeled rows for 2024.
- All 269 unit tests passing throughout the refactor.
