# Feature Catalog — us_futures V1

Status: **locked for build** (v2), 2026-04-24. Reflects two review rounds with the domain feedback accepted.

This doc inventories every feature that enters the V1 directional-ES model. Each feature lists: PHASE1 tier, data requirement, whether it's already in `feature-factory` or we need to write it, and any implementation nuance.

**Modeling convention** (per `PHASE1_DIRECTIONAL_ES.md §3`): LightGBM primary + meta-labeling; triple-barrier labels; walk-forward with purge+embargo; 2025 held out.

**Feature selection pipeline**: ~125 base features → variant grid → ~400–550 candidates → dashboard IC + ILP (ρ<0.45) → ~40–80 survivors → model.

---

## Legend

| Token | Meaning |
|---|---|
| `bar` | time-bar OHLCV + aggressor-signed volume |
| `sub_bar` | 5-sec sub-bar stats aggregated into the 15-min bar |
| `L1` | top-of-book (bid/ask/size) |
| `L2` | MBP-10 (10 levels per side, prices/sizes/orders) |
| `book_snap` | L1–L10 snapshot at bar close (stored in 5-sec base bar) |
| `ff/<path>` | reuse `feature-factory` class at that path |
| `NEW/<file>` | class we need to write |
| `V1` / `V1.5` / `V3+` | phase at which feature enters the active set |

---

## Tier 1 — ES microstructure + liquidity (47 features)

Direct book/flow signals on the traded contract.

### Reusable from feature-factory

| # | Feature | Source | Data |
|---|---|---|---|
| T1.01 | Microprice | ff/l1/Microprice | L1 |
| T1.02 | MicropriceDrift | ff/l1/MicropriceDrift | L1 |
| T1.03 | Mid-price return | ff/l1/MidPriceReturn | L1 |
| T1.04 | OrderImbalance (L1) | ff/l1/OrderImbalance | L1 |
| T1.05 | SpreadAbs | ff/l1/SpreadAbs | L1 |
| T1.06 | SpreadRelBps | ff/l1/SpreadRelBps | L1 |
| T1.07 | SpreadVolatilityRatio | ff/l1/SpreadVolatilityRatio | L1 |
| T1.08 | QuoteSlopeProxy | ff/l1/QuoteSlopeProxy | L1 |
| T1.09 | VolumeImbalance (L2) | ff/l2/VolumeImbalance | L2 |
| T1.10 | DistanceWeightedImbalance | ff/l2/DistanceWeightedImbalance | L2 |
| T1.11 | CumulativeImbalance | ff/l2/CumulativeImbalance | L2 |
| T1.12 | ImbalancePersistence | ff/l2/ImbalancePersistence | L2 |
| T1.13 | BasicSpread | ff/l2/BasicSpread | L2 |
| T1.14 | DepthWeightedSpread | ff/l2/DepthWeightedSpread | L2 |
| T1.15 | LiquidityAdjustedSpread | ff/l2/LiquidityAdjustedSpread | L2 |
| T1.16 | SpreadAcceleration | ff/l2/SpreadAcceleration | L2 |
| T1.17 | SpreadZScore | ff/l2/SpreadZScore | L2 |
| T1.18 | OrderCountImbalance | ff/l2/OrderCountImbalance | L2 |
| T1.19 | OrderSizeImbalance | ff/l2/OrderSizeImbalance | L2 |
| T1.20 | HerfindahlHirschmanIndex | ff/l2/HerfindahlHirschmanIndex | L2 |
| T1.21 | OrderFlowImbalance (bar) | ff/bar/agg/OrderFlowImbalance | bar |
| T1.22 | AggressorSideVolume | ff/bar/agg/AggressorSideVolume | bar |
| T1.23 | LargeTradeVolumeShare | ff/bar/agg/LargeTradeVolumeShare | bar |
| T1.24 | QuoteToTrade | ff/bar/agg/QuoteToTrade | bar |
| T1.25 | QuoteMovementDirectionality | ff/bar/agg/QuoteMovementDirectionality | bar |
| T1.26 | BidAskDepthRatio | ff/bar/agg/BidAskDepthRatio | bar |
| T1.27 | SideWeightedSpread | ff/bar/agg/SideWeightedSpread | bar |
| T1.28 | SideConditionedLiquidityShift | ff/bar/agg/SideConditionedLiquidityShift | bar |
| T1.29 | LiquidityMigration | ff/bar/agg/LiquidityMigration | bar |
| T1.30 | AmihudIlliquidity | ff/bar/single/AmihudIlliquidity | bar |

### New feature-factory classes we'll write

| # | Feature | File | Data | Notes |
|---|---|---|---|---|
| T1.31 | **DeepOFI (multi-level weighted)** | NEW `features/custom.py` | L2 + sub_bar | Σ w_i · signed ΔQ_L_i across L1–L10, distance-decay weighted; Kolm-Turiel-Westray style |
| T1.32 | **Depth aggregates total** | NEW | L2 | Σ bid_sz L1-10, Σ ask_sz L1-10 |
| T1.33 | **Depth imbalance at k** | NEW | L2 | (Σ bid_{1..k} − Σ ask_{1..k}) / total, k∈{1,3,10} |
| T1.34 | **Kyle's λ proxy** | NEW | bar | Δprice / net aggressor-signed volume |
| T1.35 | **Effective spread** | NEW | sub_bar | mean over trades of 2·\|trade_px − mid_at_trade\| |
| T1.36 | **Realized spread** | NEW | sub_bar | 2·sign·(trade_px − mid_{t+5s}) |
| T1.37 | **Effective spread asymmetry** | NEW | sub_bar | buy-side eff spread − sell-side |
| T1.38 | **Book resilience** | NEW | sub_bar | depth at L1 N sec after a large trade, normalized |
| T1.39 | **Implied volume share** | NEW | bar | frac_implied_volume (uses Algoseek Flag=1) |
| T1.40 | **VPIN** (volume-synchronized imbalance) | NEW `features/engines.py` | volume bucket | Easley-Lopez de Prado; Algoseek aggressor flag bypasses tick-rule error; asof-joined to 15-min bar close |
| T1.41 | **VPIN bucket velocity** | NEW | bar | number of VPIN buckets completed within the 15-min bar |
| T1.42 | **VPIN staleness** | NEW | bar | seconds since last completed bucket at bar close |
| T1.43 | **Cancel-to-Trade Ratio** | NEW | sub_bar | (net bid+ask size decrement without trade) / executed volume; **measurement note**: MBP-10 inference, not MBO truth (see Implementation Traps §8.G) |
| T1.44 | **Hawkes imbalance (fast, HL=5s)** | NEW `features/engines.py` | sub_bar | λ_buy − λ_sell using recursive exponential decay |
| T1.45 | **Hawkes imbalance (slow, HL=60s)** | NEW | sub_bar | same with 60-sec half-life |
| T1.46 | **Hawkes acceleration** | NEW | sub_bar | fast_imbalance − slow_imbalance (fast diverging from slow = regime signal) |
| T1.47 | **Hidden / Iceberg Execution Ratio** | NEW | sub_bar | rolling sum where aggressor_volume_executed > L1_size_before_trade AND price did not tick → hidden liquidity absorbed |

### Deprioritized (present in feature-factory, but NOT grid-expanded for ES)

- `AbsVolumeImbalance`, `DollarImbalance` — for single-instrument ES, correlation with VolumeImbalance (T1.09) is >0.99. Keep the `ff` class available but do not variant-expand. ILP will confirm drop.

---

## Tier 2 — ES time-series / volatility (29 features)

| # | Feature | Source | Data |
|---|---|---|---|
| T2.01 | LogReturn | ff/bar/single/LogReturn | bar |
| T2.02 | CumulativeReturn | ff/bar/single/CumulativeReturn | bar |
| T2.03 | IntradayReturn | ff/bar/single/IntradayReturn | bar |
| T2.04 | OvernightReturn | ff/bar/single/OvernightReturn | bar |
| T2.05 | VWAPReturn | ff/bar/single/VWAPReturn | bar |
| T2.06 | ReturnAutocorrelation | ff/bar/single/ReturnAutocorrelation | bar |
| T2.07 | RealizedVolatility | ff/bar/single/RealizedVolatility | sub_bar |
| T2.08 | RealizedVolatilityEwma | ff/bar/single/RealizedVolatilityEwma | sub_bar |
| T2.09 | RealizedVolatilityStd | ff/bar/single/RealizedVolatilityStd | sub_bar |
| T2.10 | RangeBasedVolatility (Parkinson + GK) | ff/bar/single/RangeBasedVolatility | bar |
| T2.11 | **RealizedHigherMoments (skew, kurt, quart)** | ff/bar/single/RealizedHigherMoments | sub_bar — **needs ≥60 sub-samples per bar** |
| T2.12 | VolatilityOfVolatility | ff/l1/VolatilityOfVolatility | sub_bar |
| T2.13 | UpVolatility (semi-vol up) | ff/l1/UpVolatility | sub_bar |
| T2.14 | DownVolatility (semi-vol down) | ff/l1/DownVolatility | sub_bar |
| T2.15 | VolDirectionRatio | ff/l1/VolDirectionRatio | sub_bar |
| T2.16 | TickVolatility | ff/l1/TickVolatility | sub_bar |
| T2.17 | TickReturnHigherMoments | ff/l1/TickReturnHigherMoments | sub_bar |
| T2.18 | JumpIndicator | ff/bar/single/JumpIndicator | sub_bar |
| T2.19 | JumpIntensity | ff/l1/JumpIntensity | sub_bar |
| T2.20 | VolatilityRatio | ff/bar/single/VolatilityRatio | bar |
| T2.21 | VolumeSurprise | ff/bar/single/VolumeSurprise | bar |
| T2.22 | Turnover | ff/bar/single/Turnover | bar |
| T2.23 | PriceVolumeCorrelation | ff/bar/single/PriceVolumeCorrelation | bar |
| T2.24 | PriceImpactSlope | ff/bar/single/PriceImpactSlope | bar |
| T2.25 | Momentum (L2 flavor) | ff/l2/Momentum | L2 |
| T2.26 | RateOfChange | ff/l2/RateOfChange | L2 |
| T2.27 | TrendStrength | ff/l2/TrendStrength | L2 |
| T2.28 | MeanReversion | ff/l2/MeanReversion | L2 |
| T2.29 | **VWAP deviation** | NEW | bar | close − session_vwap |

---

## Tier 3 — Cross-asset lead-lag (13 features)

Predictors from related instruments, aligned to ES bar close via strict `<` asof-join (no lookahead — see Implementation Traps §8.E).

### Return-level lead-lag

| # | Predictor | Source | Lags (min) |
|---|---|---|---|
| T3.01 | NQ log-return | ff/bar/cross/BarLeadLagReturn | 1, 3, 10 |
| T3.02 | RTY log-return | ff/bar/cross/BarLeadLagReturn | 1, 3 |
| T3.03 | YM log-return | ff/bar/cross/BarLeadLagReturn | 1, 3 |
| T3.04 | ZN log-return | ff/bar/cross/BarLeadLagReturn | 1, 10 |
| T3.05 | VX front-month return | ff/bar/cross/BarLeadLagReturn | 1, 10 |
| T3.06 | 6E log-return | ff/bar/cross/BarLeadLagReturn | 1, 10 |
| T3.07 | NQ−ES beta-adjusted spread Δ | ff/l2/cross/PriceLeadLag + NEW | 1, 10 |

### Book-level cross-market

| # | Feature | Source | Data |
|---|---|---|---|
| T3.08 | **Cross-Market DeepOFI — NQ → ES** | NEW `features/custom.py` | L2 (NQ book) | compute T1.31 on NQ, emit as ES predictor |
| T3.09 | **Cross-Market DeepOFI — ZN → ES** | NEW | L2 (ZN book) | same for ZN; captures rates-equity transmission via rates book flow |
| T3.10 | LeadLagAsymmetryScore | ff/bar/cross/LeadLagAsymmetryScore | bar | |
| T3.11 | CrossCorrelation (ES × other) | ff/l2/cross/CrossCorrelation | L2 | |
| T3.12 | ETFLeadBeta | ff/bar/cross/ETFLeadBeta | bar | |
| T3.13 | ETFShockFollowThrough | ff/bar/cross/ETFShockFollowThrough | bar | |

---

## Tier 4 — Cross-sectional equity-market features (V1.5 only)

**Deferred to V1.5** (not in V1 directional build). When we do add, **scope is restricted to 11 SPDR Select Sector ETFs only** — no single-stock TAQ ingestion. Rationale: 11 ETFs gives 95% of macro equity signal for ~1% of engineering cost vs 500+ single-stock TAQ pipelines.

Universe: XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE, XLC.

| # | Feature | Source |
|---|---|---|
| T4.01 | Sector dispersion | ff/bar/cross/ReturnDispersion |
| T4.02 | Breadth (% sectors positive) | ff/bar/cross/Breadth |
| T4.03 | Leader-laggard spread | ff/bar/cross/LeaderLaggardSpread |
| T4.04 | Cross-sectional PC1 (sectors) | ff/bar/cross/FactorModelResidual |
| T4.05 | ReturnConcentration | ff/bar/cross/ReturnConcentration |
| T4.06 | ResidualDispersion | ff/bar/cross/ResidualDispersion |
| T4.07 | Risk-on/off proxy (XLY − XLP) | NEW |
| T4.08 | Financials vs defensives (XLF − XLU) | NEW |

---

## Tier 5 — Vol / options regime (16 features)

### Part A — VX futures book (8 features)

Computed **directly from our VX/UX L2 book** — no Cboe spot index needed (it updates only every 15s and suffers from SPX option mid-jitter). All features are microsecond-aligned with ES.

| # | Feature | Source | Data |
|---|---|---|---|
| T5.01 | VX1 (front-month) mid price | NEW | VX L1 |
| T5.02 | VX1 20d z-score | NEW | VX bar history |
| T5.03 | **VX1 − VX2 calendar spread** | NEW | VX L1 | term-structure slope, direct from our data |
| T5.04 | VX1 / VX2 ratio | NEW | VX L1 | scale-invariant version of T5.03 |
| T5.05 | **VX OFI (DeepOFI on VX1 book)** | NEW | VX L2 | applies T1.31 to VX — flow into volatility |
| T5.06 | VX1 spread z-score | ff/l2/SpreadZScore on VX | VX L2 | vol-market liquidity regime |
| T5.07 | VX term-structure curvature | NEW | VX multi-expiry | 2nd-difference across VX1, VX2, VX3 |

### Part B — Dealer GEX regime (9 features, from WRDS `optionm` SPX chain)

**Why**: Dealer gamma exposure is the dominant 15-min regime driver post-2022. Baltussen-Da-Lammers-Martens (2021) show GEX sign predicts intraday momentum vs mean-reversion; Barbon-Buraschi (2021) document the zero-gamma-flip amplification regime; Brogaard-Han-Won (2024) on 0DTE dealer-hedging flows.

**Data source**: WRDS `optionm.opprcd_YYYY` — SPX (and SPXW) options EOD chain with strike, expiry, cp_flag, open_interest, impl_volatility, delta, gamma. Free on our WRDS subscription. Daily refresh, but most features are **intraday-varying via ES price moving around the static daily profile**. For V1 sufficient; SpotGamma intraday upgrade deferred to V2.

| # | Feature | Formula | Notes |
|---|---|---|---|
| T5.08 | **GEX total ($ notional, log-signed)** | `sign × log(1 + \|Σ_k sign_k · OI_k · gamma_k · S² · 100\|)` | per-day scalar; regime magnitude |
| T5.09 | **GEX sign regime** | {+1, −1, 0} based on total GEX | tree-friendly regime indicator |
| T5.10 | **ES distance to zero-gamma flip** | `ES_bar_close − zero_gamma_strike_ES_adj` | **intraday-varying** (ES moves around static flip strike) |
| T5.11 | **ES distance to flip, in bp** | T5.10 / ES_close × 10000 | scale-invariant |
| T5.12 | **0DTE GEX share** | `Σ_{0DTE options} abs(gamma·OI) / Σ_all abs(gamma·OI)` | afternoon-regime intensity marker |
| T5.13 | **ES distance to max call-OI strike** | nearest large call cluster → afternoon resistance |
| T5.14 | **ES distance to max put-OI strike** | nearest large put cluster → afternoon support |
| T5.15 | **Zero-gamma cross flag (rolling N bars)** | did ES cross flip level in last N bars? → regime-transition marker |
| T5.16 | **GEX × VIX interaction** | `GEX_sign × VIX_zscore` | negative GEX + high VIX = max trend potential; positive GEX + low VIX = max mean-reversion |

---

## Tier 5b — Term structure (parked for V3+, NOT in V1)

**Removed from V1 directional ES model.** ES basis between quarterly expiries is strictly a function of cost-of-carry (r_f − dividend yield) and does not move with intraday directional equity flow. Worthless for outright directional prediction.

Infrastructure retained for future work on CL / NG / ZN / SR3 (curve trades in Book B, V3+).

---

## Tier 6 — Temporal / event-window (8 features)

| # | Feature | Source | Notes |
|---|---|---|---|
| T6.01 | IntradaySeasonalDeviation | ff/bar/single/IntradaySeasonalDeviation | |
| T6.02 | IsMonday / IsFriday | ff/bar/single/IsMonday, IsFriday | |
| T6.03 | IsMonthStart / IsMonthEnd | ff/bar/single/IsMonthStart, IsMonthEnd | |
| T6.04 | **minute_of_day (integer)** | NEW | **0–1439 integer, NOT sin/cos** — trees split on this natively |
| T6.05 | Distance to next macro release (minutes) | NEW | to next FOMC/NFP/CPI/EIA/ISM |
| T6.06 | EventWindowFlag (±15 min) | NEW | per event type, binary |
| T6.07 | Settlement-window flag (15:45–16:00 ET) | NEW | |
| T6.08 | Post-FOMC 2-hr window flag | NEW | |

`is_rth` (08:30–15:00 CT) lives in the bar schema itself, not as a feature but as a filter/metadata flag.

Removed: sin/cos cyclical encoding (NN-only artifact; trees prefer integer minute_of_day).

---

## Tier 7 — Accumulation / fakeout / reversal patterns (12 features, stateful)

Stateful rolling-window features. Live in `src/features/patterns.py`. All operate on top of the bar parquet.

| # | Feature | Captures | Overfit risk |
|---|---|---|---|
| T7.01 | **Absorption score** | Σ\|aggressor_$\| / (return_vol × notional); sustained one-sided flow with weak price response | medium |
| T7.02 | **Queue replenishment rate** | events/min where L1 depleted then refilled | medium (MBP-10 noise) |
| T7.03 | **Volume-at-price concentration** | % of bar volume within ±k-tick of close | low |
| T7.04 | **Breakout magnitude** | (intrabar_high − prior_N_high) / ATR, symmetric for lows | low |
| T7.05 | **Breakout-reversal flag** (Wyckoff upthrust/spring) | breached prior N-min H/L, closed back inside by >k·ATR | medium |
| T7.06 | **Post-breakout flow reversal** | aggressor sign flip within M sec of breakout | medium |
| T7.07 | **Spike-and-fade volume** | vol_T > k·rolling_mean AND vol_{T+1} < (1/k)·rolling_mean | medium |
| T7.08 | **Imbalance persistence (bar-level)** | run-length of consecutive same-sign aggressor-dominant bars | low |
| T7.09 | **CVD_rth / price divergence flag** | `price_high > rolling_high(30) AND cvd_rth < rolling_cvd_high(30)` (or mirror short-side). Classic mid-freq mean-reversion | low |
| T7.10 | **Range compression indicator** | rolling σ(high − low) / ATR — pre-accumulation marker | low |
| T7.11 | **Round-number pin distance (N=5, 25, 50)** | `min(close % N, N − close % N)`, for N∈{5, 25, 50}; 0DTE pinning regime | low |
| T7.12 | **Hidden liquidity rolling ratio** | 15-min rolling sum of T1.47 (hidden execution ratio) | low |

Removed: Stop-run composite (T7.09 in v1) — 3-event composite with too many tunables, high overfit risk. Replaced by CVD divergence (T7.09 here), which is a well-documented 2-parameter signal.

---

## Deliberately excluded (to prevent scope creep)

- European/Asian indices during US RTH — no incremental signal
- Individual commodity futures (CL/GC/NG) as ES predictors — noisy at bar resolution
- Intraday credit spreads (HYG/LQD) — daily yes, intraday noisy
- Single-stock TAQ (>11 tickers) — overhead not justified vs sector ETFs
- Sin/cos cyclical encodings — NN/linear artifacts, not useful for trees
- Term-structure of ES futures — pure cost-of-carry, no directional info
- Spot VIX from Cboe — derived from SPX option midpoint jitter; use our VX L2 book instead
- Kitchen-sink correlation chasing — no ex-ante thesis → don't include

---

## Bar-schema dependencies

### 5-sec base bar (~60 cols, ~17,000 rows/day, ~5 MB parquet/day)

```
# Identity
ts, root, expiry, is_rth, is_session_warm

# OHLCV + volume
open, high, low, close, volume, dollar_volume

# Aggressor-signed volume (Algoseek flag, not Lee-Ready)
buys_qty, sells_qty, trades_count, unclassified_count

# Implied-vs-direct split (for T1.39)
implied_volume, implied_buys, implied_sells

# L1 snapshot at bar close
bid_close, ask_close, mid_close, microprice_close, spread_abs_close

# L1 spread sub-bar stats (for T1.05-07, T1.16-17)
spread_mean_sub, spread_std_sub, spread_max_sub, spread_min_sub

# L1-L10 book snapshot at bar close (for all L2 features)
bid_px_L1..L10, bid_sz_L1..L10, bid_ord_L1..L10
ask_px_L1..L10, ask_sz_L1..L10, ask_ord_L1..L10
book_ts_close   # actual timestamp of the snapshot

# Effective-spread aggregates over trades in the sub-bar (for T1.35-37)
effective_spread_sum, effective_spread_count
effective_spread_buy_sum, effective_spread_sell_sum

# Large-trade flags (for T1.23, T1.47)
n_large_trades, large_trade_volume

# Cancel proxy — MBP-10 inferred (see §8.G for caveats)
net_bid_decrement_no_trade_L1..L5
net_ask_decrement_no_trade_L1..L5

# Running delta (CVD) with dual reset policy
cvd_globex    # resets Sun 18:00 ET
cvd_rth       # resets 09:30 ET each RTH day
bars_since_rth_reset  # counter for RTH-bounded rolling windows

# Hidden-liquidity tracker (for T1.47)
hidden_absorption_volume    # volume where aggressor_vol > L1_size_pre_trade AND no price tick
```

### 15-min aggregated bar (~90 cols, ~94 rows/day, ~30 KB parquet/day)

Additional columns computed from the 5-sec sub-bars:

```
# Realized moments (from sub-bar log returns)
rv_5s, rv_bipower_5s, realized_skew_5s, realized_kurt_5s, realized_quarticity_5s

# Range-based vol
parkinson_vol, gk_vol

# OFI per level summed across sub-bars (for T1.31)
ofi_L1..L10, deep_ofi_weighted

# Microprice drift
microprice_mean_drift, microprice_net_drift

# Flow aggregates
aggressor_net_dollar, top10pct_trade_share

# Quote/trade intensity
quote_update_count, quote_to_trade_ratio

# Session VWAP state
vwap_session_close, close_minus_vwap

# Effective spread aggregates
eff_spread_mean, eff_spread_asymmetry

# Liquidity-quality
book_resilience, cancel_to_trade_ratio

# Hawkes intensities (recursive, see §8.C)
hawkes_buy_fast, hawkes_sell_fast, hawkes_imbalance_fast
hawkes_buy_slow, hawkes_sell_slow, hawkes_imbalance_slow
hawkes_acceleration, hawkes_is_warm

# VPIN (asof-joined from volume-bucket stream, see §8.B)
vpin, vpin_buckets_completed_in_bar, vpin_staleness_sec

# Hidden liquidity
hidden_liquidity_ratio

# Pin distances
round_number_pin_dist_5, round_number_pin_dist_25, round_number_pin_dist_50
```

---

## Variant-expansion strategy

Per feature, apply the following variants where applicable. **Geometric spacing** to avoid near-duplicate features.

| Dimension | Values |
|---|---|
| Rolling window | {3, 10, 30, 60} bars |
| Lag | {1, 3, 10} bars |
| Depth | {1, 3, 10} levels |
| Transform | `raw`, `rolling_z`, `ffd_d={0.3, 0.4, 0.5}`, `ffd_auto_d`, `ema_hl={5m, 15m, 60m}`, `slope_N`, `rank_over_window` |

Expected candidate pool after expansion: **~400–550** candidates. Post dashboard IC + ILP (ρ<0.45): **~40–80** survivors enter the LightGBM model.

---

## Implementation traps (§8) — NON-NEGOTIABLE

Ignore these at your peril; they silently corrupt features.

### §8.A — FracDiff: use Fixed-Window (FFD), not expanding

**Trap**: true fractional differencing needs expanding window back to t=0 (binomial weights never fully zero). Naive implementation = O(N²) memory, blows up on multi-year bar history.

**Fix**: Fixed-Window FracDiff (FFD) per AFML Ch. 5.
- Compute weights recursively: `w_k = w_{k-1} * -(d - k + 1) / k`, `w_0 = 1`
- Truncate at `|w_k| < tau` (tau = 1e-5 typical)
- Apply as causal FIR via `scipy.signal.lfilter(weights, [1.0], series)` — **not** `convolve` (which is non-causal and can leak future info at edges)
- Typical weight counts: d=0.3 → ~40, d=0.5 → ~200
- First `len(weights)` bars are NaN — downstream must honor

**Auto-d variant**: for each feature we also compute one FFD where d is chosen as the smallest d such that an ADF test on the training window rejects unit-root at p<0.01. Per AFML 5.5.

### §8.B — VPIN: volume bucket timestamps are asynchronous to time bars

**Advantage**: Algoseek aggressor flag bypasses tick-rule ~30% classification error — our VPIN is strictly better than academic VPIN.

**Trap**: VPIN is emitted on volume-bucket boundaries (every N contracts), NOT at 15-min bar boundaries. Naive join → stale or future-leaked values.

**Fix**:
- Volume bucket size parametric per symbol: ES RTH 50k contracts; grid {25k, 50k, 100k}
- Overnight bucket size differs; alternative: skip overnight VPIN or use 10k
- Produce a VPIN event stream with its own timestamps
- **asof-join backward** to the 15-min bar close with strict `<` predicate and **max 30-min staleness** — if no bucket has completed in 30 min, emit NaN
- End-of-day: discard partial bucket; do NOT carry to next day
- Bonus features (T1.41, T1.42): `vpin_buckets_completed_in_bar` (volume-clock velocity), `vpin_staleness_sec`

### §8.C — Hawkes: recursive update using actual Δt

**Trap**: rolling exponential decay via convolution is O(N·W) and computationally abusive on years of 5-sec data.

**Fix**: recursive formulation, O(1) per update:
```
λ_t = λ_{t-1} * exp(-β * Δt_actual) + N_t
β = ln(2) / HL
```

**Sub-traps**:
- `Δt_actual` is elapsed time between consecutive 5-sec observations — **not** assumed 5s (if a bar is missing, actual Δt could be 10s or 15s). Carry `ts_prev` through the recursion.
- `N_t` is aggressor-signed *volume* in the bar (not count), so large institutional trades weight appropriately.
- Warmup: mark `hawkes_is_warm = False` for first `5×HL` bars after session start. For HL=60s, 5 min warmup; for HL=5s, 25 sec.
- Four state variables per bar: (λ_buy_fast, λ_sell_fast, λ_buy_slow, λ_sell_slow). Derived: (imbalance_fast, imbalance_slow, acceleration).

### §8.D — RTH CVD rolling windows must dynamically bound at reset

**Trap**: if CVD resets at 09:30 ET and you compute `rolling_cvd_high(30 bars)`, the first 30 bars of RTH will include pre-RTH zeros or use artificial reset boundary, causing false divergence signals exactly during the period (first hour) where the CVD divergence signal is most used.

**Fix**:
```python
effective_window = min(W, bars_since_rth_reset)
if effective_window < MIN_BARS:  # e.g., 5
    return NaN
rolling_max = series[-effective_window:].max()
```

**Additional policy**: both `cvd_globex` (reset Sun 18:00 ET) and `cvd_rth` (reset 09:30 ET daily) have hard resets. Don't be clever with continuous overnight-to-RTH CVD — weekend gaps and overnight-regime differences produce spurious divergences.

### §8.E — Cross-market timestamp alignment (strict `<`)

**Trap**: computing `deep_ofi(NQ)` aligned to ES bar close time T. If the asof-join uses `<=` (default in many libraries), an NQ event at exactly T could end up in the feature for the ES bar closing at T — subtle but real lookahead.

**Fix**: use **strict `<`** in the asof-join. Polars: `strategy="backward"` with careful inspection; pandas: `merge_asof(direction="backward", allow_exact_matches=False)`.

### §8.F — Round-number pin distance is era-dependent

**Trap**: 50-pt grid works for ES ~5500 (2024+), but at ES ~3000 (pre-2020) meaningful spacing was 25pt; 0DTE also introduces 5pt and 10pt strikes.

**Fix**: compute for N ∈ {5, 25, 50} in parallel, emit all three, let ILP pick based on era. Formula:
```
dist = min(close % N, N − close % N)
```
(note symmetric — both sides of the strike).

### §8.G — Algoseek aggressor: session-boundary contamination

**Trap**: first trades post-daily-reopen (17:00 ET ES Globex after 1-hr maintenance halt) may arrive before initial bid/ask quotes settle. Algoseek's aggressor classification is unreliable during this warmup.

**Fix**: `is_session_warm` flag on bars; first K=30 sec of each session marked `False`. VPIN, Hawkes, cumulative_delta computations respect this flag. Strategy/labels can also gate on it.

### §8.I — Dealer positioning estimation is the hardest part of GEX

True dealer gamma requires knowing **customer flow direction** per option. We don't have that directly from WRDS optionm (it has OI but not signed dealer vs customer side).

**Naive heuristic** (ship V1 with this): dealers are short calls (retail buys calls for convex upside), long puts (retail sells puts for income). Gives correct **regime sign** ~80–85% of the time on SPX per academic consensus — which captures ~80% of the signal.

**Formula** (`dealer_sign_assumption="naive_call_short"`):
```
dealer_gamma_contribution_k = (-1 if call else +1) × OI_k × gamma_k × S² × 100
```

**V2 upgrade path**: calibrate via CBOE COT customer-flow reports or published papers with explicit customer flow data. For V1, naive sign → regime indicator is sufficient.

### §8.J — 0DTE gamma explosion

0DTE options have extreme gamma near the money (T→0 limit of Black-Scholes). One ATM 0DTE option with 5000 OI can dominate the daily GEX sum via its huge per-contract gamma.

**Mitigations**:
- Cap per-option `gamma×OI` contribution at the 99th percentile of that day's options before summing (prevents a single line from dominating)
- Separate tracking: `gex_without_0dte` vs `gex_0dte_only` — let ILP pick which dimension matters
- Report `gex_0dte_share` (T5.12) as an explicit feature — informs strategy when 0DTE dynamics dominate (typically post-14:00 ET)

### §8.K — SPX vs ES basis adjustment

WRDS `optionm` options are on **SPX** (cash index). We trade **ES** futures. Fair-value basis = SPX + carry ≈ +20 to +60 bp typically. For distance features (T5.10, T5.13, T5.14) compute against SPX strikes but **convert to ES-equivalent**:

```
zero_gamma_strike_ES = zero_gamma_strike_SPX + (ES_spot − SPX_spot)
```

Ignore this at your peril when ES runs 40 bp above SPX (would shift distance-to-flip by ~20 ES points — that's 10× a typical tick and destroys the feature's meaning).

**Implementation**: join the daily SPX close (from WRDS) to each bar's ES price to compute the basis adjustment.

### §8.L — GEX lookahead control

The daily GEX profile is computed from **T−1 EOD** options data (available after market close). All GEX features for ES bars on day T are therefore known at T-open — no lookahead.

**But**: if we ever switch to intraday GEX (V2 SpotGamma / Menthor Q), the profile is published at vendor-specific timestamps (typically every 5 min or 15 min). Feature compute must then:
- asof-join backward on **actual vendor publication timestamp**, not nominal bar close
- Apply strict `<` predicate (same rule as §8.E)
- Introduce `gex_staleness_sec` as a companion feature so model can down-weight stale values

### §8.M — Cancel-to-Trade is INFERRED from MBP-10, not measured

**Honesty requirement**: we don't have MBO (order-by-order) data. We infer cancellations from snapshot deltas:
```
cancel_proxy_L1_bid = max(0, L1_bid_sz(t-Δ) − L1_bid_sz(t) − (aggressive_sells_at_L1 in Δ))
```

**Limitations**:
- Can't distinguish pure cancel from cancel-and-replace at same price
- Can't distinguish cancel from level vacation (price moved, old level no longer L1)
- Cancel-to-Trade Ratio has measurement noise; account for this in ablation — expect the feature's OOS IC to be noisier than peers by ~20–30%

**Mitigations**: use as regime/toxicity indicator, not a precise stand-alone signal. Combine with Hawkes imbalance for composite "fake-move" detector.

---

## Engine design — `src/features/engines.py`

Reusable primitives with the gotchas baked in. All pytest-covered.

```python
def ffd_weights(d: float, tau: float = 1e-5) -> np.ndarray: ...
def fracdiff_series(x: pl.Series, d: float, tau: float = 1e-5) -> pl.Series: ...
def fracdiff_auto_d(x: pl.Series, p_value: float = 0.01) -> tuple[pl.Series, float]: ...

def hawkes_intensity_recursive(
    ts: pl.Series,           # 5-sec bar timestamps (UTC)
    signed_volume: pl.Series,  # aggressor-signed volume per bar
    hl_seconds: float,
    session_boundaries: list[datetime] = None,  # optional warmup resets
) -> pl.DataFrame:
    """Returns (lambda_buy, lambda_sell, imbalance, is_warm) per row."""

def vpin_volume_buckets(
    sub_bars: pl.DataFrame,
    bucket_size: int,
    aggressor_col: str = "aggressor_sign",
    keep_partial: bool = False,
) -> pl.DataFrame:
    """Returns bucket-closed events with (ts_close, vpin, bucket_volume, n_trades).
    Caller asof-joins backward to 15-min bars."""

def cvd_with_dual_reset(
    bars: pl.DataFrame,
    globex_reset_utc: tuple[int, int] = (22, 0),   # Sun 18:00 ET = 22:00 UTC
    rth_reset_utc: tuple[int, int] = (13, 30),     # 09:30 ET = 13:30 UTC
) -> pl.DataFrame:
    """Adds cvd_globex, cvd_rth, bars_since_rth_reset columns."""

def rolling_rth_bounded(
    series: pl.Series,
    window: int,
    bars_since_reset: pl.Series,
    min_bars: int = 5,
    agg: Literal["max", "min", "mean", "std"] = "max",
) -> pl.Series:
    """Rolling aggregate that dynamically bounds at RTH reset (see §8.D)."""

def round_number_pin_distance(close: pl.Series, N: int) -> pl.Series: ...
```

---

## Unit test checklist (pytest, catches §8 silent bugs)

- [ ] `ffd_weights(d=0.3)` weight length consistent with tau truncation
- [ ] `fracdiff_series(d=1)` on AR(1) ≈ first-difference
- [ ] `fracdiff_series(d=0)` ≈ identity
- [ ] `fracdiff_auto_d` picks smallest d passing ADF on synthetic I(1) + noise
- [ ] `hawkes_intensity_recursive` decays to <0.01 after 5×HL bars of zero volume
- [ ] Hawkes handles missing bars (Δt=15s, not 5s) without phantom intensity
- [ ] `hawkes_is_warm = False` for first 5×HL bars
- [ ] `vpin_volume_buckets` on all-buys synthetic → VPIN=1; on equal buys/sells → VPIN=0
- [ ] VPIN partial-bucket discard verified
- [ ] `cvd_rth` starts at 0 at 09:30 ET exactly
- [ ] `rolling_rth_bounded` returns NaN for first `min_bars` bars after reset
- [ ] `rolling_rth_bounded` window effective shrinks from W→bars_since_reset as session progresses
- [ ] Cross-market asof-join excludes exact-timestamp matches (strict `<`)
- [ ] `round_number_pin_distance(close=5000, N=50)` = 0; at 5025, = 25; at 5049.75, = 0.25

---

## Final feature count

| Tier | V1 count | Notes |
|---|---|---|
| T1 microstructure + liquidity | 47 | +17 vs v1 of this doc |
| T2 time-series / vol | 29 | unchanged |
| T3 cross-asset lead-lag | 13 | +2 (Cross-Market DeepOFI) |
| T4 cross-sectional equity | **deferred V1.5** | restricted to 11 SPDR ETFs when enabled |
| T5 vol/options regime | 16 | 8 VX-book features (own data) + 8 dealer-GEX features (WRDS optionm) |
| T5b term structure | **0** (parked) | re-enabled for CL/NG/ZN in V3+ |
| T6 temporal/event | 8 | sin/cos removed |
| T7 patterns | 12 | +2 (CVD divergence, Round-num pin) |
| **Total V1 base** | **125** | before variant expansion |
| Post variant × transform | ~400–550 | |
| Post ILP (ρ<0.45) | ~40–80 | |

---

## Open items (non-blocking, can discuss later)

1. **SpotGamma / Menthor Q** intraday-GEX subscription — evaluate after V1 lands and we see EOD-GEX IC. If material, upgrade to intraday GEX in V2.
2. T4 Tier 4 enable gate — what net Sharpe lift threshold on SPDR-only triggers adding them to V1.5?
3. Alternative-bar experiments (dollar bars, volume bars, imbalance bars) — Phase 2 ablation?
4. Meta-labeling feature set — inherit all 40–80 from primary, or curate a smaller explicit list?
5. Customer-flow calibration for GEX dealer-sign assumption — V2 refinement once V1 signs off.

All non-blocking for the build. Starting work on the 5-sec bar builder + SPX chain WRDS pull next.
