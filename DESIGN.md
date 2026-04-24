# Futures Intraday Stat-Arb — Design Document

Status: draft, 2026-04-23. Prepared alongside initial repo setup.

**Modeling convention (carries across phases)**: primary model trained on **triple-barrier labels** (AFML Ch. 3) with **meta-labeling** on top for bet-size gating; tuning via **walk-forward Bayesian optimization with purging + embargo**; 2025 is the holdout fold, untouched during any parameter selection. Details in [`PHASE1_DIRECTIONAL_ES.md`](PHASE1_DIRECTIONAL_ES.md) §3.

---

## 0. Key translation: daily → intraday

The equity stat-arb pipeline (`tognn_us`) labels returns at 1/2/3/5-**day** horizons, retrains IPCA yearly, rebalances daily. For multi-intraday trading on futures we keep the **panel + IPCA + Soyster** architecture and change three things:

| Dimension | Equity (current `tognn_us`) | Futures (proposed `us_futures`) |
|---|---|---|
| Bar frequency | 1 day | 15-min primary; 5-min bars for microstructure feature inputs |
| Cross-section axis | ~2000 stocks × daily | ~30 liquid contracts × 15-min bar |
| Label horizon | 1d / 2d / 3d / 5d returns | 15-min / 30-min / 60-min / 4-hour forward returns |
| IPCA refit cadence | yearly expanding window | monthly or quarterly expanding window |
| Rebalance | daily close | every 15-min bar in RTH (26 rebalances/day) |
| Residualization | sector β, cap-weighted | asset-class β (equity idx / rates / energy / ag / metals / FX / STIR) |

**The 864-feature dashboard's t-stats in `tognn_us` do not transfer directly** — those are daily-IC t-stats on equities. We rerun the same style of dashboard on futures bars; most *feature construction* carries over but statistical significance must be re-established in the new domain and horizon.

### Constraint check
- No co-location → **exclude** pure market-making, queue-position-aware passive, sub-second lead-lag arb, iceberg/sweep detection as offensive alpha, pure latency arb.
- MBP-10 available → **include** multi-level OFI at bar horizons, top-of-book microstructure features at minute aggregation.
- Algoseek pre-classifies trade aggressor → OFI/VPIN-adjacent features are cleaner than standard tick datasets (no Lee-Ready step).

---

## 1. Book A — Cross-sectional panel stat-arb

**Goal**: exploit short-horizon predictability *across* a liquid futures panel using factor-style alphas. Direct extension of the existing equity pipeline.

### 1.1 Architecture

```
Algoseek TAQ + MBP-10
   → polars ingestion (symbol|exchange prefixed, per feature-factory spec)
   → raw bars (15-min OHLCV + aggregated book stats) + tick features
   → feature-factory single features (L1/L2/bar)
   → feature-factory cross features (panel)
   → residualize (asset-class + front-contract β)
   → ema / slope / rank variants (preprocess_for_ipca-equivalent)
   → IPCA (K=3–5, expanding window, monthly refit)
   → alpha signal per bar per contract
   → Soyster robust optimizer w/ vol target + bar-level gross cap
   → execute at next bar open
```

### 1.2 Signal table — elaborated

Each row = one alpha family contributing features to the panel. The panel is fed to IPCA; IPCA finds which latent factors explain forward returns. You don't trade any single signal — you trade the IPCA-combined alpha.

| # | Signal family | What it measures | Inputs | Bar size used | Forward horizon it predicts | Feature-factory classes (ready to use) | Transfer effort | Expected IC contribution |
|---|---|---|---|---|---|---|---|---|
| **A1** | **Intraday cross-sectional seasonality** (Heston-Korajczyk-Sadka) | Cross-sectional return at lag k×13 bars (1 day) and k×26 (2 days); same-time-of-day return predicts same-time-of-day | Mid-price returns at 30-min resolution | 30-min | 30-min fwd | `IntradaySeasonalDeviation` (bar/single), `CrossSectionalReturnZScore` (bar/cross) | direct port | moderate; documented in equities, likely transfers to index/commodity futures |
| **A2** | **Intraday time-series momentum** (Gao et al.) | First-30-min return predicts last-30-min; also short-term trend per contract | 15/30-min log returns | 15-min | 60-min / close | `Momentum` (l2), `LogReturn` + `ReturnAutocorrelation` (bar), `TrendStrength` (l2) | direct | strong in CL/NQ/copper per lit |
| **A3** | **Multi-level order flow imbalance (deep OFI)** | Net signed order flow across book levels 0–9 over short window; predicts near-term mid-price drift | MBP-10 (ΔL1...ΔL10 bid/ask sizes + prices) | 1-min feature → aggregated to 15-min | 15–30 min fwd | `OrderFlowImbalance` (bar/agg) + `AggregateImbalance` / `DistanceWeightedImbalance` (l2/single); Kolm-style deep OFI needs a new `DeepOFI` class | **mostly ready** — need one new class combining L1..L10 signed ΔQ | highest standalone IC of microstructure group |
| **A4** | **Minute-level lead-lag** | Lagged returns from leader contracts (ES, CL, ZN, GC) predict laggards in same asset class | Mid returns, 1-min aligned | 1-min feature → 15-min | 5–30 min fwd | `BarLeadLagReturn` (bar/cross), `LeadLagReturn` (l1/cross), `PriceLeadLag` (l2/cross) | direct | moderate within asset classes, weakens across |
| **A5** | **Term-structure features** (per root) | Front-deferred basis, curve slope, curvature, basis-momentum, roll yield | Multiple expiries per root | daily + intraday | 15-min → multi-day | *new:* `BasisSlope`, `BasisMomentum`, `CurveCurvature`, `RollYield` classes (extend `Spread*` family) | **new work** (~4–5 classes) | unique to futures; important for commodities and rates |
| **A6** | **Realized vol features** | HAR-RV, range-based (Parkinson, GK), vol-of-vol, up/down vol | Tick returns, OHLC | 5-min → 15-min | vol gating + mild return prediction | `RealizedVolatility` (+Std/Ewma), `RangeBasedVolatility`, `VolatilityOfVolatility`, `UpVolatility`, `DownVolatility`, `TickVolatility` | direct | small standalone, big sizing overlay |
| **A7** | **Aggressor imbalance** (signed trade flow) | Net buy-initiated vs sell-initiated volume; Algoseek pre-classifies so no Lee-Ready needed | TAQ `Type=TRADE AGRESSOR ON BUY/SELL` | 1-min → 15-min | 15-min fwd | `AggressorSideVolume` (bar/agg), `BarVolumeImbalance`, `DollarImbalance` (l2) | direct; Algoseek flag makes it cleaner than equity version | reliable across all futures |
| **A8** | **Event-window conditioners** | Dummies for FOMC/NFP/CPI/EIA-release windows (±30-min); macro calendar integration | Macro calendar JSON | feature is bar-level | gate/scale A1–A7 | *new:* `EventWindowFlag` class + Fed/BLS/EIA calendar ingestion | new (~1 class + calendar file) | not alpha by itself; improves risk control 20–30% |

**Why IPCA still works here**: the panel has instruments × time × features; IPCA reduces it to K latent factors whose loadings are linear in the features. On futures this is cleaner because contracts are far more homogeneous within asset class than stocks are. Expect **K=3 to remain right**, possibly K=4 with term-structure features added.

### 1.3 Feature budget for Book A

Starting inventory from `feature-factory`:
- **~32 bar-single features** (RV, autocorrelation, Parkinson/GK, skew/kurt, momentum, seasonality dummies, Amihud, VWAP return, overnight/intraday split, turnover, jumps)
- **~17 bar-cross features** (FactorModelResidual, CointegrationZScore, IdiosyncraticVolatility, ReturnDispersion, ReturnConcentration, OUHalfLife, BarLeadLagReturn, LeaderLaggardSpread, ResidualBandPositionStretch, ResidualDispersion, ResidualShockRepair, CrossSectionalReturnZScore/StdDev, ETFLeadBeta, ETFShockFollowThrough, Breadth, LeadLagAsymmetryScore)
- **~22 bar-aggregation features** (OrderFlowImbalance, AggressorSideVolume, BidAskDepthRatio, LargeTradeVolumeShare, LiquidityMigration, QuoteToTrade, SideConditionedLiquidityShift, ThresholdCrossings, VolatilityTriggered, QuoteMovementDirectionality, etc.)
- **~17 L1-single features** (Microprice, MicropriceDrift, spread family, up/down vol, jumps)
- **~90 L2-single features** (imbalance family 20+, spread family 20+, technical-indicator family ~30, rolling-stat family ~15)
- **~13 L2-cross features** (CrossCorrelation, DepthImbalanceDiff, HedgeRatio, InformationShare, LiquidityRatio, MicropriceDiff, OFICorrelation, PairsSpreadZScore, PriceLeadLag, RealizedVolatilityRatio, RelativeQuotedSpread, RollingBeta, HalfLifeMeanReversion)
- **yjia subpackage**: 13 research families (Auction, Corr, Flow, InformedPriceLevel, LiqTurn, MomRev, Moments, Shape, Shock, TradeCount, VolHF, VolumeShare) — inspect for extras

**Total: ~200+ feature classes already implemented.** Parametrized by depth / window / lag, so easily >500 candidate features once grid-expanded.

**New classes to add (~7):**
1. `DeepOFI` (L2 single) — signed ΔQ across all 10 levels, weighted by depth distance
2. `BasisSlope` (bar cross, within root) — front vs back contract log-ratio
3. `BasisMomentum` (bar cross) — change in basis slope over k bars
4. `CurveCurvature` (bar cross, needs ≥3 expiries) — 2nd-difference across the strip
5. `RollYield` (bar single) — annualized basis as carry
6. `EventWindowFlag` (bar single) — macro calendar dummy
7. `TermStructureResidual` (bar cross, rates) — deviation from fitted NS curve

### 1.4 Expected output for Book A

- 150–250 feature×variant candidates
- Dashboard v2 on futures bars (ic / t / icir / hit at 15/30/60/240-min horizons) — futures analog of `tognn_us/data/feature_dashboard_v2_sector_beta.csv`
- ILP selection → ~40–80 survivor features (ρ<0.45)
- IPCA K=3, monthly refit, 2023–2025 OOS
- Target net Sharpe **2.5–3.5** after realistic costs

---

## 2. Book B — Spreads sub-book

**Goal**: harvest mean-reversion in structural spreads. Different math (cointegration / PCA residual + z-score), separate signal stack, same execution infra.

### 2.1 Architecture

```
For each spread family:
  define legs & weight scheme (static or Kalman-filtered)
  compute spread series (using native CME ICS where available)
  fit half-life / Ornstein-Uhlenbeck params on rolling window
  z-score signal
  entry at |z| > z_in; exit at |z| < z_out; stop on half-life × k timeout
  separate position-sizing: per-spread vol target
```

No cross-sectional panel here — each spread is a standalone model. Reuses feature-factory classes: `PairsSpreadZScore`, `HalfLifeMeanReversion`, `HedgeRatio`, `CointegrationZScore`, `OUHalfLife`, `InformationShare`.

### 2.2 Spread table — elaborated

| # | Spread family | Legs (weights) | Weighting method | Signal | Holding | Why this survives | Algoseek coverage |
|---|---|---|---|---|---|---|---|
| **B1** | **Treasury curve flies** | FYT (+3 ZF / −2 ZN); NOB (ZN vs ZB); PCA-residual 2-5-10 (ZT/ZF/ZN); 5-10-30 (ZF/ZN/ZB); TN inclusions | DV01-neutral (static); PCA-residual (rolling 60-day) | Z-score of curvature residual vs OU mean | minutes–hours | Exchange-defined ICS exist (FYT, NOB, TUT, TUB, FOB, TYB); curve mean-reversion post-auction; $100M+ capacity | Full MBP-10 on all 5 legs |
| **B2** | **Crush (ZS-ZM-ZL)** | −1 ZS, +1 ZM, +1 ZL (×unit-conversions) | Static (industry-standard crush margin) | Z-score of margin vs rolling mean; asymmetric exits (winners shorter than losers per Rechner) | hours–days | Native CME crush ICS; structural processing relationship; 2.38 Sharpe net reported | All 3 legs |
| **B3** | **Crack (CL-RB-HO 3-2-1)** | +3 CL, −2 RB, −1 HO | Static or slow rolling β | Z-score of crack residual | hours–days | Native ICS; seasonal (summer gasoline / winter heating) | All 3 legs |
| **B4** | **SOFR pack butterflies** | Red-Green-Blue pack fly; or (SR3 Y1 − 2×Y2 + Y3) | Static pack weights | Z-score of pack fly vs rolling mean + Fed-calendar gating | intraday to days | Native CME Pack Butterfly (launched 2024-07-29); cleaner than legging | SR3 depth available |
| **B5** | **WTI-Brent (CL-BZ)** | +1 CL, −1 BZ | Kalman-filtered hedge ratio (1:1 can drift) | Z-score of residual, 40–80d window | days | Structural (Cushing vs waterborne); widens on geopolitics, reverts | Both legs |
| **B6** | **Gold-Silver (GC-SI)** | Ratio or Kalman β | Rolling β | Z-score ratio | days–weeks | Long-run cointegration; useful diversifier | Both legs |
| **B7** | **Calendar spreads — MAKER ONLY** | CL M1-M2, NG M1-M2, etc. | Native calendar ICS | Bollinger z-score | minutes | Only if posting passive for rebate; taking is negative net Sharpe | Yes, but execution-infrastructure-limited; defer |

**Deliberately skipped**:
- ES-NQ-YM-RTY "pairs" — just risk-on/off beta, not structural
- Goldman roll front-running — edge degraded post-2012, not intraday anyway

### 2.3 Execution note for Book B

Always prefer the **exchange-defined spread** on Globex (ICS, UDS butterflies, pack flies). Implied pricing populates both the spread book and the outright legs, so fills are cleaner and leg risk is eliminated. Never leg in manually for B1–B4.

### 2.4 Expected output for Book B

- 6 live spread models initially (B1–B6; B7 deferred until maker infra)
- Each independently sized to hit ~10–20% of book-level vol
- Per-spread target net Sharpe 0.8–2.0
- Combined (if correlations stay low): net Sharpe **1.5–2.5**

---

## 3. Combined book

- Book A and Book B run in parallel, target correlation **ρ < 0.3** (structurally likely — panel factor alphas vs curve residuals)
- Capital allocation: start 70% A / 30% B; re-weight by realized Sharpe after 6 months live
- Combined target: net Sharpe **3–4**, Sortino 4–5, max DD 8–12%
- Vol target at book level (same Soyster machinery, one level up)

---

## 4. Phased plan

### Phase 0 — Setup (2 weeks)
- Create `us_futures` repo (done)
- Lock trading universe (see [`configs/universe.yaml`](configs/universe.yaml), 30 contracts)
- Ingest pipeline: Algoseek TAQ + depth → polars → prefixed-column parquet per (date, symbol)
- Decide bar grid: **15-min primary**; 5-min for microstructure feature inputs
- Macro calendar ingest (FOMC, NFP, CPI, EIA, ISM)

### Phase 1 — Book A skeleton on bar features only (4 weeks)
- Bar generation from TAQ (OHLCV + aggressor-side volume from Algoseek trade flags)
- Feature-factory wiring: bar-single + bar-cross + bar-aggregation features
- Add new classes: `BasisSlope`, `BasisMomentum`, `CurveCurvature`, `RollYield`, `EventWindowFlag` (~5 classes)
- Run dashboard v2 on futures bars → futures-IC t-stats
- Stage-2 ILP selection → ~40 features
- IPCA K=3 → Soyster backtest 2023–2024 IS, 2025 OOS
- **Checkpoint**: net Sharpe ≥ 2.0 on bar-only features? If not, diagnose before adding L2.

### Phase 2 — Add L2 microstructure to Book A (4–6 weeks)
- MBP-10 ingest (larger data volume; decide cache strategy on slate vs local)
- Compute L2 single features at 1-min then aggregate to 15-min bars
- Add `DeepOFI` class (multi-level OFI)
- Add L2 cross features (DepthImbalanceDiff, OFICorrelation, InformationShare, PriceLeadLag)
- Re-run dashboard + ILP + IPCA (K=3 or 4)
- **Checkpoint**: Sharpe lift from L2 features ≥ 0.5? If not, reconsider L2 cost.

### Phase 3 — Book B (3 weeks, parallel to Phase 2)
- Implement B1 (Treasury flies) first — highest expected Sharpe, clean ICS data
- Add B2/B3 (crush, crack), B4 (SOFR), B5 (WTI-Brent), B6 (Gold-Silver)
- Each independently backtested; gate on min Sharpe 0.8 before including
- Skip B7 until maker infra exists

### Phase 4 — Execution realism & combined book (2 weeks)
- Cost modeling (half-spread, fees, impact — 1.5–3× backtest naive)
- Stop slippage distributions fit from Algoseek data
- Combined optimizer: Book-A weights + Book-B positions jointly optimized at book level
- Paper-trade simulation before live
- **Target checkpoint**: net combined Sharpe 3+ on 2025 OOS

### Phase 5 — Ongoing enhancements
- Vol/options overlay (UX term structure, GEX flip-level gating)
- Event-specific overlays (pre-FOMC scaling)
- MBO data if budget allows (enables queue-position passive execution for Book B calendar spreads → unlocks B7)
- Meta-labeling (AFML Ch. 3) on primary signal for bet-size conditioning

---

## 5. Open decisions

Picking these unblocks the rest of Phase 0:

1. **Sharpe target lock**: proposed **3–4 net combined** (realistic). Confirm or override.
2. **Bar size**: **15-min primary** (26 bars RTH). Alternatives: 5-min (finer, more overfit risk) or 30-min (coarser, fewer trades).
3. **Capital scale**: affects which signals are capacity-constrained. $10M? $100M?
4. **Data staging**: process locally (Expansion drive direct) vs push to Quartz/slate first?
5. **2025 depth data**: TAQ covers through 2025-08-25 but depth stops 2024-12-31. Plan to extend MBP-10 into 2025 before Phase 2.
