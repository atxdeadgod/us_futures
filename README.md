# us_futures

Intraday futures stat-arb research and production code.

- **Design (long-term vision)**: see [`DESIGN.md`](DESIGN.md) — 30-contract cross-asset, Book A + Book B
- **Phase 1 (first milestone)**: see [`PHASE1_DIRECTIONAL_ES.md`](PHASE1_DIRECTIONAL_ES.md) — single-instrument directional ES
- **Universe**: see [`configs/universe.yaml`](configs/universe.yaml)
- **Upstream**: builds on the equity stat-arb pipeline in `tognn_us` (cross-sectional IPCA + Soyster) and the feature library in `feature-factory`
- **Data**: Algoseek MBP-10 depth + TAQ, 2020–2025; sourced locally from the Expansion drive (not checked in)

## Layout

```
us_futures/
├── DESIGN.md           # full design document
├── README.md
├── configs/            # universe, feature configs, IPCA configs
├── src/                # ingestion, features, panel, IPCA, optimizer, backtest
├── docs/               # supplementary notes, dashboards
└── data/               # local caches (gitignored)
```
