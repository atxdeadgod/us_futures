# Phase C — Trade Manager Design (Skeleton)

Status: design notes / spec, NOT implemented. To be built after Phase A (primary
classifier) and Phase B (magnitude regressor) are working.

The trade manager sits BETWEEN model inference and the executor. The model
outputs per-bar predictions; the trade manager owns position state and decides
when to enter, hold, flip, or exit. Its job is to translate per-bar accuracy
into stable, cost-aware live trading.

---

## The problem this layer solves

The primary model is stateless and reacts to every 15-min bar. In normal vol
regimes, ~30-40% of bars produce a "label predict" of one direction or the
other. Many of those flips are within the natural noise of the price
process — small whipsaws that flip P(+1) > P(-1) one bar and P(-1) > P(+1)
the next.

Without a state-aware policy on top, the strategy:
- Trades on every flip → enormous transaction cost burn
- Loses to slippage on every reversal (each flip is a round-trip cost ~0.5pt ES)
- Has unstable PnL even when the model is correct on average

With a noise-gated state machine:
- Position only flips when the flipping signal exceeds a vol-scaled threshold
- Whipsaws within noise are absorbed (position holds)
- Real reversals go through (when signal magnitude clears the noise floor)
- Net: model accuracy translates into clean, durable position runs

---

## Layered architecture

```
Labels (DONE) ✅
  ↓
Feature library (TC variants, overnight, smoothed)
  ↓
IC dashboard + ILP feature selection
  ↓
Primary classifier (LightGBM 3-class, calibrated)        Phase A
  ↓
Magnitude regressor (LightGBM, predict |ret|/TC-ATR)     Phase B
  ↓
Per-bar inference emits: (P(+1), P(-1), P(0), pred_magnitude, TC-ATR)
  ↓
TRADE MANAGER STATE MACHINE                              Phase C  ← this doc
  ↓
Backtest + cost model
  ↓
Live (IBKR)
```

The trade manager's INPUTS at each bar:
- `P(+1, -1, 0)` — calibrated class probabilities from primary
- `pred_magnitude` — magnitude regressor's predicted |move| in TC-ATR units
- `TC_ATR` — current bar's time-conditional ATR (noise floor)
- `current_position` — internal state ∈ {-1, 0, +1}
- `current_position_age_bars` — bars since last position change
- (Optional) `current_PnL_pct` — for adaptive exit logic

Its OUTPUT at each bar:
- `target_position` ∈ {-1, 0, +1}
- (V2: continuous `target_position` ∈ [-1, +1] for size scaling)

---

## State machine

States: `LONG`, `SHORT`, `FLAT`. Transitions:

```
                      ┌──────────────────┐
                      │       FLAT       │
                      └──┬─────────────┬─┘
                         │             │
                         │ enter_long  │ enter_short
                         ↓             ↓
                ┌─────────────┐   ┌─────────────┐
                │    LONG     │   │    SHORT    │
                └──────┬──────┘   └──────┬──────┘
                       │                 │
                       │  flip_to_short  │
                       └─────→  ←────────┘
                       │  flip_to_long   │
                       └─────  ──────────┘
                       │                 │
                       │  exit           │ exit
                       └──→ FLAT ←───────┘
```

### Transition rules

**FLAT → LONG**: `P(+1) > tau_high_enter` AND `pred_magnitude > k_enter * TC_ATR`

**FLAT → SHORT**: `P(-1) > tau_high_enter` AND `pred_magnitude > k_enter * TC_ATR`

**LONG → FLAT**: `P(+1) < tau_low_hold` (conviction collapsed)

**SHORT → FLAT**: `P(-1) < tau_low_hold`

**LONG → SHORT** (noise-gated flip): require ALL of:
- `P(-1) > tau_high_flip`  (high opposite conviction)
- `pred_magnitude > k_flip * TC_ATR`  (move size exceeds noise floor)
- `current_position_age_bars > min_hold_bars`  (don't flip immediately after entering)

**SHORT → LONG**: symmetric.

**Hold (no transition)**: any sub-threshold or same-direction signal.

### Why each rule

- `tau_high_enter`, `tau_high_flip`: classifier confidence required to act.
  Higher than the classifier's "predict this class" threshold (e.g., > 0.5)
  because acting on every >50% prediction is too aggressive.
- `tau_low_hold` < `tau_high_enter`: hysteresis band. Once you're in a
  position, conviction can drop to `tau_low_hold` before exiting. This
  asymmetry keeps positions sticky against minor doubt.
- `k_enter`, `k_flip`: noise multipliers in TC-ATR units. `k_flip > k_enter`
  because flipping costs 2 round-trips (exit + reverse) vs 1 for entry from
  flat. The magnitude gate ensures the flip is worth the cost.
- `min_hold_bars`: prevents pathological 1-bar position duration. Forces the
  model to commit briefly even if the next bar reverses.

### Hyperparameters (to be tuned per instrument)

| Parameter | Initial guess | Range |
|---|---|---|
| `tau_high_enter` | 0.55 | 0.50–0.65 |
| `tau_low_hold` | 0.45 | 0.40–0.50 |
| `tau_high_flip` | 0.60 | 0.55–0.70 |
| `k_enter` | 1.0 | 0.5–2.0 |
| `k_flip` | 1.5 | 1.0–3.0 |
| `min_hold_bars` | 2 | 1–8 |

Tune via grid search on the IS window, evaluate by:
- Net PnL after costs
- Trade count (lower is better, all else equal)
- Sharpe ratio
- Max drawdown
- Average position holding time

---

## Why this is separate from the model

If we baked stability into the model (e.g., regularize predictions to match
recent predictions), we'd conflate two distinct concepts:
- "model is uncertain about direction this bar" (real epistemic uncertainty)
- "model wants to flip but is being held back to reduce churn" (artificial damping)

Separating them keeps the model honest about per-bar predictions while letting
trade-management policy be tuned independently. We can A/B test policies
(different `tau_high`, hysteresis bands, magnitude multipliers) without
retraining models. We can have one policy for low-vol regimes and another for
high-vol — same model.

---

## Design choices that affect upstream phases

To keep Phase C clean we need Phase A (primary) and Phase B (magnitude) to
emit specific things:

1. **Calibrated probabilities from primary.** `P(+1) > 0.55` must mean
   ACTUAL 55% confidence, not "model is moderately confident relative to its
   training distribution." Use Platt scaling or isotonic calibration on a
   held-out fold inside IS.
2. **Magnitude in physical units.** Predict in TC-ATR multiples (e.g., target
   = `|realized_ret_pts| / TC_ATR_at_bar`). Then `pred_magnitude > 1.5` is a
   directly-meaningful "this move will be 1.5× typical-this-hour size."
3. **TC-ATR per-bar carried through inference.** Already computed in the
   labeling pipeline; just need to keep it as a column in the inference
   DataFrame.
4. **All three class probabilities preserved.** Even if `argmax(P) == +1`,
   the manager wants `P(-1)` and `P(0)` separately to compute thresholds.

---

## Continuous-position extension (V2)

For a continuous `target_position ∈ [-1, +1]`:

```
signal_strength = (P(+1) - P(-1)) * pred_magnitude
target_position = clip(signal_strength / k_normalize, -1, +1)

# Trade only if change exceeds dead-band
if abs(target_position - current_position) > dead_band:
    rebalance to target_position
else:
    hold
```

Smoother PnL but trades more often. Defer to V2 once V1 discrete state
machine is validated.

---

## Open questions

- **Adaptive thresholds**: should `tau_high`, `k_flip` adjust to current
  realized regime (e.g., higher in volatile periods)? Initial: no, keep
  static. V1.5 candidate.
- **Time-of-day awareness**: should we restrict ENTRIES to specific hours
  (avoid first 15min after RTH open due to noise) but allow EXITS from any
  hour? Probably yes — most real strategies do this.
- **Cost-of-flip penalty**: include round-trip cost in the magnitude gate
  explicitly: `pred_magnitude * P(direction) > k_flip * TC_ATR + 2 * round_trip_cost_pts`?
- **Daily PnL stop / position limits**: orthogonal to the state machine but
  needed before live trading.

---

## Status

- Phase A (primary classifier): NEXT
- Phase B (magnitude regressor): AFTER Phase A
- Phase C (this doc): designed, not implemented
- Backtest engine: implements this state machine to evaluate policies on IS
- Live: implements same state machine, hooks to IBKR API
