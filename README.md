# PFE Multi-Signal Trading Strategy (`Final_1.ipynb`)

A daily long/short strategy on **Pfizer (PFE)** that builds two independent families of signals, turns each into a ±1 position, and then blends the two positions into a final trade signal. All weights and thresholds are tuned with [Optuna](https://optuna.org/) on a training window and then evaluated out-of-sample.

## Data and splits

- Source: `yfinance`, daily, 2010-01-01 → 2025-04-16 (unadjusted `Close`, plus `Volume` for Phase 1).
- Phase 2 also downloads competitors (`JNJ, MRK, AZN, GSK, LLY, ABBV, BMY`) and sector ETFs (`XLV, IHE`).

| Split | Period | Use |
|---|---|---|
| Train | 2010-01-01 → 2018-12-31 | Fit weights/thresholds |
| Validation | 2019-01-01 → 2022-12-31 | Split defined, not used in the notebook |
| Test | 2023-01-01 → end | Out-of-sample evaluation |

**Look-ahead protection:** every input series (price, volume, all tickers) is shifted by one day, so the signal for day *t* only uses data up to *t-1*. P&L is computed on the unshifted `Price OG`.

## Phase 1 – Price/volume signals (PFE only)

Each signal is normalised (rolling z-score, clipped) to roughly **[-1, 1]**.

| Signal | Idea |
|---|---|
| `vwap` | Z-score of the 5-day VWAP minus 20-day VWAP (short vs. long volume-weighted trend). |
| `momentum` | 5-day price change scaled by volume relative to its 20-day average, z-scored over 30 days. |
| `divergence` | Product of z-scored 10-day price slope and volume slope: positive when price and volume agree, negative when they diverge. |
| `breakout` | Distance beyond the prior 20-day high/low (in units of rolling std), weighted by the volume percentile (only top ~20% volume days count fully). |
| `obv` | Z-score of On-Balance Volume vs. its 14-day mean. |
| `vpt` | Z-score of Volume-Price Trend vs. its 14-day mean. |

**Volume regime overlay** (`regime_enhance`): volume > mean + 1 std over 50 days is "high", < mean − 1 std is "low". `momentum` is multiplied by 1.5 / 0.5 and `breakout` by 1.3 / 0.6 in high / low regimes (then re-clipped to [-1, 1]).

## Phase 2 – Competitor and sector signals

Computed from daily returns of PFE, its peers and the healthcare ETFs.

| Signal | Idea |
|---|---|
| `rc` (rolling correlation) | 20-day correlation of PFE with XLV/IHE. In sector uptrends low correlation is bullish; in downtrends high correlation is bearish. |
| `rss` (relative strength) | PFE's 20-day mean return vs. peer average and vs. sector average, standardised and squashed with `tanh`. |
| `coin` (cointegration pairs) | For each peer, a 60-day Engle-Granger test; if p < 0.05, trade the PFE–peer price spread z-score as mean reversion (`-tanh(z/2)`), averaged across cointegrated peers. |
| `sector` (rotation) | 30-day PFE momentum relative to sector momentum, averaged with PFE's percentile rank among peers. |
| `beta` (beta-adjusted) | Residual of PFE's latest return vs. what its 60-day beta to XLV implies, traded as mean reversion. |
| `vr` (volatility regime) | PFE's 30-day volatility relative to peers/sector, inverted (relatively calm = bullish). |

## How signals are combined

Combination happens in two stages.

### Stage 1 – within each phase (`getFinalSignal`)

```
score_t   = Σ  w_i · signal_i,t          (w_i ≥ 0, Σ w_i = 1)
position_t = +1 if score_t > t else -1
```

Optuna (`getOptim`, 1000 trials, TPE sampler) searches the six weights (sampled in [0,1] and normalised to sum to 1) and the threshold `t ∈ [-1, 1]`, maximising **total return** on the training set. This yields `fs1` (Phase 1 positions) and `fs2` (Phase 2 positions), each a ±1 series.

### Stage 2 – across the two phases (`getCombinedSignal`)

```
combined_t = w[0] · fs1_t + w[1] · fs2_t
position_t = +1 if combined_t > t else -1
```

A second Optuna study (`getOptim3`, 500 trials) tunes the two blend weights (normalised to sum to 1) and threshold `t`, this time maximising the **Sharpe ratio**. Because `fs1` and `fs2` are ±1, the blend acts as a weighted vote: if the two agree the position follows them; if they disagree, the higher-weighted model wins, with `t` shifting the bias toward long or short.

### Backtest mechanics (`ApplyStrategy`)

- Always invested: fixed 1,000,000 notional, long (+1) or short (−1) each day.
- Daily P&L = `ΔPrice / Price_{t-1} × 1,000,000 × position`, accumulated additively (no compounding).
- Reported: total return multiple, annualised Sharpe (√252), CAGR, max drawdown.
- Benchmarks plotted: always-long daily-rebalanced, and buy & hold.

## Phases of the notebook

1. **Phase 1** – build and optimise price/volume strategy on train data.
2. **Phase 2** – build and optimise competitor/sector strategy on train data.
3. **Phase 3** – optimise `(w1,t1)`, `(w2,t2)`, then the blend `(w,t)`; run `simulate3`.
4. **Phase 4** – freeze all learned parameters and run `test()` on the 2023+ data.

## Running it

```bash
pip install pandas numpy matplotlib yfinance scikit-learn statsmodels scipy optuna
jupyter notebook Final_1.ipynb
```

Run cells top to bottom (the cointegration and beta signals use Python loops and the Optuna studies take a while).

## Caveats

- Stage 1 optimises total return while Stage 2 optimises Sharpe; this is intentional in the code but worth knowing when comparing numbers.
- The validation split is defined but never used for tuning or early stopping, so overfitting to the train window is controlled only by the final test set.
- No transaction costs, slippage or borrow costs are modelled, and the strategy flips between full long and full short daily, so real-world results would be lower.
- Unadjusted prices ignore dividends in P&L.
- Optuna results vary run to run unless a sampler seed is set.
