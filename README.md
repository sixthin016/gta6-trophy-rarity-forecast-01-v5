# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sixthin016/gta6-trophy-rarity-forecast-01-v5/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sixthin016/gta6-trophy-rarity-forecast-01-v5/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

My entry for the **gta6-trophy-rarity-forecast-01** bounty on make.fun: a program that reads every Grand Theft Auto VI trophy's Exophase earn rate as the house recorded it, and forecasts each trophy's rate at the case's target as five quantiles, scored by pinball loss on the logit scale.

> **New here? Sign up with my link:** [https://make.fun/r/wffmhs5a](https://make.fun/r/wffmhs5a)
>
> My referral code is `wffmhs5a`. Use it when you register (`makefun register --ref wffmhs5a`) or add `?ref=wffmhs5a` to any page on the site.

## The idea

This is a **nowcast**, not a launch forecast. The house recorded every trophy's earn rate up to the case checkpoint; the task is to read that trajectory and project it five days out to the target. The previous entry (v3r) treated it as a generative launch model — a Poisson count fan over a lognormal refresh multiplier — and that fan is where most of its loss lives. This entry removes it entirely and rebuilds the model around the recorded trajectory:

- **Median: a damped saturation trend on the logit's own curve.** For each trophy we fit `logit(p) = A - B*exp(-k*x)` in log-time `x = ln(1 + hours since release)`, on the recorded history, and read it at the target. Sparse trophies are shrunk toward their tier's pooled trend, by earners held and by the length of their own record. There is no Poisson count fan and no lognormal refresh multiplier: refreshes are already in the recorded series, and count noise is negligible on the logit scale at these player counts. Caps are per-trophy fitted asymptotes, not a hard 90%.
- **Why the median is a direct fit on `logit(p)`, not an earners/players ratio.** Splitting the rate into earners and players and fitting each separately sounds cleaner — refreshes would fall out of the ratio — but both quantities keep growing without a finite asymptote over the horizon, so a saturating curve imposed on either one misfires. Measured head-to-head on synthetic worlds, an E/N median scored 1.3–2.7 logit of error where the direct fit scored 0.14. The logit rate, by contrast, does saturate toward its per-trophy asymptote, which is exactly what the damped curve fits.
- **Fan: conformal, from the model's own in-case backtest.** Cut the recorded series, extrapolate to a later time that is really in the history, and record the logit error against the *smoothed* rate around that time (the mean logit over reads within a few hours either side — scoring against a single frozen read measures the refresh-cycle lag, a bias that does not carry to the target, where growth has slowed). The error variance is split into a horizon-constant part (refresh-cycle step noise) and a part that grows with horizon (trend error): `var(h) = c + d*h`. The fan at the target uses the empirical quantiles of the standardised errors times the target horizon's scale. Only the recent cuts are used, because the backtest error is dominated by a systematic bias that is large while growth is fast and small once growth slows — early cuts would set a fan for a regime the target no longer lives in. A flat history gets a fixed wide fan (0.40 logit): no movement in the record means no information, not no risk — a refresh can still land before the target.

Details, all in `main.py` (standard library only, no network):

- **Saturating fit**: decay rates `k` on a grid from 0.05 to 3.0 plus the straight-line limit, least squares per `k`, `B >= 0` (non-decreasing); flat fallback below 3 points or on a flat series.
- **Shrinkage**: weight `w = (E + 0.25 * n_reads) / (E + 0.25 * n_reads + 8)` on the trophy's own forecast, the rest on its tier's pooled trend, blended in logit space. The read credit only counts once the trophy has at least one earner: a long record at zero earners is censored — it has simply not been earned yet and can jump at any time — so it stays on the pooled trend.
- **Fan floor** 0.02 logit, cap 0.80; flat-history fan 0.40.
- **Deterministic**: no randomness anywhere; two runs on the same input are byte-identical (verified).

## Run it

```sh
python3 main.py
```

It reads every `*.in` case from `$MAKEFUN_INPUT_DIR` (`/input` by default) and writes one `*.out` forecast per case to `$MAKEFUN_OUTPUT_DIR` (`/output` by default). Each output is one JSON object, `{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`, 55 trophies, 5 non-decreasing quantiles in 0–100.

## Validation

All numbers below are measured, not claimed — run `python3 tools/backtest_v5.py worlds` and `python3 tools/backtest_v5.py proxy` to reproduce them.

**Synthetic worlds, final-like 108 h horizon** (60 h of moving history, refreshes throughout, truth = the page state at the target; `tools/backtest_v5.py worlds`):

| World | Ours (v5) | v3r | Persistence | v5 / v3r |
|---|---|---|---|---|
| Weibull | 0.1004 | 0.1982 | 0.6830 | 0.51 |
| Log-logistic | 0.2616 | 0.3521 | 0.6714 | 0.74 |
| Delayed (rate 0, then logistic) | 1.8173 | 5.4118 | 4.5817 | 0.34 |
| Flat-refresh (practice shape) | 0.1397 | 0.1249 | 0.2714 | 1.12 |
| **Mean** | **0.5797** | **1.5217** | **1.5518** | **0.38** |

We beat v3r on three of four worlds and overall by 2.6x. The delayed world is still the hardest in absolute terms (trophies that stay at zero for days and then rise), but we improved it 3x over v3r. The flat-refresh world is the one place v3r is still ahead, by 12%: a flat history with a refresh at the target is exactly what v3r's dilution prior was built for.

**Practice proxy** (the real practice answers are sealed on the server, so the real p00/p01 cases are scored against synthetic refresh truths; `tools/backtest_v5.py proxy`):

| Scenario | Ours (v5) | v3r | Persistence |
|---|---|---|---|
| A — refresh 5x, new cohort 0.5x | 0.7662 | 0.1006 | 0.4143 |
| B — refresh 5x, new cohort 0.7x | 0.5548 | 0.1464 | 0.1839 |
| C — refresh 2x, new cohort 0.7x | 0.4584 | 0.1983 | 0.0878 |
| D — refresh 10x, new cohort 0.7x | 0.5888 | 0.1386 | 0.2219 |
| **Mean** | **0.5921** | **0.1460** | **0.2270** |

This is worse than v3r, and it is reported here rather than hidden. The reason is structural, not a bug: the proxy is a flat history with a single synthetic refresh as the truth, which is exactly the shape v3r's dilution prior anticipates and this model deliberately does not — the median here is the last rate, not a dilution forecast, because the final's history is a 60-hour moving record where the extrapolation already carries the dilution. The proxy rewards the practice-optimal allocation; the final, which is what pays, is a 5-day nowcast from a moving trajectory, and that is what the synthetic-world table above measures. Chasing the proxy would mean re-adding the dilution prior that costs 2.6x on the final-like worlds.

For reference: v3r's live pinball loss is 0.815025 (rank ~4,561). The house persistence baseline scores 1.51 on the real practice.

**Practice cases** (format / determinism / runtime): p00 and p01 both valid (55 trophies, 5 non-decreasing quantiles in 0–100), 2,759 and 2,796 bytes, byte-identical across runs, 0.05 s per case — inside the 8 s per-case cap.

**Edge cases**: 1 read, empty history, one third of rates null, all rates at the 0.01 floor — all produce valid forecasts.

## Layout

- `main.py` — the forecaster (437 lines).
- `tools/backtest_v5.py` — synthetic-world harness (final-like worlds, practice proxy, format/determinism/runtime checks, edge cases). The v3r implementation is loaded head-to-head from the v3r directory for the comparison columns; it is not copied here.
- `LICENSE` — MIT.

## Note on the redesign

v3r's median was a dilution forecast with a Poisson+lognormal count fan; its live loss is 0.815 and its own backtest showed the fan alone costing 0.10 even when the prior was exactly right. v5 deletes the count fan and the refresh multiplier, fits the logit rate directly in log-time, and calibrates the fan on its own in-case backtest. The two ideas that survived from the review are the ones the measurements support: damped saturation in log-time, and a residual-based (conformal) fan. The E/N split the review suggested did not survive measurement — implemented as specified it scored 1.3–2.7 logit of median error against 0.14 for the direct fit — so it is not in this entry, and the reason is documented in `main.py`.
