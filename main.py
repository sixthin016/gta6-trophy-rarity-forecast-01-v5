"""gta6-trophy-rarity-forecast-01 entry, v5: recorded-trajectory nowcast.

The task is a nowcast, not a launch forecast: read every trophy's earn
rate as the house recorded it up to the case checkpoint, and forecast
each rate at the case target (5 days out for the final) as five
quantiles, scored by pinball loss on the logit scale.

Model, in one paragraph.  For each trophy we fit a damped saturation
trend to its logit earn rate in log-time, on the recorded history, and
read it at the target; sparse trophies are shrunk toward their tier's
pooled trend by earners held.  The median is that extrapolation,
re-centred by nothing else.  The fan is not a prior: it is conformal,
taken from the model's own in-case backtest (cut the recorded series,
extrapolate, score the logit error), with the error variance split
into a horizon-constant part (refresh-cycle step noise) and a part
that grows with horizon (trend error), and a floor so the fan can
never collapse on a flat history.  There is no Poisson count fan and
no lognormal refresh multiplier: refreshes are already in the
recorded series, and count noise is negligible on the logit scale at
these player counts.  Caps are per-trophy fitted asymptotes, not a
hard 90%.

Why the median is a direct fit on logit(p) and not an E/N ratio:
the earners and the player count each keep growing without a finite
asymptote over the horizon, so a saturating curve imposed on either
one misfires (verified on synthetic worlds: an E/N median scored
1.3-2.7 logit of error where the direct fit scored 0.14).  The
logit rate, by contrast, does saturate toward its per-trophy
asymptote, which is exactly what the damped curve fits.

Standard library only, no network, no randomness: two runs on the
same input are byte-identical.

Run:  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a
# module shipped beside this file imports.  (v5 ships one file only;
# the line stays so the layout matches the kit's starter.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the levels, for the fallback fan only.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Log-time origin: the game's release, 2026-11-19T00:00:00Z, the same
# origin the house curve baseline uses.  x = ln(1 + hours since then).
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Grid of decay rates for the saturating fit y = A - B*exp(-k*x).
# Small k is nearly a straight line in log-time; large k flattens fast.
K_GRID = [0.05, 0.08, 0.12, 0.18, 0.27, 0.40, 0.60, 0.90, 1.35, 2.0, 3.0]

# Shrinkage strength for sparse trophies: a trophy whose last read
# holds E earners keeps weight E / (E + K_SHRINK) on its own forecast
# and takes the rest from its tier's pooled forecast.
K_SHRINK = 8.0
# Pseudo-earners credited per read a trophy was observed: a long,
# stable record is evidence even at few earners, so the shrinkage
# weight counts reads as well as earners.
BETA_PER_READ = 0.25

# The fan's logit scale never goes below this: on a flat history the
# backtest sees no error at all, and a zero-width fan would pay for
# every trophy the moment anything moves.
SD_FLOOR = 0.02
# ... and never above this: a wider fan pays more than it covers.
SD_CAP = 0.8
# Fan scale for a flat history (no movement recorded): a refresh can
# still land before the target, and its jump is of this order.
SD_FLAT = 0.40

# In-case backtest: cut the history at these fractions and extrapolate
# to these horizons (hours), and score the logit error.  Only the
# recent cuts are used: the backtest error is dominated by a
# systematic bias that is large while growth is fast and small once
# growth slows, and the target sits in the slow-growth regime, so
# early cuts (whose bias is an order of magnitude larger than the
# target's) must not set the fan.
CUT_FRACTIONS = [0.67, 0.83]
BACKTEST_HORIZONS_H = [6.0, 12.0, 24.0, 48.0]


def clamp_pct(x):
    """A rate in percent, clamped to the range the logit can take."""
    return min(99.99, max(0.01, x))


def logit_pct(p):
    p = clamp_pct(p) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def parse_time(s):
    """An ISO-8601 UTC timestamp to epoch seconds."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def log_time(t_epoch):
    """Log-time x = ln(1 + hours since release), floored at 0."""
    hours = (t_epoch - RELEASE.timestamp()) / 3600.0
    return math.log1p(hours) if hours > 0 else 0.0


def _ols(xs, ys):
    """Least squares y ~ a + b*x; returns (a, b, sse)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, sse


def sat_fit(xs, ys):
    """Fit y = A - B*exp(-k*x) with B >= 0 (non-decreasing in x).

    For each k on the grid the curve is linear in exp(-k*x), so the
    fit is ordinary least squares; the best k is the one with the
    smallest squared error.  A straight line in x is tried too, as
    the k -> 0 limit.  With fewer than 3 points, or a flat series,
    the fit is flat at the last value.  Returns (A, B, k) with k = 0
    meaning the straight-line fit (A is then the intercept and B the
    negative slope, kept only when the slope is non-negative).
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n < 3 or all(y == ys[0] for y in ys):
        return ys[-1], 0.0, 0.0
    best = None  # (sse, A, B, k)
    for k in K_GRID:
        us = [math.exp(-k * x) for x in xs]
        a, c, sse = _ols(us, ys)
        b = -c
        if b < 0:
            # A decreasing curve is not a saturation: fall back to flat.
            sse = sum((y - ys[-1]) ** 2 for y in ys)
            cand = (sse, ys[-1], 0.0, k)
        else:
            cand = (sse, a, b, k)
        if best is None or cand[0] < best[0]:
            best = cand
    a, b, sse = _ols(xs, ys)
    if b < 0:
        cand = (sum((y - ys[-1]) ** 2 for y in ys), ys[-1], 0.0, 0.0)
    else:
        cand = (sse, a, -b, 0.0)
    if best is None or cand[0] < best[0]:
        best = cand
    _, a, b, k = best
    return a, b, k


def sat_value(fit, x):
    """The fitted curve at log-time x."""
    a, b, k = fit
    if k == 0:
        return a - b * x
    return a - b * math.exp(-k * x)


def read_case(path):
    case = json.loads(path.read_text(encoding="utf-8"))
    hist = []
    for read in case["history"]:
        t = parse_time(read["at"])
        rates = read["rates"]
        n = read["players_tracked"]
        earners = []
        for r in rates:
            earners.append(None if r is None else r / 100.0 * n)
        hist.append({"t": t, "x": log_time(t), "n": n,
                     "rates": rates, "earners": earners})
    hist.sort(key=lambda r: r["t"])
    return case, hist


def fit_trophy(hist, idx):
    """Saturating fit of one trophy's logit rate in log-time; returns
    the fit, the earners at the last read where it was observed, and
    the number of reads where it was observed."""
    xs, ys = [], []
    last_e = 0.0
    n_obs = 0
    for r in hist:
        v = r["rates"][idx]
        if v is None:
            continue
        xs.append(r["x"])
        ys.append(logit_pct(v))
        last_e = r["earners"][idx]
        n_obs += 1
    return sat_fit(xs, ys), last_e, n_obs


def fit_tier_pool(hist, tier_idx):
    """Saturating fit of a tier's pooled logit rate: per read, the
    pooled rate is the tier's total earners over its observed
    trophies divided by the players tracked."""
    xs, ys = [], []
    for r in hist:
        tot, m = 0.0, 0
        for idx in tier_idx:
            e = r["earners"][idx]
            if e is not None:
                tot += e
                m += 1
        if m == 0:
            continue
        xs.append(r["x"])
        ys.append(logit_pct(100.0 * tot / (m * r["n"])))
    return sat_fit(xs, ys)


def median_forecast(case, hist, target_t=None):
    """The median rate (percent) for every trophy at the target.

    A damped saturation trend on each trophy's logit rate in
    log-time, read at the target; sparse trophies are shrunk toward
    their tier's pooled trend by earners held.
    """
    if target_t is None:
        target_t = parse_time(case["target"])
    target_x = log_time(target_t)
    tiers = {}
    for i, tr in enumerate(case["trophies"]):
        tiers.setdefault(tr["tier"], []).append(i)
    pool_fit = {tier: fit_tier_pool(hist, idxs)
                for tier, idxs in tiers.items()}
    medians = []
    for i, tr in enumerate(case["trophies"]):
        fit, e_last, n_obs = fit_trophy(hist, i)
        lg_own = sat_value(fit, target_x)
        lg_pool = sat_value(pool_fit[tr["tier"]], target_x)
        # A trophy's own forecast earns its weight from two things:
        # the earners it holds and the length of its own record.  A
        # long, stable series is strong evidence even at few
        # earners, so a trophy with many reads keeps most of its own
        # forecast; a trophy with almost no record falls back to
        # its tier's pooled trend.  The read credit only counts once
        # the trophy has at least one earner: a long record at zero
        # earners is not evidence the trophy stays at the floor --
        # it is censored, it has simply not been earned yet, and it
        # can jump at any time, so it stays on the pooled trend.
        read_credit = BETA_PER_READ * n_obs if e_last > 0 else 0.0
        w = (e_last + read_credit) / (
            e_last + read_credit + K_SHRINK)
        lg = w * lg_own + (1.0 - w) * lg_pool
        medians.append(clamp_pct(inv_logit_pct(lg)))
    return medians


def history_is_flat(case, hist):
    """True when the recorded history barely moved: the 75th
    percentile trophy's logit rate spans less than 0.05 over the
    whole history.  The percentile (not the median) matters, because
    in a world where many trophies are censored at zero earners the
    median trophy never moves even while the trophies that did earn
    moved a lot -- that history is not flat.  A small backtest error
    is not the test -- a model can predict a moving history well --
    so flatness is read off the data itself."""
    spans = []
    for i in range(len(case["trophies"])):
        vals = [logit_pct(r["rates"][i]) for r in hist
                if r["rates"][i] is not None]
        if vals:
            spans.append(max(vals) - min(vals))
    if not spans:
        return True
    spans.sort()
    return spans[int(0.75 * (len(spans) - 1))] < 0.05


def backtest_fan(case, hist):
    """Conformal fan from the model's own in-case backtest.

    Cut the recorded series, extrapolate to a later time that is
    really in the history, and record the logit error of the median
    against the SMOOTHED rate around that time (the mean logit over
    reads within a few hours either side).  Scoring against a single
    frozen read would measure the refresh-cycle lag -- the page sits
    still between refreshes while the smooth trend keeps rising, so
    a single read is systematically below the trend in a fast-growing
    regime, a bias that does not carry to the target, where growth
    has slowed.  The smoothed truth removes that lag and leaves the
    trend error, which is what the fan must cover.  The error
    variance is split into a horizon-constant part (the remaining
    step noise) and a part that grows with horizon (trend error):
    var(h) = c + d*h, fitted on the per-horizon robust scales.  The
    fan at the target uses the empirical quantiles of the
    standardised errors times the target horizon's scale.  Returns
    (z_quantiles, sd_target).
    """
    n = len(hist)
    horizon_h = case["horizon_minutes"] / 60.0
    if n < 16 and history_is_flat(case, hist):
        # Too few reads for a backtest and the record never moved:
        # no signal at all, so a refresh can still land before the
        # target.  Use the flat-history fan.  (A longer flat history
        # does reach the backtest below, whose own errors give it a
        # fan; the floor is applied there instead, keeping its shift.)
        return Z, SD_FLAT
    if n < 16:
        return Z, max(SD_FLOOR, 0.05 * math.sqrt(horizon_h / 24.0))
    errors = []  # (horizon_hours, logit error)
    cuts = sorted(set(max(3, int(n * f)) for f in CUT_FRACTIONS))
    for cut in cuts:
        cut_t = hist[cut - 1]["t"]
        for h in BACKTEST_HORIZONS_H:
            fut = None
            for r in hist[cut:]:
                if r["t"] >= cut_t + h * 3600.0:
                    fut = r
                    break
            if fut is None:
                continue
            try:
                med = median_forecast(case, hist[:cut], fut["t"])
            except Exception:
                continue
            # Smoothed truth: mean logit over reads within +-3 h.
            window = [r for r in hist
                      if abs(r["t"] - fut["t"]) <= 3.0 * 3600.0]
            if len(window) < 3:
                continue
            for i in range(len(case["trophies"])):
                vals = []
                for r in window:
                    v = r["rates"][i]
                    if v is not None:
                        vals.append(logit_pct(v))
                if not vals:
                    continue
                truth_lg = sum(vals) / len(vals)
                errors.append((h, truth_lg - logit_pct(med[i])))
    if len(errors) < 30:
        return Z, max(SD_FLOOR, 0.05 * math.sqrt(horizon_h / 24.0))
    by_h = {}
    for h, e in errors:
        by_h.setdefault(h, []).append(e)
    points = []  # (horizon, robust variance)
    for h, es in sorted(by_h.items()):
        es_sorted = sorted(es)
        m = len(es_sorted)
        q1 = es_sorted[int(0.25 * (m - 1))]
        q3 = es_sorted[int(0.75 * (m - 1))]
        sd = (q3 - q1) / 1.349
        points.append((h, max(sd, 1e-6) ** 2))
    # var(h) = c + d*h, least squares on the per-horizon variances.
    xs = [h for h, _ in points]
    ys = [v for _, v in points]
    c, d, _ = _ols(xs, ys)
    if d < 0:
        d = 0.0
        c = sum(ys) / len(ys)
    if c < 0:
        c = 0.0
    var_target = c + d * horizon_h
    sd_target = min(SD_CAP, max(math.sqrt(var_target), SD_FLOOR))
    if history_is_flat(case, hist):
        # A long flat history reaches this backtest, but its errors
        # are only the shrinkage pull, not real movement: the fan
        # must not collapse on that.  Keep the backtest's shift and
        # raise only its scale to the flat-history level.
        sd_target = min(SD_CAP, max(sd_target, SD_FLAT))
    zs = []
    for h, e in errors:
        sd_h = math.sqrt(max(c + d * h, 1e-12))
        zs.append(e / sd_h)
    zs.sort()
    m = len(zs)
    quants = [zs[min(m - 1, max(0, int(tau * (m - 1))))]
              for tau in LEVELS]
    return quants, sd_target


def forecast_case(case, hist):
    if not hist:
        # No history at all: every rate is unknown; forecast a flat
        # 1% with a wide fan rather than crashing the case.
        return {"trophies": [
            {"id": tr["id"], "q": [0.05, 0.25, 1.0, 4.0, 20.0]}
            for tr in case["trophies"]]}
    medians = median_forecast(case, hist)
    quants, sd_target = backtest_fan(case, hist)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        lg = logit_pct(medians[i])
        qs = []
        prev = 0.0
        for tau, zq in zip(LEVELS, quants):
            v = clamp_pct(inv_logit_pct(lg + zq * sd_target))
            v = round(v, 4)
            if v < prev:
                v = prev
            qs.append(v)
            prev = v
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case, hist = read_case(path)
        forecast = forecast_case(case, hist)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
