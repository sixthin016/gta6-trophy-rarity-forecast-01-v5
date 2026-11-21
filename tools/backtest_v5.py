"""Local verification harness for the v5 nowcast entry (fresh code).

Builds Exophase-like worlds with a known truth, runs v5's main.py on
them exactly as the grader would (subprocess, env dirs), and scores
the result with the contest's own pinball loss on the logit scale,
head-to-head against the v3r entry and the house persistence baseline
on the same cases.  Also scores the practice proxy, and checks format,
determinism, runtime and degenerate edge cases.

Worlds (all share the observation mechanics from the published rules:
the page is frozen between refreshes and jumps at a refresh, because a
refresh brings in a batch of newly tracked players):
  weibull      -- true rate p(t) = 1 - exp(-((t+2)/lam)^k)
  loglogistic  -- true rate p(t) = 1 / (1 + (t0/t)^b)
  delayed      -- true rate 0 until a per-trophy delay, then logistic
  flat-refresh -- history perfectly flat, one refresh lands at target
                  (the practice shape); new players earn at 0.7x

The final-like worlds use a 60 h history (30-min reads, the final
case's shape) and a 108 h horizon.  The truth at the target is the
page state at the first 10-minute slot at or after the target, the
same object the house's answering read prints.
"""

import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

V5 = Path(__file__).resolve().parent.parent
MAIN_V5 = V5 / "main.py"
MAIN_V3R = Path("/home/hatch/workspace/makefun/"
                "gta6-trophy-rarity-forecast-01-v3r/main.py")
KIT_IN = Path("/home/hatch/workspace/makefun/"
              "gta6-trophy-rarity-forecast-01-kit/"
              "gta6-trophy-rarity-forecast-01-kit/input")

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
TIERS = (["platinum"] + ["gold"] * 2 + ["silver"] * 6 + ["bronze"] * 46)
T0 = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(p):
    p = clamp_pct(p) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def pinball(y, qs):
    tot = 0.0
    for tau, q in zip(LEVELS, qs):
        d = logit_pct(y) - logit_pct(q)
        tot += 2.0 * max(tau * d, (tau - 1.0) * d)
    return tot / len(LEVELS)


def score_forecast(truth, forecast):
    by_id = {r["id"]: r["q"] for r in forecast["trophies"]}
    vals = [pinball(truth[i], by_id[i + 1]) for i in range(len(truth))]
    return sum(vals) / len(vals)


def persistence_forecast(case):
    """The house baseline (RULES.txt formula), for comparison only."""
    last = case["history"][-1]["rates"]
    d = case["horizon_minutes"] / 1440.0
    spread = 0.5 * math.sqrt(d)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        r = last[i] if last[i] is not None else 0.0
        lg = logit_pct(r)
        rows.append({"id": tr["id"],
                     "q": [round(inv_logit_pct(lg + z * spread), 4)
                           for z in Z]})
    return {"trophies": rows}


def _fmt(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def true_rate(world, prm, t):
    if t <= 0:
        return 0.0
    if world == "weibull":
        lam, k = prm
        return 100.0 * (1.0 - math.exp(-((t + 2.0) / lam) ** k))
    if world == "loglogistic":
        t0, b = prm
        return 100.0 / (1.0 + (t0 / t) ** b)
    if world == "delayed":
        delay, s, cap = prm
        if t < delay:
            return 0.0
        return cap / (1.0 + math.exp(-(t - delay) / s))
    if world == "flat-refresh":
        base, = prm
        return base
    raise ValueError(world)


def make_params(world, rng):
    prm = []
    for tier in TIERS:
        scale = {"platinum": 3.0, "gold": 1.6, "silver": 1.0,
                 "bronze": 0.55}[tier]
        if world == "weibull":
            prm.append((rng.uniform(250, 900) * scale,
                        rng.uniform(0.7, 1.4)))
        elif world == "loglogistic":
            prm.append((rng.uniform(150, 700) * scale,
                        rng.uniform(0.8, 1.6)))
        elif world == "delayed":
            prm.append((rng.uniform(20, 300) * scale,
                        rng.uniform(8, 40),
                        rng.uniform(0.5, 60)))
        elif world == "flat-refresh":
            prm.append((rng.uniform(0.5, 3.0),))
    return prm


def build_case(world, seed, hist_hours=60.0, horizon_hours=108.0):
    rng = random.Random(seed)
    prm = make_params(world, rng)
    # Player counts: a saturating launch curve, not exponential
    # growth (the addressable player base is finite).  N_smooth(t)
    # is a log-logistic rise toward N_inf; the page only updates at
    # refreshes, so a read shows the curve's value at the last
    # refresh at or before it.
    n_inf = math.exp(rng.uniform(math.log(200000.0), math.log(2000000.0)))
    t1 = rng.uniform(30.0, 120.0)
    g = rng.uniform(0.8, 1.5)

    def n_smooth(t):
        if t <= 0:
            return 638.0
        return max(638.0, n_inf / (1.0 + (t1 / t) ** g))

    # Refresh clock, simulated past the target.
    refreshes = []
    t = rng.uniform(2.0, 5.0)
    end = hist_hours + horizon_hours + 1.0
    while t < end:
        refreshes.append(t)
        t += rng.uniform(3.5, 6.5)
    if world == "flat-refresh":
        refreshes = [hist_hours + horizon_hours - 0.25]

    def page_state(t_read):
        """(players, rates) the page prints at t_read: the state as of
        the last refresh at or before it, rates to 2 decimals."""
        past = [rt for rt in refreshes if rt <= t_read]
        rt = past[-1] if past else 0.0
        if world == "flat-refresh":
            nn = 638.0 if not past else 638.0 * 6.06
        else:
            nn = n_smooth(rt) if rt else 638.0
        rates = []
        for i in range(55):
            if world == "flat-refresh":
                e_old = prm[i][0] / 100.0 * 638.0
                e_new = (0.7 * prm[i][0] / 100.0 * (nn - 638.0)
                         if rt else 0.0)
                rates.append(round(100.0 * (e_old + e_new) / nn, 2))
            else:
                rates.append(round(true_rate(world, prm[i],
                                             rt if rt else 0.1), 2))
        return int(round(nn)), rates

    hist = []
    t = 0.1
    while t < hist_hours:
        nn, rates = page_state(t)
        hist.append({"slot": _fmt(T0 + timedelta(hours=t)),
                     "at": _fmt(T0 + timedelta(hours=t)),
                     "players_tracked": nn,
                     "trophies_earned": int(sum(r * nn / 100.0
                                                for r in rates)),
                     "rates": rates})
        t += 0.5
    # Truth: page state at the first 10-minute slot at/after target.
    t_tgt = hist_hours + horizon_hours
    slot = math.ceil(t_tgt * 6) / 6.0
    _, truth = page_state(slot)
    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "syn-%s-%d" % (world, seed),
        "at": _fmt(T0 + timedelta(hours=hist_hours)),
        "horizon_minutes": int(horizon_hours * 60),
        "target": _fmt(T0 + timedelta(hours=t_tgt)),
        "levels": LEVELS,
        "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                 "release_date": "2026-11-19", "source": "synthetic"},
        "trophies": [{"id": i + 1, "name": "Trophy %d" % (i + 1),
                      "description": "", "tier": TIERS[i],
                      "points": {"platinum": 300, "gold": 90,
                                 "silver": 30, "bronze": 15}[TIERS[i]],
                      "secret": False} for i in range(55)],
        "history": hist,
        "notes": "synthetic world %s seed %d" % (world, seed),
    }
    return case, truth


def run_main(main_path, case):
    with tempfile.TemporaryDirectory() as td:
        ind = Path(td) / "in"
        outd = Path(td) / "out"
        ind.mkdir()
        (ind / (case["id"] + ".in")).write_text(json.dumps(case))
        env = dict(os.environ, MAKEFUN_INPUT_DIR=str(ind),
                   MAKEFUN_OUTPUT_DIR=str(outd))
        t0 = time.time()
        r = subprocess.run([sys.executable, str(main_path)], env=env,
                           capture_output=True, text=True, timeout=60)
        dt = time.time() - t0
        if r.returncode != 0:
            raise RuntimeError("main.py failed: " + r.stderr[-2000:])
        return json.loads((outd / (case["id"] + ".out")).read_text()), dt


def check_valid(forecast, n=55):
    assert set(forecast.keys()) == {"trophies"}, forecast.keys()
    assert len(forecast["trophies"]) == n
    ids = sorted(r["id"] for r in forecast["trophies"])
    assert ids == list(range(1, n + 1)), ids[:5]
    for r in forecast["trophies"]:
        assert set(r.keys()) == {"id", "q"}
        q = r["q"]
        assert len(q) == 5
        assert all(0.0 <= v <= 100.0 for v in q), q
        assert all(q[i] <= q[i + 1] for i in range(4)), q


def synthetic_worlds():
    print("== synthetic worlds (final-like: 60 h history, 108 h horizon) ==")
    print("%-14s %-6s %10s %10s %10s %8s" % ("world", "seed", "v5",
                                             "v3r", "persist", "v5/v3r"))
    totals = {}
    for world in ("weibull", "loglogistic", "delayed", "flat-refresh"):
        for seed in (101, 202):
            case, truth = build_case(world, seed)
            ours, dt = run_main(MAIN_V5, case)
            check_valid(ours)
            v3r, _ = run_main(MAIN_V3R, case)
            check_valid(v3r)
            lo = score_forecast(truth, ours)
            l3 = score_forecast(truth, v3r)
            lp = score_forecast(truth, persistence_forecast(case))
            totals.setdefault(world, []).append((lo, l3, lp))
            print("%-14s %-6d %10.4f %10.4f %10.4f %8.3f   (%.1fs)"
                  % (world, seed, lo, l3, lp, lo / l3, dt))
    print("-- means --")
    all_o, all_3, all_p = [], [], []
    for w, vals in totals.items():
        mo = sum(v[0] for v in vals) / len(vals)
        m3 = sum(v[1] for v in vals) / len(vals)
        mp = sum(v[2] for v in vals) / len(vals)
        all_o.append(mo)
        all_3.append(m3)
        all_p.append(mp)
        print("%-14s v5 %.4f  v3r %.4f  persistence %.4f  v5/v3r %.3f"
              % (w, mo, m3, mp, mo / m3))
    print("OVERALL mean: v5 %.4f  v3r %.4f  persistence %.4f"
          % (sum(all_o) / len(all_o), sum(all_3) / len(all_3),
             sum(all_p) / len(all_p)))


def practice_cases():
    print("== practice cases: format / determinism / runtime ==")
    for name in ("p00", "p01"):
        src = KIT_IN / (name + ".in")
        with tempfile.TemporaryDirectory() as td:
            ind = Path(td) / "in"
            out1 = Path(td) / "o1"
            out2 = Path(td) / "o2"
            ind.mkdir()
            (ind / (name + ".in")).write_text(src.read_text())
            env = dict(os.environ)
            for outd in (out1, out2):
                env2 = dict(env, MAKEFUN_INPUT_DIR=str(ind),
                            MAKEFUN_OUTPUT_DIR=str(outd))
                t0 = time.time()
                r = subprocess.run([sys.executable, str(MAIN_V5)],
                                   env=env2, capture_output=True,
                                   text=True, timeout=60)
                dt = time.time() - t0
                assert r.returncode == 0, r.stderr[-1000:]
                print("%s run: %.2fs" % (name, dt))
            b1 = (out1 / (name + ".out")).read_bytes()
            b2 = (out2 / (name + ".out")).read_bytes()
            assert b1 == b2, "non-deterministic output!"
            assert len(b1) <= 65536, len(b1)
            check_valid(json.loads(b1))
            print("%s: valid, %d bytes, byte-identical across runs"
                  % (name, len(b1)))


def practice_proxy():
    print("== practice proxy (real p00/p01, synthetic refresh truths) ==")
    print("%-6s %-4s %10s %10s %10s" % ("case", "scen", "v5", "v3r",
                                       "persist"))
    scenarios = {"A": (5.0, 0.5), "B": (5.0, 0.7),
                 "C": (2.0, 0.7), "D": (10.0, 0.7)}
    rows = []
    for name in ("p00", "p01"):
        case = json.loads((KIT_IN / (name + ".in")).read_text())
        ours, _ = run_main(MAIN_V5, case)
        v3r, _ = run_main(MAIN_V3R, case)
        last = case["history"][-1]["rates"]
        pers = persistence_forecast(case)
        for scen, (mult, phi) in scenarios.items():
            truth = []
            for r in last:
                p = 1.0 if r >= 99.99 else phi
                truth.append(round(r * (1.0 + p * (mult - 1.0)) / mult, 2))
            lo = score_forecast(truth, ours)
            l3 = score_forecast(truth, v3r)
            lp = score_forecast(truth, pers)
            rows.append((scen, lo, l3, lp))
            print("%-6s %-4s %10.4f %10.4f %10.4f"
                  % (name, scen, lo, l3, lp))
    print("-- means over both cases --")
    for scen in scenarios:
        vals = [(lo, l3, lp) for s, lo, l3, lp in rows if s == scen]
        mo = sum(v[0] for v in vals) / len(vals)
        m3 = sum(v[1] for v in vals) / len(vals)
        mp = sum(v[2] for v in vals) / len(vals)
        print("scenario %s: v5 %.4f  v3r %.4f  persistence %.4f"
              % (scen, mo, m3, mp))
    all_o = [lo for _, lo, _, _ in rows]
    all_3 = [l3 for _, _, l3, _ in rows]
    all_p = [lp for _, _, _, lp in rows]
    print("OVERALL mean: v5 %.4f  v3r %.4f  persistence %.4f"
          % (sum(all_o) / len(all_o), sum(all_3) / len(all_3),
             sum(all_p) / len(all_p)))


def edge_cases():
    print("== edge cases ==")
    base, _ = build_case("weibull", 303)
    cases = {}
    one = dict(base)
    one["history"] = base["history"][:1]
    one["id"] = "edge-1read"
    cases["1 read"] = one
    empty = dict(base)
    empty["history"] = []
    empty["id"] = "edge-empty"
    cases["empty history"] = empty
    nulls = dict(base)
    nulls["history"] = [dict(h, rates=[None if (i % 3 == 0) else r
                                      for i, r in enumerate(h["rates"])])
                        for h in base["history"]]
    nulls["id"] = "edge-nulls"
    cases["null rates (1/3)"] = nulls
    floor = dict(base)
    floor["history"] = [dict(h, rates=[0.01] * 55) for h in base["history"]]
    floor["id"] = "edge-floor"
    cases["all rates at floor"] = floor
    for name, c in cases.items():
        out, _ = run_main(MAIN_V5, c)
        check_valid(out)
        print("%-20s OK" % name)


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    if "all" in which or "worlds" in which:
        synthetic_worlds()
    if "all" in which or "practice" in which:
        practice_cases()
    if "all" in which or "proxy" in which:
        practice_proxy()
    if "all" in which or "edge" in which:
        edge_cases()
    print("ALL CHECKS DONE")
