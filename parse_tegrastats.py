"""
parse_tegrastats.py -- turn Jetson power logs into the MEASURED parameters of the simulation.

Protocol (one power mode for every log; same peripherals/SSH session; delete old *.log first,
tegrastats may append to an existing file):

  idle1 : sudo tegrastats --interval 1000 --logfile idle1.log --start ; sleep 720 ; sudo tegrastats --stop
  light : sudo tegrastats --interval 1000 --logfile light.log --start
          python3 inference_logger_template.py --mode light --fps 30 --minutes 11
          sudo tegrastats --stop
  twist : same with --mode twist and twist.log
  idle2 : same as idle1 with idle2.log (checks thermal / background drift)

Then:
  python parse_tegrastats.py build --idle idle1.log idle2.log --light light.log --twist twist.log \
         --counts-light inference_counts_light.csv --counts-twist inference_counts_twist.csv
  python ekf_edge_battery.py --params params_measured.json > run_output.txt
  python parse_tegrastats.py plot --idle idle1.log idle2.log --light light.log --twist twist.log   # report figure

The first --skip-head seconds (default 120: model loading + warm-up) and the last --skip-tail
seconds (default 10) of every power log are dropped. `build` computes
  P_IDLE      = mean VDD_IN power of the idle log(s)                          [W]
  E_INF       = (mean power of light.log - P_IDLE) / fps_light                [J/inference]
  E_INF_TWIST = (mean power of twist.log - P_IDLE) / fps_twist                [J/inference]
  R_PROC      = (100 * sigma_P * DT / (3600 * C_AH * V_nom))^2                 [%^2]
                sigma_P = std of the DT-second mean powers in light.log; V_nom = mean pack OCV
  N_HIGH      = fps_light * DT;   N_TWIST_MAX = fps_twist * DT if the fallback is slower.
fps is the rate actually achieved, taken from the logger CSV (recommended) or --fps-light/--fps-twist.

Notes
  * tegrastats prints "RAIL cur_mW/avg_mW"; only the first (instantaneous) value is used.
  * On the Orin Nano developer kit VDD_IN is the module input rail; carrier-board loads (fan, USB,
    SSD, regulators) are not included. E_INF is a difference and is barely affected; P_IDLE is a
    lower bound of what a battery would see (see README for the optional external-meter check).
  * C_AH and N_CELLS are ASSUMED (the battery you would deploy with); they only enter R_PROC here.
"""
import os
import re
import csv
import json
import argparse
import numpy as np

RAIL_RE = re.compile(r"\b([A-Z][A-Z0-9_]*)\s+(\d+)mW/(\d+)mW")
MEAN_OCV_PER_CELL = 3.6583      # mean of 3.0 + 3.1 s - 5.0 s^2 + 3.1 s^3 over s in [0, 1]


def read_power(path, rail="VDD_IN"):
    """Instantaneous power [W] per tegrastats line for one rail."""
    vals, seen = [], set()
    with open(path, errors="ignore") as f:
        for line in f:
            found = {m.group(1): int(m.group(2)) for m in RAIL_RE.finditer(line)}
            seen.update(found)
            if rail in found:
                vals.append(found[rail] / 1000.0)
    if not vals:
        raise SystemExit(f"rail '{rail}' not found in {path}. Rails seen: {sorted(seen) or 'none'}")
    return np.asarray(vals)


def trim(p, interval, head, tail):
    """Drop the first `head` and last `tail` seconds of a power trace."""
    h, t = int(round(head / interval)), int(round(tail / interval))
    q = p[h:len(p) - t] if t > 0 else p[h:]
    if len(q) < 2:
        raise SystemExit("log too short after trimming; record longer or lower --skip-head")
    return q


def stats(p, interval, window):
    k = len(p) // window
    wm = p[:k * window].reshape(k, window).mean(axis=1) if k >= 2 else None
    return dict(n=len(p), minutes=len(p) * interval / 60.0, mean=float(p.mean()),
                std=float(p.std(ddof=1)) if len(p) > 1 else 0.0,
                win_std=float(wm.std(ddof=1)) if wm is not None else float("nan"),
                n_win=0 if wm is None else len(wm))


def fps_from_counts(path):
    """Achieved inferences per second from the logger CSV (minute,count,mean_latency_ms)."""
    with open(path) as f:
        counts = [int(r["count"]) for r in csv.DictReader(f)]
    if not counts:
        raise SystemExit(f"{path}: no full minutes recorded")
    return float(np.mean(counts)) / 60.0, len(counts)


def cmd_stats(a):
    p = trim(read_power(a.log, a.rail), a.interval, a.skip_head, a.skip_tail)
    s = stats(p, a.interval, int(round(60 / a.interval)))
    print(f"{a.log}: {s['n']} samples ({s['minutes']:.1f} min)")
    print(f"  mean = {s['mean']:.3f} W   std of 1-s samples = {s['std']:.3f} W")
    if s["n_win"] >= 2:
        print(f"  std of 60-s means = {s['win_std']:.3f} W  ({s['n_win']} windows)")
    else:
        print("  log shorter than 2 minutes: cannot estimate the 60-s fluctuation (record >= 10 min)")


TS_RE = re.compile(r"^(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d)")
GR_RE = re.compile(r"GR3D_FREQ (\d+)%")
GT_RE = re.compile(r"gpu@([\d.]+)C")


def read_trace(path, rail="VDD_IN"):
    """Per-line timestamp [s], power [W], GPU utilisation [%], GPU temperature [C]."""
    from datetime import datetime
    t, pw, gu, gt = [], [], [], []
    for i, line in enumerate(open(path, errors="ignore")):
        m = {mm.group(1): int(mm.group(2)) for mm in RAIL_RE.finditer(line)}
        if rail not in m:
            continue
        ts = TS_RE.search(line)
        t.append(datetime.strptime(ts.group(1), "%m-%d-%Y %H:%M:%S").timestamp() if ts else float(i))
        pw.append(m[rail] / 1000.0)
        g = GR_RE.search(line)
        gu.append(float(g.group(1)) if g else np.nan)
        g = GT_RE.search(line)
        gt.append(float(g.group(1)) if g else np.nan)
    return np.array(t), np.array(pw), np.array(gu), np.array(gt)


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logs = [(p, "idle") for p in a.idle] + [(a.light, "light")] + ([(a.twist, "twist")] if a.twist else [])
    logs.sort(key=lambda pl: read_trace(pl[0], a.rail)[0][0])
    t0 = min(read_trace(p, a.rail)[0][0] for p, _ in logs)
    col = {"idle": "tab:gray", "light": "tab:blue", "twist": "tab:orange"}
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for path, kind in logs:
        t, pw, gu, gt = read_trace(path, a.rail)
        tm = (t - t0) / 60.0
        h, tl = int(round(a.skip_head / a.interval)), int(round(a.skip_tail / a.interval))
        used = slice(h, len(pw) - tl if tl else None)
        name = path.rsplit("/", 1)[-1].replace(".log", "")
        ax[0].plot(tm, pw, color=col[kind], lw=0.6, alpha=0.35)
        ax[0].plot(tm[used], pw[used], color=col[kind], lw=0.8, label=f"{name}: {pw[used].mean():.3f} W")
        ax[0].hlines(pw[used].mean(), tm[used][0], tm[used][-1], colors="k", lw=1.2, linestyles="--")
        ax[1].plot(tm, gu, color=col[kind], lw=0.8)
        ax[2].plot(tm, gt, color=col[kind], lw=0.8)
    ax[0].set(ylabel=f"{a.rail} power [W]", title="Jetson power log (faint = trimmed start/end, dashed = mean used)")
    ax[0].legend(fontsize=8, ncol=2)
    ax[1].set(ylabel="GPU utilisation [%]")
    ax[2].set(ylabel="GPU temperature [C]", xlabel="session time [min]")
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    if os.path.dirname(a.out):
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


def cmd_build(a):
    window = int(round(a.dt / a.interval))
    rd = lambda path: trim(read_power(path, a.rail), a.interval, a.skip_head, a.skip_tail)
    idle_parts = [rd(p) for p in a.idle]
    if len(idle_parts) > 1:
        means = [float(q.mean()) for q in idle_parts]
        print("idle logs: " + ", ".join(f"{p} {m:.3f} W" for p, m in zip(a.idle, means)))
        if max(means) - min(means) > 0.2:
            print("WARNING: idle power drifted by more than 0.2 W between the idle logs "
                  "(temperature, fan or background tasks changed); P_IDLE uses their average")
    idle = stats(np.concatenate(idle_parts), a.interval, window)
    light = stats(rd(a.light), a.interval, window)
    twist = stats(rd(a.twist), a.interval, window) if a.twist else None

    fps_l = a.fps_light
    if a.counts_light:
        fps_l, nm = fps_from_counts(a.counts_light)
        print(f"achieved fps (light) from {a.counts_light}: {fps_l:.2f} over {nm} min")
    if fps_l is None:
        raise SystemExit("give --counts-light (recommended) or --fps-light")
    fps_t = a.fps_twist if a.fps_twist is not None else fps_l
    if a.counts_twist:
        fps_t, nm = fps_from_counts(a.counts_twist)
        print(f"achieved fps (twist) from {a.counts_twist}: {fps_t:.2f} over {nm} min")

    p_idle = idle["mean"]
    e_inf = (light["mean"] - p_idle) / fps_l
    sigma_p = light["win_std"]
    if np.isnan(sigma_p):
        raise SystemExit("light.log is too short to estimate the minute-to-minute fluctuation; record >= 10 min")
    v_nom = a.n_cells * MEAN_OCV_PER_CELL
    sigma_x = 100.0 * sigma_p * a.dt / (3600.0 * a.c_ah * v_nom)
    params = {
        "_note": "MEASURED with tegrastats (P_IDLE, E_INF, E_INF_TWIST, R_PROC, N_HIGH, N_TWIST_MAX); "
                 "C_AH, N_CELLS, DT are ASSUMED and copied here for consistency",
        "P_IDLE": round(p_idle, 4), "E_INF": round(e_inf, 5), "R_PROC": float(f"{sigma_x**2:.4g}"),
        "N_HIGH": int(round(fps_l * a.dt)),
        "C_AH": a.c_ah, "N_CELLS": a.n_cells, "DT": a.dt,
        "_measured_detail": {"rail": a.rail, "fps_light": round(fps_l, 3), "P_light_W": round(light["mean"], 4),
                             "sigma_P_60s_W": round(sigma_p, 4), "idle_min_used": round(idle["minutes"], 1),
                             "light_min_used": round(light["minutes"], 1),
                             "skip_head_s": a.skip_head, "skip_tail_s": a.skip_tail},
    }
    print(f"P_IDLE = {p_idle:.3f} W")
    print(f"light  : {light['mean']:.3f} W at {fps_l:.2f} fps -> E_INF = {e_inf:.4f} J/inference")
    print(f"minute-to-minute fluctuation sigma_P = {sigma_p:.3f} W -> sigma = {sigma_x:.4f} %/step -> R_PROC = {sigma_x**2:.3e} %^2")
    if twist:
        e_tw = (twist["mean"] - p_idle) / fps_t
        params["E_INF_TWIST"] = round(e_tw, 5)
        params["_measured_detail"].update(fps_twist=round(fps_t, 3), P_twist_W=round(twist["mean"], 4),
                                          twist_min_used=round(twist["minutes"], 1))
        print(f"twist  : {twist['mean']:.3f} W at {fps_t:.2f} fps -> E_INF_TWIST = {e_tw:.4f} J/inference")
        d_tw, d_li = twist["mean"] - p_idle, light["mean"] - p_idle
        above = f"x{d_tw/d_li:.2f}" if d_li > 0.05 else "n/a"
        print(f"  inference power above idle: fallback {d_tw:.2f} W vs normal {d_li:.2f} W ({above}); "
              f"total {twist['mean']:.2f} W vs {light['mean']:.2f} W (x{twist['mean']/light['mean']:.2f})")
        if twist["mean"] / light["mean"] < 1.5:
            print("NOTE: total power under the fallback is < 1.5x normal -> experiment 1 may be weak; "
                  "check its result, or re-measure with TWIST_KIND=xai (forward + gradient explanation)")
        if fps_t < fps_l - 1e-6:
            params["N_TWIST_MAX"] = int(round(fps_t * a.dt))
            print(f"  fallback path cannot hold {fps_l:.1f} fps -> N_TWIST_MAX = {params['N_TWIST_MAX']} per step")
    bad = []
    if e_inf <= 0 or light["mean"] - p_idle < 0.05:
        bad.append(f"the light run drew no more power than idle ({light['mean'] - p_idle:+.3f} W)")
    if twist and params.get("E_INF_TWIST", 0.0) <= e_inf:
        bad.append("the fallback run does not cost more energy per inference than the light run")
    if bad:
        print("ERROR: " + "; ".join(bad) + ".")
        print("  The model most likely did not run during the logs. Check mean_latency_ms in the counts CSVs:")
        print("  about 4 ms (light) and 12 ms (twist) are the template's placeholder sleeps, not inference.")
        if not a.force:
            print("  params_measured.json NOT written (use --force to write it anyway).")
            raise SystemExit(2)
    if a.n_low is not None:
        params["N_LOW"] = a.n_low
    if a.t is not None:
        params["T"] = a.t
    for name, s in (("idle", idle), ("light", light)) + ((("twist", twist),) if twist else ()):
        if s["minutes"] < 8.0:
            print(f"WARNING: only {s['minutes']:.1f} min of the {name} log remain after trimming; "
                  f"record 12 min per log")
    with open(a.out, "w") as f:
        json.dump(params, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats", help="mean / fluctuation of one rail in one log")
    s.add_argument("log")
    s.add_argument("--rail", default="VDD_IN")
    s.add_argument("--interval", type=float, default=1.0, help="seconds between tegrastats lines")
    s.add_argument("--skip-head", type=float, default=120.0, help="seconds dropped at the start")
    s.add_argument("--skip-tail", type=float, default=10.0, help="seconds dropped at the end")
    s.set_defaults(func=cmd_stats)
    b = sub.add_parser("build", help="compute the measured parameters and write a params JSON")
    b.add_argument("--idle", required=True, nargs="+", help="one or more idle logs (e.g. before and after)")
    b.add_argument("--light", required=True)
    b.add_argument("--twist", help="log of the fallback path (experiment 1)")
    b.add_argument("--counts-light", help="logger CSV of the light run (gives the achieved fps)")
    b.add_argument("--counts-twist", help="logger CSV of the twist run")
    b.add_argument("--fps-light", type=float, help="achieved fps of the light run if no CSV")
    b.add_argument("--fps-twist", type=float, help="achieved fps of the twist run if no CSV")
    b.add_argument("--c-ah", type=float, default=2.6, help="ASSUMED pack capacity [Ah]")
    b.add_argument("--n-cells", type=int, default=4, help="ASSUMED series cells")
    b.add_argument("--dt", type=float, default=60.0, help="simulation step [s]")
    b.add_argument("--n-low", type=int, help="inferences per step in the quiet phase")
    b.add_argument("--t", type=int, help="number of simulation steps")
    b.add_argument("--rail", default="VDD_IN")
    b.add_argument("--interval", type=float, default=1.0)
    b.add_argument("--skip-head", type=float, default=120.0, help="seconds dropped at the start of every log")
    b.add_argument("--skip-tail", type=float, default=10.0, help="seconds dropped at the end of every log")
    b.add_argument("--force", action="store_true", help="write the JSON even if the results look invalid")
    b.add_argument("--out", default="params_measured.json")
    b.set_defaults(func=cmd_build)
    q = sub.add_parser("plot", help="figure of the measured power / GPU traces for the report")
    q.add_argument("--idle", required=True, nargs="+")
    q.add_argument("--light", required=True)
    q.add_argument("--twist")
    q.add_argument("--rail", default="VDD_IN")
    q.add_argument("--interval", type=float, default=1.0)
    q.add_argument("--skip-head", type=float, default=120.0)
    q.add_argument("--skip-tail", type=float, default=10.0)
    q.add_argument("--out", default="figures/fig0_jetson_power.png")
    q.set_defaults(func=cmd_plot)
    a = ap.parse_args()
    a.func(a)
