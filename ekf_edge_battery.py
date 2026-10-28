"""
EKF estimation of battery SOC on a battery-powered edge-AI camera (Jetson Orin Nano class)
==========================================================================================
Intelligent Robotics midterm project.  Notation follows the lecture (03: KF/EKF) and the
assignment: R = process-noise covariance, Q = measurement-noise covariance.

  state        x_t : SOC [%], charge-based (remaining charge / capacity). Not measurable directly.
  control      u_t : energy drawn during step t [J] = P_idle*dt + n_t*E_inf
                     known from the device's own inference counter n_t. Like odometry, it
                     describes the change of state, so it is treated as control (lecture 02 p.47).
  motion       x_t = g(x_{t-1}, u_t) + eps_t,     eps_t ~ N(0, R_t)
               g(x, u) = x - 100*u / (3600*C_AH*h(x))
               constant-power load: at lower voltage the same energy costs more charge.
               G_t = dg/dx = 1 + 100*u*h'(x) / (3600*C_AH*h(x)^2)          (evaluated at mu_{t-1})
               R_t = R_PROC + (R_REL * d_t)^2,  d_t = x - g(x, u) (SOC drop of the step)
  measurement  z_t = h(x_t) + delta_t,            delta_t ~ N(0, Q_t)
               h(x) = N_CELLS*(3.0 + 3.1 s - 5.0 s^2 + 3.1 s^3),  s = x/100  (simplified OCV)
               H_t = dh/dx                                                 (evaluated at mu_bar_t)
  EKF          mu_bar = g(mu, u);  Sig_bar = G Sig G + R
               z_hat = h(mu_bar);  r = z - z_hat;  S = H Sig_bar H + Q
               K = Sig_bar H / S;  mu = mu_bar + K r;  Sig = (1 - K H) Sig_bar

The filter never receives the simulated true state x_t; it only uses u_t and z_t.

Parameters (see DEFAULTS):
  MEASURED on the Jetson (parse_tegrastats.py -> params_measured.json):
      P_IDLE, E_INF, E_INF_TWIST, R_PROC, N_HIGH, N_TWIST_MAX
  ASSUMED: battery (C_AH, N_CELLS), voltage-sensor noise Q_MEAS, R_REL, initial belief, scenario.

Run:
  python ekf_edge_battery.py                                   # placeholder defaults
  python ekf_edge_battery.py --params params_measured.json     # after measuring on the Jetson
Outputs: tables on stdout (save with `> run_output.txt`), figures in ./figures (or --outdir).
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")   # --outdir overrides

# ----------------------------------------------------------------------------
# 1. Parameters
# ----------------------------------------------------------------------------
DEFAULTS = dict(
    # ---- MEASURED on the device (placeholders of Orin-Nano magnitude) ------
    P_IDLE=4.5,          # [W] idle power (OS, camera, radio)
    E_INF=0.08,          # [J/inference] KD student, TensorRT FP16 (normal operation)
    E_INF_TWIST=0.35,    # [J/inference] same model after silent fallback (e.g. PyTorch FP32)
    R_PROC=6e-5,         # [%^2] minute-to-minute power fluctuation under steady load
    N_TWIST_MAX=None,    # [inferences/step] max rate the fallback path can sustain (None = no cap)
    # ---- ASSUMED -------------------------------------------------------------
    C_AH=2.6,            # [Ah] pack capacity (4S 2600 mAh ~ 38 Wh with this OCV curve)
    N_CELLS=4,           # series cells -> 12.0..16.8 V
    Q_MEAS=0.0016,       # [V^2] pack-voltage noise, sigma = 40 mV (ADC + ripple + unmodeled sag)
    R_REL=0.1,           # 10 % of each step's SOC drop is unknown (capacity, E_inf drift, DVFS)
    X0_TRUE=90.0,        # true initial SOC [%] (charged before installation)
    MU0=85.0,            # initial belief mean: last estimate stored by the OS
    SIG0=25.0,           # initial belief variance [%^2] (sigma = 5 %)
    # ---- scenario -----------------------------------------------------------
    DT=60.0,             # [s] one step = 1 minute
    T=180,               # steps (3 hours)
    N_HIGH=1800,         # inferences/step in busy phases (30 fps)
    N_LOW=300,           # inferences/step in the quiet phase (5 fps)
    SWITCH_T=91,         # experiment 1: silent fallback starts at this step
    R_INFLATE=10.0,      # experiment 1, filter C: R multiplied by this ("admit model doubt")
    ALARM_SOC=20.0,      # low-battery alarm threshold [%]
    T_EXP1=600,          # experiment 1 runs until the true SOC hits 0 % (at most T_EXP1 steps);
                         #   steps after T keep the busy rate
    # ---- appendix experiments -----------------------------------------------
    DROP_T0=60, DROP_T1=90,   # appendix A1: no measurement in [DROP_T0, DROP_T1]
    Q_HIGH_SIGMA=0.10,        # appendix A2: voltage noise sigma under load [V]
    # ---- evaluation ---------------------------------------------------------
    MC_RUNS=100,         # Monte Carlo repetitions
    SENS_RUNS=30,        # runs per cell of the R_REL x voltage-noise sensitivity table (0 = skip)
    NIS_WINDOW=10,       # moving-average window for the residual consistency check
    NIS_QUANTILE=0.995,  # detection threshold: this chi-square quantile of the W-step average
    NIS_PERSIST=5,       # flag only if the average stays above the threshold this many steps in a row
                         #   (W=10, 0.995, 5 -> ~3 % false alarms per 3-h run in Monte Carlo)
    SEED=7,
)
P = dict(DEFAULTS)


def load_params(path):
    """Override DEFAULTS from a JSON file. Keys starting with '_' are comments."""
    with open(path) as f:
        d = json.load(f)
    used, unknown = {}, []
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if k in DEFAULTS:
            P[k] = v
            used[k] = v
        else:
            unknown.append(k)
    print("Loaded parameters from", path, "->", used)
    if P["E_INF"] <= 0 or P["E_INF_TWIST"] <= P["E_INF"]:
        print("WARNING: E_INF <= 0 or E_INF_TWIST <= E_INF -- the inference load is not visible in these "
              "parameters, so experiment 1 has no effect. Re-measure before using the results.")
    if unknown:
        print("WARNING: unknown keys ignored (check spelling / old names):", unknown)


def phases():
    """Phase boundaries scaled with T: idle 1..a | busy a+1..b | quiet b+1..c | busy c+1..T."""
    T = P["T"]
    return int(T / 6), int(7 * T / 12), int(17 * T / 24)


# ----------------------------------------------------------------------------
# 2. Models
# ----------------------------------------------------------------------------
def h(x):
    """Measurement model: pack OCV [V] for SOC x [%]."""
    s = np.asarray(x) / 100.0
    return P["N_CELLS"] * (3.0 + 3.1 * s - 5.0 * s ** 2 + 3.1 * s ** 3)


def dh(x):
    """H = dh/dx [V per %]."""
    s = np.asarray(x) / 100.0
    return P["N_CELLS"] * (3.1 - 10.0 * s + 9.3 * s ** 2) / 100.0


_XS = np.linspace(0.0, 100.0, 20001)


def h_inv(z):
    """Voltage-only estimate x = h^-1(z) (h is strictly increasing); clipped to [0, 100]."""
    return np.interp(z, h(_XS), _XS)


def energy(n_t, e_inf):
    """Control u_t: energy drawn in one step [J]."""
    return P["P_IDLE"] * P["DT"] + n_t * e_inf


def drop(x, u):
    """SOC drop [%] in one step when energy u [J] is drawn at SOC x."""
    return 100.0 * u / (3600.0 * P["C_AH"] * h(x))


def g(x, u):
    """Motion model g(x_{t-1}, u_t)."""
    return x - drop(x, u)


def dg(x, u):
    """G = dg/dx."""
    return 1.0 + 100.0 * u * dh(x) / (3600.0 * P["C_AH"] * h(x) ** 2)


def proc_var(d):
    """R_t for a step whose SOC drop is d [%]."""
    return P["R_PROC"] + (P["R_REL"] * d) ** 2


def schedule(T=None):
    """n_t: inferences per step (index 0 unused). Phases follow P['T']; steps beyond it stay busy."""
    T = P["T"] if T is None else int(T)
    a, b, c = phases()
    n = np.zeros(T + 1)
    n[a + 1:b + 1] = P["N_HIGH"]
    n[b + 1:c + 1] = P["N_LOW"]
    n[c + 1:] = P["N_HIGH"]
    return n


def twist_world(n):
    """Experiment 1: from SWITCH_T the device silently runs the fallback path."""
    T = len(n) - 1
    sw = int(P["SWITCH_T"])
    e_true = np.full(T + 1, P["E_INF"], dtype=float)
    e_true[sw:] = P["E_INF_TWIST"]
    n_tw = n.copy()
    if P["N_TWIST_MAX"] is not None:
        n_tw[sw:] = np.minimum(n_tw[sw:], P["N_TWIST_MAX"])
    return n_tw, e_true


# ----------------------------------------------------------------------------
# 3. True world (the filter never sees x)
# ----------------------------------------------------------------------------
def simulate_truth(n, e_inf_true, rng, q_sigma=None):
    T = len(n) - 1
    if q_sigma is None:
        q_sigma = np.full(T + 1, np.sqrt(P["Q_MEAS"]))
    x = np.zeros(T + 1)
    z = np.full(T + 1, np.nan)
    x[0] = P["X0_TRUE"]
    for t in range(1, T + 1):
        u = energy(n[t], e_inf_true[t])
        d = drop(x[t - 1], u)
        x[t] = max(x[t - 1] - d + rng.normal(0.0, np.sqrt(proc_var(d))), 0.0)
        z[t] = h(x[t]) + rng.normal(0.0, q_sigma[t])
    return x, z


# ----------------------------------------------------------------------------
# 4. EKF (lecture 03 p.64-66)
# ----------------------------------------------------------------------------
def ekf_run(z, u, R_scale=1.0, Q=None, available=None):
    """
    z         : measurements [V] (NaN = none)
    u         : control the FILTER believes [J] (from its counter and its E_inf table)
    R_scale   : multiplies the filter's R_t (experiment 1, filter C)
    Q         : scalar or per-step array of measurement-noise variance assumed by the filter
    available : bool mask; False -> no correction (bel = bel_bar)
    """
    T = len(z) - 1
    Q = P["Q_MEAS"] if Q is None else Q
    Q = np.full(T + 1, Q) if np.isscalar(Q) else np.asarray(Q, dtype=float)
    if available is None:
        available = np.ones(T + 1, dtype=bool)
    keys = ["mu_prev", "sig_prev", "u", "G", "R", "mu_bar", "sig_bar", "H", "z_hat",
            "z", "r", "S", "K", "KH", "mu", "sig", "nis"]
    L = {k: np.full(T + 1, np.nan) for k in keys}
    mu, sig = P["MU0"], P["SIG0"]
    L["mu"][0], L["sig"][0] = mu, sig
    for t in range(1, T + 1):
        L["mu_prev"][t], L["sig_prev"][t], L["u"][t] = mu, sig, u[t]
        # ---- prediction: motion moves and spreads the belief ----
        G = dg(mu, u[t])
        d = drop(mu, u[t])
        mu_bar = mu - d                          # = g(mu, u)
        R_t = R_scale * proc_var(d)
        sig_bar = G * sig * G + R_t
        L["G"][t], L["R"][t], L["mu_bar"][t], L["sig_bar"][t] = G, R_t, mu_bar, sig_bar
        # ---- correction: measurement pulls and sharpens the belief ----
        if available[t] and not np.isnan(z[t]):
            H = dh(mu_bar)
            z_hat = h(mu_bar)                    # predicted measurement
            r = z[t] - z_hat                     # measurement residual
            S = H * sig_bar * H + Q[t]
            K = sig_bar * H / S
            mu = mu_bar + K * r
            sig = (1.0 - K * H) * sig_bar
            L["H"][t], L["z_hat"][t], L["z"][t], L["r"][t] = H, z_hat, z[t], r
            L["S"][t], L["K"][t], L["KH"][t], L["nis"][t] = S, K, K * H, r * r / S
        else:
            mu, sig = mu_bar, sig_bar
        L["mu"][t], L["sig"][t] = mu, sig
    return L


# ----------------------------------------------------------------------------
# 5. Evaluation helpers
# ----------------------------------------------------------------------------
def rmse(est, x, t0=1, t1=None):
    t1 = len(x) - 1 if t1 is None else t1
    return float(np.sqrt(np.mean((est[t0:t1 + 1] - x[t0:t1 + 1]) ** 2)))


def coverage(L, x, t0=5):
    """Fraction of steps where the truth lies inside mu +/- 2 sigma (consistent filter: ~0.95)."""
    e = np.abs(L["mu"][t0:] - x[t0:])
    return float(np.mean(e <= 2.0 * np.sqrt(L["sig"][t0:])))


def first_below(series, thr):
    idx = np.where(series[1:] < thr)[0]
    return int(idx[0] + 1) if len(idx) else None


def alarms(L, x):
    """Low-battery alarm. truth: x crosses ALARM_SOC; mean: mu < thr; cons: mu - 2 sigma < thr
    (conservative, lecture 02 p.74); death: x reaches 0 %. lead_* = minutes of warning before death."""
    thr = P["ALARM_SOC"]
    out = dict(truth=first_below(x, thr), mean=first_below(L["mu"], thr),
               cons=first_below(L["mu"] - 2.0 * np.sqrt(L["sig"]), thr), death=first_below(x, 1e-9))
    for k in ("truth", "mean", "cons"):
        out["lead_" + k] = (out["death"] - out[k]) if (out["death"] is not None and out[k] is not None) else None
    return out


def nis_ma(L, W):
    """Moving average of r^2/S over W steps (consistent filter: ~1)."""
    nis = L["nis"]
    out = np.full_like(nis, np.nan)
    for t in range(W, len(nis)):
        w = nis[t - W + 1:t + 1]
        if not np.any(np.isnan(w)):
            out[t] = w.mean()
    return out


def detect_time(L, t0):
    """Decision time (>= t0) at which the moving average of r^2/S has stayed above the
    95 % band for NIS_PERSIST consecutive steps (persistence suppresses chance exceedances)."""
    W, k = int(P["NIS_WINDOW"]), int(P["NIS_PERSIST"])
    above = nis_ma(L, W) > nis_threshold(W)
    run = 0
    for t in range(t0, len(above)):
        run = run + 1 if above[t] else 0
        if run >= k:
            return t
    return None


def false_flag(L, t0=5):
    """Would the persistent residual check fire on a consistent filter (baseline)? 1 = false alarm."""
    return 0.0 if detect_time(L, t0) is None else 1.0


def nis_bounds(W):
    """95 % band of the W-step average of r^2/S for a consistent filter (chi-square(W)/W)."""
    try:
        from scipy.stats import chi2
        return chi2.ppf(0.025, W) / W, chi2.ppf(0.975, W) / W
    except Exception:          # fallback for W = 10
        return 0.3247, 2.0483


def nis_threshold(W):
    """Detection threshold: NIS_QUANTILE of chi-square(W)/W."""
    try:
        from scipy.stats import chi2
        return chi2.ppf(P["NIS_QUANTILE"], W) / W
    except Exception:          # fallback for W = 10, quantile 0.995
        return 2.5188


def exact_correction(mu_bar, sig_bar, z, Q):
    """Exact Bayes correction on a grid for the same predicted belief (checks the EKF linearization)."""
    xs = np.linspace(0.0, 100.0, 200001)
    logp = -0.5 * (xs - mu_bar) ** 2 / sig_bar - 0.5 * (z - h(xs)) ** 2 / Q
    w = np.exp(logp - logp.max())
    w /= w.sum()
    m = float((w * xs).sum())
    v = float((w * (xs - m) ** 2).sum())
    return m, v, xs, w / (xs[1] - xs[0])


def gauss(xs, m, v):
    return np.exp(-0.5 * (xs - m) ** 2 / v) / np.sqrt(2 * np.pi * v)


# ----------------------------------------------------------------------------
# 6. Printing
# ----------------------------------------------------------------------------
def print_cards(L, x, steps, title):
    """Step-by-step calculation in the order of the assignment example:
       Jacobian -> Prediction -> Kalman gain -> Update."""
    print("\n" + "=" * 96 + "\n" + title + "\n" + "=" * 96)
    for t in steps:
        print(f"t = {t}   previous belief N({L['mu_prev'][t]:.3f}, {L['sig_prev'][t]:.4f}),  "
              f"u_t = {L['u'][t]:.1f} J,  true x_t = {x[t]:.3f}")
        print(f"  Jacobian   : G = {L['G'][t]:.5f}   H = dh/dx(mu_bar) = {L['H'][t]*1000:.3f} mV/%")
        print(f"  Prediction : mu_bar = g(mu, u) = {L['mu_prev'][t]:.3f} - {L['mu_prev'][t]-L['mu_bar'][t]:.4f} "
              f"= {L['mu_bar'][t]:.3f}   Sig_bar = G^2*{L['sig_prev'][t]:.4f} + R({L['R'][t]:.5f}) "
              f"= {L['sig_bar'][t]:.4f}")
        print(f"  Measurement: z_hat = h(mu_bar) = {L['z_hat'][t]:.4f} V   z = {L['z'][t]:.4f} V   "
              f"r = z - z_hat = {L['r'][t]*1000:+.1f} mV")
        print(f"  Gain       : S = H^2*Sig_bar + Q = {L['S'][t]:.6f}   K = Sig_bar*H/S = {L['K'][t]:.3f} %/V   "
              f"K*H = {L['KH'][t]:.3f}")
        print(f"  Update     : mu = mu_bar + K*r = {L['mu_bar'][t]:.3f} {L['K'][t]*L['r'][t]:+.3f} "
              f"= {L['mu'][t]:.3f}   Sig = (1-K*H)*Sig_bar = {L['sig'][t]:.4f}  (sigma {np.sqrt(L['sig'][t]):.3f} %)")


def print_param_summary():
    n = schedule()
    a, b, c = phases()
    mid = 50.0
    print("\nPARAMETER SUMMARY (MEASURED: P_IDLE, E_INF, E_INF_TWIST, R_PROC, N_HIGH, N_TWIST_MAX)")
    print(f"  pack {P['N_CELLS']}S {P['C_AH']} Ah, OCV {float(h(0)):.2f}..{float(h(100)):.2f} V, "
          f"usable energy ~ {P['C_AH']*float(np.mean(h(_XS))):.1f} Wh")
    for name, nn, e in [("idle", 0, P["E_INF"]), ("busy, normal", P["N_HIGH"], P["E_INF"]),
                        ("busy, fallback", P["N_HIGH"] if P["N_TWIST_MAX"] is None else min(P["N_HIGH"], P["N_TWIST_MAX"]),
                         P["E_INF_TWIST"])]:
        u = energy(nn, e)
        print(f"  {name:<15s}: u = {u:7.1f} J/step ({u/P['DT']:5.2f} W)  SOC drop at 90/50/20 % = "
              f"{float(drop(90, u)):.3f}/{float(drop(50, u)):.3f}/{float(drop(20, u)):.3f} %/step  "
              f"G at 50 % = {float(dg(50, u)):.5f}")
    u_hi = energy(P["N_HIGH"], P["E_INF"])
    print(f"  process noise sigma (busy, 50 %) = {np.sqrt(proc_var(float(drop(50, u_hi)))):.4f} %/step  "
          f"[R_PROC={P['R_PROC']:g}, R_REL={P['R_REL']:g}]")
    sq = np.sqrt(P["Q_MEAS"])
    print(f"  measurement sigma = {sq*1000:.0f} mV -> one reading pins SOC to +/- "
          f"{sq/float(dh(90)):.2f} % at 90 %, +/- {sq/float(dh(50)):.2f} % at 50 %, +/- {sq/float(dh(20)):.2f} % at 20 %")
    print(f"  schedule: idle 1..{a}, busy {a+1}..{b}, quiet {b+1}..{c}, busy {c+1}..{P['T']};  "
          f"fallback from t={P['SWITCH_T']};  alarm at {P['ALARM_SOC']} %")
    xs = P["X0_TRUE"]
    for t in range(1, P["T"] + 1):
        xs -= float(drop(xs, energy(n[t], P["E_INF"])))
    print(f"  noise-free baseline SOC at t={P['T']}: {xs:.1f} %")


# ----------------------------------------------------------------------------
# 7. Experiments
# ----------------------------------------------------------------------------
def baseline(seed):
    rng = np.random.default_rng(seed)
    n = schedule()
    T = P["T"]
    x, z = simulate_truth(n, np.full(T + 1, P["E_INF"]), rng)
    u = np.array([energy(n[t], P["E_INF"]) for t in range(T + 1)])
    return n, x, z, u, ekf_run(z, u)


def experiment1(seed):
    rng = np.random.default_rng(seed)
    n = schedule(P["T_EXP1"])
    n_tw, e_true = twist_world(n)
    x, z = simulate_truth(n_tw, e_true, rng)
    dead = first_below(x, 1e-9)                    # device shuts down when the true SOC hits 0 %
    if dead is not None:
        n_tw, e_true, x, z = n_tw[:dead + 1], e_true[:dead + 1], x[:dead + 1], z[:dead + 1]
    T = len(x) - 1
    u_true = np.array([energy(n_tw[t], e_true[t]) for t in range(T + 1)])
    u_believed = np.array([energy(n_tw[t], P["E_INF"]) for t in range(T + 1)])   # counter is right, E_inf table is stale
    A = ekf_run(z, u_true)                                  # told about the fallback
    B = ekf_run(z, u_believed)                              # not told, same R
    C = ekf_run(z, u_believed, R_scale=P["R_INFLATE"])      # not told, admits model doubt
    return n_tw, x, z, A, B, C


def monte_carlo(runs):
    sw = int(P["SWITCH_T"])
    base_rmse, base_cov, cc_rmse, v_rmse = [], [], [], []
    res = {k: dict(rmse=[], cov=[], delay=[], lead=[], late=0, none=0, det=[]) for k in "ABC"}
    ideal, false_rate = [], []
    for k in range(runs):
        seed = 10_000 + k
        n, x, z, u, L = baseline(seed)
        base_rmse.append(rmse(L["mu"], x, 5))
        base_cov.append(coverage(L, x, 5))
        cc = ekf_run(z, u, available=np.zeros(len(z), dtype=bool))
        cc_rmse.append(rmse(cc["mu"], x, 5))
        v_rmse.append(rmse(h_inv(z), x, 5))
        false_rate.append(false_flag(L, 5))
        _, x1, _, A, B, C = experiment1(seed)
        for name, Lf in zip("ABC", (A, B, C)):
            dt_ = detect_time(Lf, sw)
            res[name]["det"].append(np.nan if dt_ is None else dt_ - sw)
            res[name]["rmse"].append(rmse(Lf["mu"], x1, sw))
            res[name]["cov"].append(coverage(Lf, x1, sw))
            al = alarms(Lf, x1)
            if name == "A" and al["lead_truth"] is not None:
                ideal.append(al["lead_truth"])
            if al["truth"] is None:
                continue
            if al["cons"] is None:                       # no alarm before the shutdown
                res[name]["none"] += 1
                res[name]["delay"].append(np.nan)
                res[name]["lead"].append(0.0)
            else:
                res[name]["delay"].append(al["cons"] - al["truth"])
                res[name]["lead"].append(al["lead_cons"] if al["lead_cons"] is not None else np.nan)
    def ms(a, fmt="6.3f"):
        a = np.asarray(a, dtype=float)
        if a.size == 0 or np.all(np.isnan(a)):
            return "   n/a"
        return f"{np.nanmean(a):{fmt}} +/- {np.nanstd(a):.3f}"
    print("\n" + "=" * 96 + f"\nMONTE CARLO ({runs} runs)\n" + "=" * 96)
    print(f"  baseline EKF           RMSE = {ms(base_rmse)} %   2-sigma coverage = {ms(base_cov)}")
    print(f"  coulomb counting only  RMSE = {ms(cc_rmse)} %   (prediction only, never corrected)")
    print(f"  voltage only h^-1(z)   RMSE = {ms(v_rmse)} %   (measurement only)")
    print(f"  residual check on the baseline (no hidden change): false alarm in {int(np.sum(false_rate))}/{runs} runs")
    print(f"  experiment 1 (t >= {sw}):")
    print(f"    ideal warning (truth crosses {P['ALARM_SOC']:.0f} % -> shutdown): {ms(ideal, '6.1f')} min")
    for name in "ABC":
        r = res[name]
        print(f"    {name}: RMSE = {ms(r['rmse'])} %  2-sigma coverage = {ms(r['cov'])}  "
              f"alarm (mu-2s) delay = {ms(r['delay'], '+6.1f')} min, warning before shutdown = "
              f"{ms(r['lead'], '6.1f')} min, no alarm before shutdown in {r['none']}/{runs} runs")
        d = np.asarray(r["det"], dtype=float)
        print(f"       residual check flags the change after {ms(d, '6.1f')} min "
              f"(never flagged in {int(np.isnan(d).sum())}/{runs} runs)")


def sensitivity(runs):
    """Experiment 1 under other ASSUMED values of R_REL and the voltage noise (world and filter alike).
    Shows which conclusions are robust and which magnitudes depend on the assumptions."""
    base = dict(P)
    sw = int(P["SWITCH_T"])
    print("\n" + "=" * 96 + f"\nSENSITIVITY of experiment 1 to the assumed R_REL and voltage noise ({runs} runs per cell)\n" + "=" * 96)
    print("  R_REL  noise[mV] |  A: 2s-cover  alarm delay |  B: 2s-cover  alarm delay  max err[%] |  C: alarm delay")
    for rrel in (0.05, 0.1, 0.2):
        for qmv in (20, 40, 80):
            P.update(R_REL=rrel, Q_MEAS=(qmv / 1000.0) ** 2)
            cov = {k: [] for k in "ABC"}
            dl = {k: [] for k in "ABC"}
            bmax = []
            for k in range(runs):
                _, x1, _, A, B, C = experiment1(10_000 + k)
                bmax.append(np.max(np.abs(B["mu"][sw:] - x1[sw:])))
                for name, Lf in zip("ABC", (A, B, C)):
                    cov[name].append(coverage(Lf, x1, sw))
                    al = alarms(Lf, x1)
                    dl[name].append(np.nan if (al["cons"] is None or al["truth"] is None) else al["cons"] - al["truth"])
            mark = "  <- assumed in this report" if (rrel == base["R_REL"] and abs((qmv / 1000.0) ** 2 - base["Q_MEAS"]) < 1e-12) else ""
            print(f"   {rrel:4.2f}   {qmv:5d}    |     {np.mean(cov['A']):.2f}       {np.nanmean(dl['A']):+5.1f}    |"
                  f"     {np.mean(cov['B']):.2f}       {np.nanmean(dl['B']):+5.1f}      {np.mean(bmax):5.2f}   |"
                  f"     {np.nanmean(dl['C']):+5.1f}{mark}")
    P.clear()
    P.update(base)


# ----------------------------------------------------------------------------
# 8. Figures
# ----------------------------------------------------------------------------
def fig_models():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    xs = np.linspace(0, 100, 401)
    ax[0].plot(xs, h(xs), lw=2)
    ax[0].set(xlabel="SOC x [%]", ylabel="pack OCV h(x) [V]", title=f"Measurement model h(x)  ({P['N_CELLS']}S)")
    ax[1].plot(xs, dh(xs) * 1000, lw=2, color="tab:red")
    sq = np.sqrt(P["Q_MEAS"]) * 1000
    hmax = float(dh(0)) * 1000                       # place labels inside the U, clear of the curve
    for xx, pos in ((20, (26, 0.82 * hmax)), (90, (44, 0.69 * hmax)), (50, (33, 0.40 * hmax))):
        ax[1].annotate(f"{xx}%: 1 reading = +/-{sq/(float(dh(xx))*1000):.1f}% SOC",
                       (xx, float(dh(xx)) * 1000), textcoords="data", xytext=pos, fontsize=8,
                       arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        ax[1].plot(xx, float(dh(xx)) * 1000, "o", color="tab:red", ms=4)
    ax[1].set(xlabel="SOC x [%]", ylabel="H = dh/dx [mV per %]",
              title=f"Jacobian H (measurement noise sigma = {sq:.0f} mV)")
    for name, nn, e, col in [("idle", 0, P["E_INF"], "tab:gray"), ("busy, normal", P["N_HIGH"], P["E_INF"], "tab:blue"),
                             ("busy, fallback", P["N_HIGH"] if P["N_TWIST_MAX"] is None else min(P["N_HIGH"], P["N_TWIST_MAX"]),
                              P["E_INF_TWIST"], "tab:orange")]:
        u = energy(nn, e)
        ax[2].plot(xs[xs >= 5], drop(xs[xs >= 5], u), color=col, lw=2, label=f"{name} ({u/P['DT']:.1f} W)")
    ax[2].set(xlabel="SOC x [%]", ylabel="SOC drop per step [%]",
              title="Motion model: x - g(x,u) (constant-power load)")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_models.png"), dpi=150)
    plt.close(fig)


def fig_baseline(n, x, z, L, cc, vonly):
    t = np.arange(len(x))
    fig, ax = plt.subplots(4, 1, figsize=(10, 11.5), sharex=True)
    s2 = 2 * np.sqrt(L["sig"])
    ax[0].plot(t, x, "k", lw=2, label="true SOC x_t (hidden from the filter)")
    ax[0].plot(t, L["mu"], "tab:blue", lw=1.6, label="EKF mu_t")
    ax[0].fill_between(t, L["mu"] - s2, L["mu"] + s2, color="tab:blue", alpha=0.2, label="mu_t +/- 2 sigma")
    ax[0].plot(t, cc["mu"], "--", color="tab:gray", lw=1.3, label="prediction only (coulomb counting)")
    ax[0].plot(t[1:], vonly[1:], ".", color="tab:green", ms=3, alpha=0.6, label="measurement only h^-1(z_t)")
    ax[0].set(ylabel="SOC [%]", title="Baseline: prediction only vs measurement only vs EKF")
    ax[0].legend(fontsize=8, loc="lower left")
    tw = ax[0].twinx()
    tw.step(t, n, where="mid", color="tab:orange", alpha=0.45, lw=1)
    tw.set_ylabel("inferences / step", color="tab:orange")
    ax[1].plot(t, z, ".", ms=3, color="tab:green", label="measured z_t")
    ax[1].plot(t, h(x), "k", lw=1, label="h(x_t) (noise-free)")
    ax[1].set(ylabel="pack voltage [V]")
    ax[1].legend(fontsize=8)
    ax[2].plot(t, L["mu"] - x, "tab:blue", lw=1.4, label="EKF error mu_t - x_t")
    ax[2].plot(t, cc["mu"] - x, "--", color="tab:gray", lw=1.1, label="prediction-only error")
    ax[2].fill_between(t, -s2, s2, color="tab:blue", alpha=0.2, label="+/- 2 sigma (filter's own claim)")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set(ylabel="error [%]", ylim=(-7, 7))
    ax[2].legend(fontsize=8, loc="lower right")
    ax[3].plot(t, L["KH"], "tab:purple", lw=1.4)
    ax[3].set(yscale="log", ylim=(1e-3, 1.5), ylabel="K*H (share of the\nmeasurement accepted, log)",
              xlabel="time step t [min]")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_baseline.png"), dpi=150)
    plt.close(fig)


def fig_steps(L, x, steps):
    """Lecture 03 p.52 style: predicted belief, likelihood on x, posterior (EKF and exact)."""
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.ravel()
    for i, t in enumerate(steps):
        a = axs[i]
        mb, sb, H, r, Q = L["mu_bar"][t], L["sig_bar"][t], L["H"][t], L["r"][t], P["Q_MEAS"]
        lm, lv = mb + r / H, Q / H ** 2                       # linearized likelihood on x
        em, ev, gx, gd = exact_correction(mb, sb, L["z"][t], Q)
        sd = np.sqrt(max(sb, lv))
        lo, hi = max(min(mb, lm, x[t]) - 3.5 * sd, 0.0), min(max(mb, lm, x[t]) + 3.5 * sd, 100.0)
        xs = np.linspace(lo, hi, 800)
        a.plot(xs, gauss(xs, mb, sb), "--", color="tab:orange", label=f"predicted N({mb:.2f}, {sb:.3f})")
        a.plot(xs, gauss(xs, lm, lv), ":", color="tab:green", lw=2, label=f"likelihood on x N({lm:.2f}, {lv:.3f})")
        a.plot(xs, gauss(xs, L["mu"][t], L["sig"][t]), color="tab:blue", lw=2,
               label=f"EKF posterior N({L['mu'][t]:.2f}, {L['sig'][t]:.3f})")
        sel = (gx >= lo) & (gx <= hi)
        a.plot(gx[sel], gd[sel], color="k", lw=1, label=f"exact posterior N({em:.2f}, {ev:.3f})")
        a.axvline(x[t], color="gray", lw=1, ls="-.", label=f"true x = {x[t]:.2f}")
        peak = max(gauss(xs, mb, sb).max(), gauss(xs, lm, lv).max(),
                   gauss(xs, L["mu"][t], L["sig"][t]).max(), gd[sel].max() if sel.any() else 0.0)
        a.set_ylim(0, 1.65 * peak)                            # headroom for the legend
        a.set_title(f"t = {t}:  K*H = {L['KH'][t]:.3f},  H = {H*1000:.1f} mV/%", fontsize=10)
        a.set_xlabel("SOC [%]")
        a.legend(fontsize=7, loc="upper left")
        a.grid(alpha=0.3)
    for j in range(len(steps), len(axs)):
        axs[j].axis("off")
    axs[-1].text(0.0, 0.5,
                 "Prediction moves and spreads the belief (orange).\n"
                 "The measurement, mapped to SOC by the local slope H,\n"
                 "gives the likelihood on x (green): narrow where h is steep,\n"
                 "wide where h is flat.  The EKF posterior (blue) lies between,\n"
                 "weighted by K*H = Sig_bar / (Sig_bar + Q/H^2).\n"
                 "Black = exact Bayes correction of the same prediction:\n"
                 "differs from the EKF only when the prediction is wide (t = 1).",
                 fontsize=9, va="center", family="monospace")
    fig.suptitle("Prediction -> Correction for consecutive steps")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_steps.png"), dpi=150)
    plt.close(fig)


def fig_exp1(x, A, B, C):
    t = np.arange(len(x))
    sw, thr, W = int(P["SWITCH_T"]), P["ALARM_SOC"], int(P["NIS_WINDOW"])
    fig, ax = plt.subplots(4, 1, figsize=(10, 13), sharex=True)
    ax[0].plot(t, x, "k", lw=2, label="true SOC")
    styles = [("A", A, "tab:green", "-", "A: told about the fallback"),
              ("B", B, "tab:red", "-", "B: not told, R as modeled"),
              ("C", C, "tab:orange", "--", f"C: not told, R x{P['R_INFLATE']:g}")]
    for name, L, col, ls, lab in styles:
        ax[0].plot(t, L["mu"], color=col, ls=ls, lw=1.4, label=lab)
    ax[0].axvline(sw, color="gray", ls=":", lw=1)
    ax[0].axhline(thr, color="tab:purple", ls=":", lw=1)
    al0 = alarms(A, x)
    al_t, dead = al0["truth"], al0["death"]
    if al_t:
        ax[0].plot(al_t, x[al_t], "kv", ms=9, label=f"truth crosses {thr:.0f} % (t={al_t})")
    if dead:
        for a_ in ax:
            a_.axvline(dead, color="k", lw=1.2)
        ax[0].text(dead - 1, 0.9 * P["X0_TRUE"], f"battery empty\n(t={dead})", ha="right", fontsize=8)
    for name, L, col, ls, lab in styles:
        al = alarms(L, x)["cons"]
        if al:
            ax[0].plot(al, L["mu"][al], "v", color=col, ms=8, label=f"{name}: alarm mu-2s<{thr:.0f} % (t={al})")
        else:
            ax[0].plot([], [], "v", color=col, ms=8, label=f"{name}: no alarm before shutdown")
    ax[0].set(ylabel="SOC [%]", title="Experiment 1: silent fallback (hidden software state) from t = %d" % sw)
    ax[0].legend(fontsize=8, loc="lower left")
    for name, L, col, ls, lab in styles:
        ax[1].plot(t, L["mu"] - x, color=col, ls=ls, lw=1.4, label=f"{name}: error")
    sB = 2 * np.sqrt(B["sig"])
    ax[1].fill_between(t, -sB, sB, color="tab:red", alpha=0.15, label="B: +/- 2 sigma claimed by the filter")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].axvline(sw, color="gray", ls=":", lw=1)
    m = np.nanmax(np.abs(np.r_[A["mu"][2:] - x[2:], B["mu"][2:] - x[2:], C["mu"][2:] - x[2:]]))
    ax[1].set(ylabel="error [%]", ylim=(-max(6, 1.15 * m), max(6, 1.15 * m)))
    ax[1].legend(fontsize=8, loc="upper left")
    ax[2].plot(t, A["r"] * 1000, color="tab:green", lw=1, alpha=0.8, label="A: residual r_t")
    ax[2].plot(t, B["r"] * 1000, color="tab:red", lw=1, label="B: residual r_t")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].axvline(sw, color="gray", ls=":", lw=1)
    ax[2].set(ylabel="r_t = z_t - h(mu_bar_t) [mV]")
    ax[2].legend(fontsize=8)
    lo, hi = nis_bounds(W)
    ax[3].plot(t, nis_ma(A, W), color="tab:green", lw=1.4, label="A")
    ax[3].plot(t, nis_ma(B, W), color="tab:red", lw=1.4, label="B")
    ax[3].plot(t, nis_ma(C, W), color="tab:orange", ls="--", lw=1.2, label="C")
    ax[3].axhspan(lo, hi, color="gray", alpha=0.15, label=f"95 % band for a consistent filter ({lo:.2f}..{hi:.2f})")
    thr_n = nis_threshold(W)
    ax[3].axhline(thr_n, color="k", ls="--", lw=1,
                  label=f"detection threshold {thr_n:.2f} (held {int(P['NIS_PERSIST'])} steps)")
    tdB = detect_time(B, sw)
    if tdB:
        ax[3].plot(tdB, nis_ma(B, W)[tdB], "r*", ms=12, label=f"B flagged at t={tdB}")
    ax[3].axvline(sw, color="gray", ls=":", lw=1)
    ax[3].set(yscale="log", ylabel=f"mean of r^2/S over {W} steps", xlabel="time step t [min]")
    ax[3].legend(fontsize=8, loc="upper left")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_exp1_fallback.png"), dpi=150)
    plt.close(fig)


def fig_appendix(n, x, z, u, L2, xq, Lq_known, Lq_unknown):
    t = np.arange(len(x))
    d0, d1 = int(P["DROP_T0"]), int(P["DROP_T1"])
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    s2 = 2 * np.sqrt(L2["sig"])
    ax[0].plot(t, L2["mu"] - x, "tab:blue", lw=1.4, label="error (no measurement in the shaded interval)")
    ax[0].fill_between(t, -s2, s2, color="tab:blue", alpha=0.2, label="+/- 2 sigma")
    ax[0].axvspan(d0, d1, color="gray", alpha=0.15)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set(ylabel="error [%]", ylim=(-1.5, 1.5),
              title="Appendix A1: measurement dropout (prediction only); initial transient off-scale")
    ax[0].legend(fontsize=8)
    ax[1].plot(t, Lq_known["mu"] - xq, "tab:green", lw=1.4, label="filter knows the noise grew")
    ax[1].plot(t, Lq_unknown["mu"] - xq, "tab:red", lw=1.2, label="filter keeps the old Q")
    ax[1].axhline(0, color="k", lw=0.8)
    a, _, _ = phases()
    ax[1].axvspan(a + 1, P["T"], color="gray", alpha=0.1)
    ax[1].set(ylabel="error [%]", xlabel="time step t [min]", ylim=(-1.5, 1.5),
              title=f"Appendix A2: voltage noise {np.sqrt(P['Q_MEAS'])*1000:.0f} -> {P['Q_HIGH_SIGMA']*1000:.0f} mV under load")
    ax[1].legend(fontsize=8)
    for a_ in ax:
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figA_appendix.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 9. Main
# ----------------------------------------------------------------------------
def main():
    T, sw = P["T"], int(P["SWITCH_T"])
    print_param_summary()

    # ---------------- baseline ----------------
    n, x, z, u, L = baseline(P["SEED"])
    cc = ekf_run(z, u, available=np.zeros(T + 1, dtype=bool))       # prediction only
    vonly = h_inv(z)                                                 # measurement only
    print_cards(L, x, [1, 2, 3, 4], "BASELINE, steps 1-4 (for section (c))")
    print(f"\nBaseline (t >= 5): RMSE EKF {rmse(L['mu'], x, 5):.3f} %,  prediction only {rmse(cc['mu'], x, 5):.3f} %,  "
          f"measurement only {rmse(vonly, x, 5):.3f} %,  2-sigma coverage {coverage(L, x, 5):.2f}")

    # linearization check: same predicted belief, exact vs EKF correction
    plateau = int(np.nanargmin(np.abs(L["mu_bar"][5:] - 53.8)) + 5)
    print("\nLINEARIZATION CHECK (exact Bayes correction of the same predicted belief)")
    for t in (1, plateau):
        em, ev, _, _ = exact_correction(L["mu_bar"][t], L["sig_bar"][t], L["z"][t], P["Q_MEAS"])
        print(f"  t={t:3d}: predicted N({L['mu_bar'][t]:.2f}, {L['sig_bar'][t]:.3f})  EKF N({L['mu'][t]:.3f}, {L['sig'][t]:.4f})  "
              f"exact N({em:.3f}, {ev:.4f})  h^-1(z) = {float(h_inv(L['z'][t])):.2f}  true {x[t]:.2f}  "
              f"slope at mu_bar {L['H'][t]*1000:.1f} vs at h^-1(z) {float(dh(h_inv(L['z'][t])))*1000:.1f} mV/%")

    fig_models()
    fig_baseline(n, x, z, L, cc, vonly)
    fig_steps(L, x, [1, 2, 3, 4, plateau])

    # ---------------- experiment 1 ----------------
    n_tw, x1, z1, A, B, C = experiment1(P["SEED"])
    print_cards(B, x1, [sw, sw + 1, sw + 2, sw + 3], f"EXPERIMENT 1, filter B (not told), steps {sw}-{sw+3}")
    W = int(P["NIS_WINDOW"])
    al0 = alarms(A, x1)
    print(f"\nExperiment 1 (t >= {sw}): truth crosses {P['ALARM_SOC']:.0f} % at t={al0['truth']}, "
          f"battery empty (shutdown) at t={al0['death']}")
    print(f"  residual check (mean r^2/S over {W} steps above {nis_threshold(W):.2f} for {int(P['NIS_PERSIST'])} steps): "
          f"B flagged at t={detect_time(B, sw)}, "
          f"C at t={detect_time(C, sw)}, A at t={detect_time(A, sw)}")
    for name, Lf in zip("ABC", (A, B, C)):
        al = alarms(Lf, x1)
        print(f"  {name}: RMSE {rmse(Lf['mu'], x1, sw):.3f} %  max|err| {np.max(np.abs(Lf['mu'][sw:]-x1[sw:])):.2f} %  "
              f"2-sigma coverage {coverage(Lf, x1, sw):.2f}  mean K*H {np.nanmean(Lf['KH'][sw:]):.3f}  "
              f"mean r^2/S {np.nanmean(Lf['nis'][sw:]):.2f}  | alarm(mu) t={al['mean']}, alarm(mu-2s) t={al['cons']}, "
              f"warning before shutdown {al['lead_cons']} min")
    fig_exp1(x1, A, B, C)

    # ---------------- appendix ----------------
    avail = np.ones(T + 1, dtype=bool)
    avail[int(P["DROP_T0"]):int(P["DROP_T1"]) + 1] = False
    L2 = ekf_run(z, u, available=avail)
    d0, d1 = int(P["DROP_T0"]), int(P["DROP_T1"])
    print(f"\nAppendix A1 (no measurement t={d0}..{d1}): sigma {np.sqrt(L2['sig'][d0-1]):.3f} -> "
          f"{np.sqrt(L2['sig'][d1]):.3f} % at the end of the gap -> {np.sqrt(L2['sig'][d1+3]):.3f} % after 3 corrections;"
          f"  RMSE {rmse(L2['mu'], x, 5):.3f} % (baseline {rmse(L['mu'], x, 5):.3f} %)")
    a, _, _ = phases()
    rng3 = np.random.default_rng(P["SEED"])
    q_sig = np.full(T + 1, np.sqrt(P["Q_MEAS"]))
    q_sig[a + 1:] = P["Q_HIGH_SIGMA"]
    xq, zq = simulate_truth(n, np.full(T + 1, P["E_INF"]), rng3, q_sigma=q_sig)
    Lk = ekf_run(zq, u, Q=q_sig ** 2)
    Lu = ekf_run(zq, u)
    print(f"Appendix A2 (noise {np.sqrt(P['Q_MEAS'])*1000:.0f} -> {P['Q_HIGH_SIGMA']*1000:.0f} mV from t={a+1}): "
          f"RMSE knows {rmse(Lk['mu'], xq, a+1):.3f} %, keeps old Q {rmse(Lu['mu'], xq, a+1):.3f} %")
    fig_appendix(n, x, z, u, L2, xq, Lk, Lu)

    # ---------------- Monte Carlo ----------------
    monte_carlo(int(P["MC_RUNS"]))
    if int(P["SENS_RUNS"]) > 0:
        sensitivity(int(P["SENS_RUNS"]))
    print("\nFigures written to", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", help="JSON with measured/assumed parameters (see params_example.json)")
    ap.add_argument("--outdir", help="folder for the figures (default: ./figures next to this file)")
    args = ap.parse_args()
    if args.outdir:
        OUT = os.path.abspath(args.outdir)
    os.makedirs(OUT, exist_ok=True)
    if args.params:
        load_params(args.params)
    main()
