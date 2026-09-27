"""
Slide-ready figures + key numbers for the IITG HVSR survey deck.

    python scripts/make_figures.py

Re-runs the pipeline (so outputs/ and data/processed/ are refreshed), then writes
outputs/figures/*.png and outputs/insights.json. Every figure is built from the
real phyphox recordings in data/raw -- no synthetic data anywhere.
"""
import os
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import welch, spectrogram, detrend

sys.path.insert(0, os.path.dirname(__file__))
import hvsr_pipeline as hp
from basemap import fetch_osm
from noise_models import NLNM, NHNM, peterson

FIG = "outputs/figures"
os.makedirs(FIG, exist_ok=True)

# ---- style: validated categorical palette (dataviz reference), fixed site order ----
SITES = ["old_sac", "kv_gate", "core_4", "lecture_hall_lake", "barak_umium"]  # collection order
LABEL = {"old_sac": "Old SAC", "kv_gate": "KV Gate", "core_4": "Core 4",
         "lecture_hall_lake": "Lecture Hall lake", "barak_umium": "Barak–Umiam"}
COLOR = dict(zip(SITES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
GOOD, CRIT, WARN = "#0ca30c", "#d03b3b", "#fab219"
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": ["Segoe UI", "Segoe UI Symbol", "DejaVu Sans"], "font.size": 11,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "axes.axisbelow": True, "lines.linewidth": 2, "lines.solid_capstyle": "round",
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def caption(fig, text, y=-0.02):
    fig.text(0.01, y, text, color=MUTED, fontsize=9, ha="left", va="top")


def db(asd):
    return 20 * np.log10(asd)


# =====================================================================================
print("running pipeline ...")
R = hp.run()
S = R["summary"].set_index("location_id").loc[SITES]
RES, WIN, PROV = R["results"], R["windows"], R["provenance"]
ACC = {a.location_id.iloc[0]: a for a in R["accel"]}
FS = {k: float(S.loc[k, "sample_rate_hz"]) for k in SITES}
FLOOR_DB = db(hp.PHONE_FLOOR_ASD)
insights = {}

# ---- 01 collection timeline --------------------------------------------------------
print("figures ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True, gridspec_kw={"width_ratios": [44, 7], "wspace": 0.04})
kind_name = {"accel": "Vibration", "gps": "GPS", "incl": "Tilt"}
for ax in axes:
    for i, k in enumerate(SITES):
        for _, r in PROV[PROV.location_id == k].iterrows():
            s, e = r.start_ist.tz_localize(None), r.end_ist.tz_localize(None)
            full = r.kind == "accel"
            ax.barh(i, (e - s).total_seconds() / 86400, left=matplotlib.dates.date2num(s), height=0.55 if full else 0.3,
                    color=COLOR[k], alpha=1 if full else 0.45, lw=0)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax.grid(axis="y", visible=False)
axes[0].set_xlim(matplotlib.dates.datestr2num("2026-09-26 19:55"), matplotlib.dates.datestr2num("2026-09-26 20:41"))
axes[1].set_xlim(matplotlib.dates.datestr2num("2026-09-26 22:43"), matplotlib.dates.datestr2num("2026-09-26 22:50"))
axes[1].spines["left"].set_visible(False)
axes[1].tick_params(left=False)
axes[0].set_yticks(range(len(SITES)), [LABEL[k] for k in SITES], color=INK)
axes[0].invert_yaxis()
axes[0].set_title("One evening, five sites: every site gets the same 3-recording protocol")
for i, k in enumerate(SITES):
    ax = axes[1] if k == "barak_umium" else axes[0]
    s = PROV[(PROV.location_id == k) & (PROV.kind == "accel")].start_ist.iloc[0].tz_localize(None)
    ax.text(matplotlib.dates.date2num(s), i - 0.4, f"{S.loc[k, 'duration_s'] / 60:.1f} min", fontsize=9, color=INK2)
axes[0].text(0.99, -0.22, "thick bar = vibration recording · thin bars = GPS fix, then tilt check  ·  26 Sep 2026, IST",
             transform=axes[0].transAxes, ha="right", fontsize=9, color=MUTED)
save(fig, "01_collection_timeline.png")

# ---- 02 site map (OSM basemap if reachable) -----------------------------------------
lat0, lon0 = S.lat.mean(), S.lon.mean()
kx = 111320 * math.cos(math.radians(lat0))


def xy(lat, lon):
    return (lon - lon0) * kx, (lat - lat0) * 110574


def osm_basemap(ax, lat_rng, lon_rng, z=16):
    got = fetch_osm(lat_rng, lon_rng, z)
    if got is None:
        return False
    img, (lo_left, lo_right, la_bot, la_top) = got
    (l, b), (r, t) = xy(la_bot, lo_left), xy(la_top, lo_right)
    ax.imshow(img, extent=(l, r, b, t), zorder=0, alpha=0.75)
    return True


fig, ax = plt.subplots(figsize=(8.5, 7))
pad = 0.004
has_map = osm_basemap(ax, (S.lat.min() - pad, S.lat.max() + pad), (S.lon.min() - pad, S.lon.max() + pad))
for k in SITES:
    x, y = xy(S.loc[k, "lat"], S.loc[k, "lon"])
    acc = S.loc[k, "gps_horizontal_accuracy_m"]
    ax.add_patch(plt.Circle((x, y), acc, color=COLOR[k], alpha=0.18, lw=0))
    ax.add_patch(plt.Circle((x, y), acc, fill=False, color=COLOR[k], lw=1))
    ax.scatter(x, y, s=90, color=COLOR[k], edgecolor=SURF, lw=2, zorder=5)
    note = f"±{acc:.0f} m" + (" (network fix, no satellite lock)" if acc > 50 else f" GNSS, {S.loc[k, 'gps_satellites']:.0f} sats")
    ax.annotate(f"{LABEL[k]}\n{note}", (x, y), xytext=(10, 8), textcoords="offset points", fontsize=10,
                color=INK, fontweight="semibold", bbox=dict(boxstyle="round,pad=0.25", fc=SURF, ec="none", alpha=0.85))
xs_, ys_ = zip(*[xy(S.loc[k, "lat"], S.loc[k, "lon"]) for k in SITES])
ax.set_xlim(min(xs_) - 250, max(xs_) + 350)
ax.set_ylim(min(ys_) - 250, max(ys_) + 250)
ax.set_aspect("equal")
ax.set_xlabel("east (m)")
ax.set_ylabel("north (m)")
ax.set_title("Survey sites, IIT Guwahati campus (circle = GPS accuracy)")
ax.plot([max(xs_) + 50, max(xs_) + 250], [min(ys_) - 200] * 2, color=INK, lw=3)
ax.text(max(xs_) + 150, min(ys_) - 185, "200 m", ha="center", fontsize=9)
if has_map:
    caption(fig, "Basemap © OpenStreetMap contributors.")
save(fig, "02_site_map.png")

# ---- 03 raw traces + QC --------------------------------------------------------------
fig, axes = plt.subplots(len(SITES), 1, figsize=(12, 8), sharex=True, sharey=True)
for ax, k in zip(axes, SITES):
    a = ACC[k]
    e = np.sqrt(sum((a[c] - a[c].median()) ** 2 for c in ("ax", "ay", "az")))
    rms1 = e.pow(2).rolling(int(FS[k]), center=True, min_periods=1).mean().pow(0.5)
    ax.plot(a.t, rms1, color=COLOR[k], lw=1.2)
    spans = []
    for _, w in WIN[(WIN.location_id == k) & ~WIN.kept].sort_values("t_start").iterrows():
        if spans and w.t_start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], w.t_end)
        else:
            spans.append([w.t_start, w.t_end])
    for t0, t1 in spans:
        ax.axvspan(t0, t1, color=CRIT, alpha=0.08, lw=0)
    kept = WIN[(WIN.location_id == k)].kept
    ax.text(1.0, 0.92, f"{LABEL[k]} · {kept.sum()}/{len(kept)} windows kept", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color=INK, fontweight="semibold")
    ax.axhline(hp.PHONE_FLOOR_ASD * np.sqrt(3 * FS[k] / 2), color=MUTED, lw=0.8)
    ax.set_yscale("log")
axes[0].set_ylim(3e-3, 3)
axes[0].set_title("Raw vibration, 1-s RMS: long quiet stretches, short bursts (footsteps, passers-by, handling); red = rejected")
axes[2].set_ylabel("|acceleration| RMS (m/s²)")
axes[-1].set_xlabel("seconds since recording start")
axes[-1].set_xlim(0, 210)
caption(fig, "Grey line = phone self-noise level expected for a 3-axis, 50 Hz-bandwidth reading. "
             "Windows rejected by STA/LTA trigger, RMS > 3× site median, or first/last 5 s "
             "(placing / picking up the phone). Barak–Umiam: sustained rise from ~92 s = machinery switching on.")
save(fig, "03_raw_traces_qc.png")

# ---- 04 spectra vs Peterson noise models -------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
T = 1 / np.logspace(np.log10(0.1), 1, 300)
for ax, comp in zip(axes, ("vertical (Z)", "horizontal (mean of X, Y)")):
    ax.fill_between(1 / T, peterson(NLNM, T), peterson(NHNM, T), color=MUTED, alpha=0.15, lw=0)
    ax.plot(1 / T, peterson(NHNM, T), color=MUTED, lw=1)
    ax.plot(1 / T, peterson(NLNM, T), color=MUTED, lw=1)
    ax.text(0.3, peterson(NHNM, np.array([1 / 0.3]))[0] + 4, "Peterson high-noise model (NHNM)", fontsize=9, color=INK2)
    ax.text(0.3, peterson(NLNM, np.array([1 / 0.3]))[0] + 4, "Peterson low-noise model (NLNM)", fontsize=9, color=INK2)
    ax.text(0.45, -128, "range where natural ground\nvibration usually sits", fontsize=9, color=MUTED, style="italic")
    for k in SITES:
        r = RES[k]
        P = r["psd_median"][2] if comp.startswith("v") else (r["psd_median"][0] + r["psd_median"][1]) / 2
        m = r["psd_freqs"] > 0
        ax.plot(r["psd_freqs"][m], 10 * np.log10(P[m]), color=COLOR[k], lw=1.4, label=LABEL[k])
    ax.axhline(FLOOR_DB, color=INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 45)
    ax.set_title(comp)
    ax.set_xlabel("Frequency (Hz)")
axes[0].set_ylim(-180, -30)
axes[0].set_ylabel("acceleration PSD (dB rel. 1 (m/s²)²/Hz)")
axes[1].legend(loc="lower right", fontsize=9, ncol=1)
axes[0].text(0.21, FLOOR_DB + 7, "phone self-noise ≈ 1 mm/s²/√Hz", fontsize=9, color=INK)
f_chk = np.array([0.5, 1, 2, 5, 9.9])
gap = FLOOR_DB - peterson(NHNM, 1 / f_chk)
insights["floor_gap_db_vs_nhnm_0p5_10hz"] = [round(float(gap.min()), 1), round(float(gap.max()), 1)]
insights["floor_gap_amplitude_x"] = [round(float(10 ** (gap.min() / 20))), round(float(10 ** (gap.max() / 20)))]
axes[0].annotate("", xy=(1, peterson(NHNM, np.array([1.0]))[0]), xytext=(1, FLOOR_DB),
                 arrowprops=dict(arrowstyle="<->", color=CRIT, lw=1.5))
axes[0].text(1.1, (FLOOR_DB + peterson(NHNM, np.array([1.0]))[0]) / 2,
             f"{gap[1]:.0f} dB\n≈ {10 ** (gap[1] / 20):.0f}× in amplitude", color=CRIT, fontsize=11, fontweight="semibold")
fig.suptitle("At the quiet sites, the phone mostly measures itself: spectra sit on its flat self-noise floor, "
             "far above typical natural ground vibration", x=0.01, ha="left", fontsize=13, fontweight="semibold")
caption(fig, "Median Welch PSD of accepted (quiet) 20 s windows. Only persistent source above the floor: Barak–Umiam machinery "
             "(16 Hz horizontal, 27–33 Hz vertical). Gap to NHNM across 0.5–10 Hz: "
             f"{gap.min():.0f}–{gap.max():.0f} dB ({10 ** (gap.min() / 20):.0f}–{10 ** (gap.max() / 20):.0f}× in amplitude).\n"
             "Peterson (1993) models describe vertical motion at seismic stations (0.1–10 s periods); busy urban sites can exceed NHNM above ~1 Hz.")
save(fig, "04_spectra_vs_noise_models.png")

# ---- 05 signal-above-floor heatmap -------------------------------------------------
rows, labels = [], []
for k in SITES:
    r = RES[k]
    for name, P in (("H", (r["psd_median"][0] + r["psd_median"][1]) / 2), ("V", r["psd_median"][2])):
        rows.append(10 * np.log10(P / hp.PHONE_FLOOR_ASD ** 2))
        labels.append(f"{LABEL[k]} · {name}")
f = RES[SITES[0]]["psd_freqs"]
fig, ax = plt.subplots(figsize=(12, 4.8))
im = ax.pcolormesh(f[1:], np.arange(len(rows)) + 0.5, np.clip(np.array(rows)[:, 1:], 0, 20), cmap=BLUES,
                   vmin=0, vmax=20, shading="nearest")
ax.set_xscale("log")
ax.set_xlim(0.3, 49)
ax.set_yticks(np.arange(len(rows)) + 0.5, labels, fontsize=9, color=INK)
ax.invert_yaxis()
ax.grid(False)
for i in range(2, len(rows), 2):
    ax.axhline(i, color=SURF, lw=2)
ax.axvspan(0.5, 15, ymin=0, ymax=1, fill=False, ec=INK2, lw=1)
ax.text(0.53, len(rows) - 0.1, "HVSR search band 0.5–15 Hz", fontsize=9, color=INK2, va="bottom")
cb = fig.colorbar(im, ax=ax, pad=0.01)
cb.set_label("dB above phone self-noise", color=INK2)
cb.outline.set_visible(False)
ax.set_xlabel("Frequency (Hz)")
ax.set_title("Where is there real signal in the quiet windows? Almost nowhere inside the HVSR band", pad=12)
caption(fig, "White = at/below the phone's self-noise (1 mm/s²/√Hz). HVSR needs BOTH H and V clearly above the floor "
             "across the band; no site has it. The only clear excess is Barak–Umiam machinery above 15 Hz.")
save(fig, "05_signal_above_floor_heatmap.png")

# ---- 06 spectrograms (Z) ---------------------------------------------------------------
fig, axes = plt.subplots(len(SITES), 1, figsize=(12, 9), sharex=True)
for ax, k in zip(axes, SITES):
    a = ACC[k]
    fq, tt, Sxx = spectrogram(detrend(a.az.values), fs=FS[k], nperseg=512, noverlap=384)
    im = ax.pcolormesh(tt, fq, np.clip(10 * np.log10(Sxx / hp.PHONE_FLOOR_ASD ** 2), 0, 25), cmap=BLUES,
                       vmin=0, vmax=25, shading="auto", rasterized=True)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 49)
    ax.grid(False)
    ax.text(0.995, 0.9, LABEL[k], transform=ax.transAxes, ha="right", va="top", fontweight="semibold",
            bbox=dict(fc=SURF, ec="none", alpha=0.8, pad=2))
axes[2].set_ylabel("Frequency (Hz), vertical axis")
axes[-1].set_xlabel("seconds since recording start")
axes[0].set_title("Vertical-axis spectrograms: short vertical stripes = transients (footsteps, passers-by); Barak–Umiam band from ~90 s = machinery")
cb = fig.colorbar(im, ax=axes, pad=0.01, shrink=0.6)
cb.set_label("dB above phone self-noise", color=INK2)
cb.outline.set_visible(False)
save(fig, "06_spectrograms_vertical.png")

# ---- 07 HVSR small multiples ----------------------------------------------------------
fig, axes = plt.subplots(1, len(SITES), figsize=(14, 3.8), sharey=True)
for ax, k in zip(axes, SITES):
    r = RES[k]
    ax.fill_between(r["freqs"], r["hv_lower"], r["hv_upper"], color=COLOR[k], alpha=0.2, lw=0)
    ax.plot(r["freqs"], r["hv_mean"], color=COLOR[k])
    ax.axhline(2, color=CRIT, lw=1)
    ax.axhline(1, color=AXIS, lw=0.8)
    ax.scatter([r["f0_hz"]], [r["a0"]], s=40, color=INK, zorder=5)
    ax.set(xscale="log", yscale="log", xlim=(0.3, 20), ylim=(0.3, 4))
    ax.set_yticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
    ax.set_xticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
    ax.set_title(LABEL[k], fontsize=11)
    ax.text(0.04, 0.05, f"max {r['a0']:.2f} @ {r['f0_hz']:.1f} Hz\nSESAME clarity {S.loc[k, 'sesame_clarity_passed']}/6",
            transform=ax.transAxes, fontsize=9, color=INK2)
    ax.set_xlabel("Hz")
axes[0].set_ylabel("H/V ratio")
axes[0].text(0.35, 2.15, "clear-peak threshold A0 > 2", fontsize=8, color=CRIT)
fig.suptitle("Corrected HVSR: flat curves near H/V = 1 at every site, with no clear resonance peak",
             x=0.01, ha="left", fontsize=13, fontweight="semibold", y=1.04)
caption(fig, "Konno-Ohmachi (b=40) smoothed, 20 s windows, log-space (geometric) stacking; band = ±1σ across windows. "
             "Flat H/V ≈ 1 is what pure sensor noise with equal per-axis levels produces.", y=-0.06)
save(fig, "07_hvsr_all_sites.png")

# ---- 08 method correction: old vs corrected on Barak–Umiam ------------------------------
k = "barak_umium"
a, fs = ACC[k], FS[k]
win = int(hp.WINDOW_SEC * fs)
old = []
for s in hp.iter_windows(len(a), win, win // 2):
    P = [welch(detrend(a[c].values[s:s + win]), fs=fs, nperseg=win)[1] for c in ("ax", "ay", "az")]
    old.append(np.sqrt((P[0] + P[1]) / (2 * P[2])))
f_old = welch(np.zeros(win), fs=fs, nperseg=win)[0]
lm = np.log(np.clip(old, 1e-12, None)).mean(0)
band = (f_old >= 0.3) & (f_old <= 15)
i_old = np.argmax(np.where(band, lm, -np.inf))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
axes[0].plot(f_old[1:], np.exp(lm[1:]), color=MUTED, lw=1)
axes[0].scatter(f_old[i_old], np.exp(lm[i_old]), color=CRIT, s=50, zorder=5)
axes[0].annotate(f"'f0 = {f_old[i_old]:.1f} Hz, A0 = {np.exp(lm[i_old]):.2f}'\n(the number in the old README)",
                 (f_old[i_old], np.exp(lm[i_old])), xytext=(-170, 25), textcoords="offset points", color=CRIT, fontsize=10,
                 arrowprops=dict(arrowstyle="->", color=CRIT))
axes[0].set_title("Before: one raw periodogram per window, no smoothing")
r = RES[k]
axes[1].fill_between(r["freqs"], r["hv_lower"], r["hv_upper"], color=COLOR[k], alpha=0.2, lw=0)
axes[1].plot(r["freqs"], r["hv_mean"], color=COLOR[k])
axes[1].set_title("After: Konno-Ohmachi smoothing + noise-floor gate")
axes[1].text(0.35, 2.3, f"max A = {r['a0']:.2f}: no clear peak · verdict: unresolved (sensor floor)", color=INK, fontsize=10)
for ax in axes:
    ax.axhline(2, color=CRIT, lw=0.8)
    ax.set(xscale="log", yscale="log", xlim=(0.3, 20), ylim=(0.3, 5), xlabel="Frequency (Hz)")
    ax.set_yticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
    ax.set_xticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
axes[0].set_ylabel("H/V ratio")
fig.suptitle("We caught our own false positive: the old '5.2 Hz' resonance was a spike in unsmoothed noise",
             x=0.01, ha="left", fontsize=13, fontweight="semibold", y=1.03)
caption(fig, "Same Barak–Umiam recording, same 20 s windows. A single-segment periodogram has only 2 degrees of freedom "
             "per bin, so the H/V of pure noise throws up tall, random spikes.")
insights["old_method_barak"] = {"f0_hz": round(float(f_old[i_old]), 2), "a0": round(float(np.exp(lm[i_old])), 2)}
save(fig, "08_method_correction_barak.png")

# ---- 09 per-window f0 wander -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 3.8))
for i, k in enumerate(SITES):
    w = WIN[(WIN.location_id == k) & WIN.kept]
    jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(w))
    ax.scatter(w.f0_window_hz, i + jitter, s=46, color=COLOR[k], edgecolor=SURF, lw=1.5, zorder=3)
    ax.text(16.5, i, f"σf = {S.loc[k, 'sigma_f_hz']:.2f} Hz", va="center", fontsize=9, color=INK2)
ax.set_xscale("log")
ax.set_xlim(0.45, 30)
ax.set_yticks(range(len(SITES)), [LABEL[k] for k in SITES], color=INK)
ax.invert_yaxis()
ax.set_xlabel("frequency of the H/V maximum in each accepted 20 s window (Hz)")
ax.set_title("A real resonance repeats window after window. Here the 'peak' lands somewhere new each time")
eps = {k: next(e for lim, e in [(0.2, .25), (0.5, .20), (1.0, .15), (2.0, .10), (np.inf, .05)] if S.loc[k, "f0_hz"] < lim)
       for k in SITES}
miss = [S.loc[k, "sigma_f_hz"] / (eps[k] * S.loc[k, "f0_hz"]) for k in SITES]
insights["sesame_c5_miss_factor"] = {k: round(float(m), 1) for k, m in zip(SITES, miss)}
caption(fig, f"SESAME criterion C5 needs σf below 5–15% of f0; every site misses it by {min(miss):.0f}–{max(miss):.0f}×. "
             "Points piling at 0.5 Hz sit on the search-band edge: another sign there is no real peak.", y=-0.04)
save(fig, "09_window_f0_scatter.png")

# ---- 10 SESAME + QC scorecard -------------------------------------------------------------
crit_cols = [c for c in S.columns if c[:2] in ("R1", "R2", "R3") or c[:2] in ("C1", "C2", "C3", "C4", "C5", "C6")]
short = {"R1_f0_gt_10_over_lw": "R1\nf0>10/lw", "R2_nc_gt_200": "R2\ncycles>200", "R3_sigmaA_lt_2_near_f0": "R3\nσA<2",
         "C1_trough_below_f0": "C1\ntrough <f0", "C2_trough_above_f0": "C2\ntrough >f0", "C3_A0_gt_2": "C3\nA0>2",
         "C4_peak_stable_5pct": "C4\nstable ±5%", "C5_sigma_f_lt_eps": "C5\nσf<ε", "C6_sigmaA_lt_theta": "C6\nσA<θ"}
qc = [("record\n≥10 min", lambda k: S.loc[k, "duration_s"] >= 600),
      ("phone\nlevel <5°", lambda k: S.loc[k, "mean_tilt_deg"] < 5),
      ("GPS\n<10 m", lambda k: S.loc[k, "gps_horizontal_accuracy_m"] < 10),
      ("signal above\nphone floor", lambda k: S.loc[k, "band_frac_above_floor"] >= hp.MIN_FRAC_ABOVE_FLOOR)]
cols = [("Collection QC", n, fn) for n, fn in qc] + [("SESAME reliability" if c[0] == "R" else "SESAME peak clarity",
                                                    short[c], (lambda c: lambda k: bool(S.loc[k, c]))(c)) for c in crit_cols]
fig, ax = plt.subplots(figsize=(13, 3.9))
ax.set_xlim(-0.5, len(cols) - 0.5)
ax.set_ylim(len(SITES) - 0.5, -2.0)
ax.axis("off")
for j, (grp, name, fn) in enumerate(cols):
    ax.text(j, -0.5, name, ha="center", va="bottom", fontsize=9, color=INK2, linespacing=1.1)
    for i, k in enumerate(SITES):
        ok = fn(k)
        ax.add_patch(plt.Rectangle((j - 0.46, i - 0.4), 0.92, 0.8, color=(GOOD if ok else CRIT), alpha=0.12, lw=0))
        ax.text(j, i, "✓" if ok else "✗", ha="center", va="center", fontsize=14, color=(GOOD if ok else CRIT), fontweight="bold")
for grp, j0, j1 in (("Collection QC", 0, 3), ("SESAME reliability", 4, 6), ("SESAME peak clarity (need ≥5/6)", 7, 12)):
    ax.plot([j0 - 0.4, j1 + 0.4], [-1.5, -1.5], color=AXIS, lw=1)
    ax.text((j0 + j1) / 2, -1.6, grp, ha="center", va="bottom", fontsize=10, color=INK, fontweight="semibold")
for i, k in enumerate(SITES):
    ax.text(-0.6, i, LABEL[k], ha="right", va="center", fontsize=10, color=INK, fontweight="semibold")
    ax.text(len(cols) - 0.35, i, "unresolved" if S.loc[k, "confidence"] == "unresolved_sensor_floor" else S.loc[k, "confidence"],
            ha="left", va="center", fontsize=10, color=CRIT)
ax.text(len(cols) - 0.35, -0.5, "verdict", ha="left", va="bottom", fontsize=9, color=INK2)
fig.suptitle("Scorecard: protocol mostly followed, but no site yields a SESAME-grade resonance peak",
             x=0.01, ha="left", fontsize=13, fontweight="semibold")
save(fig, "10_quality_scorecard.png")

# ---- 11 vibration-criteria screening (what a phone CAN do) --------------------------------
cf = 1000 * 2 ** (np.arange(-30, -14) / 3)  # 1/3-octave centres ~1..32 Hz
cf = cf[(cf > 0.95) & (cf < 41)]


def third_octave_velocity_um(fr, P):
    out = []
    for c in cf:
        m = (fr >= c * 2 ** (-1 / 6)) & (fr < c * 2 ** (1 / 6))
        out.append(np.sqrt(np.trapezoid(P[m] / (2 * np.pi * fr[m]) ** 2, fr[m])) * 1e6 if m.sum() > 1 else np.nan)
    return np.array(out)


fig, ax = plt.subplots(figsize=(11, 5.2))
crit_lines = [("ISO office", 400), ("ISO residential (day)", 200), ("ISO operating theatre", 100),
              ("VC-A (optical microscopes)", 50), ("VC-B", 25), ("VC-C", 12.5)]
for name, v in crit_lines:
    ax.axhline(v, color=AXIS, lw=0.9)
    ax.text(42, v, name, va="center", fontsize=9, color=INK2)
fr0 = RES[SITES[0]]["psd_freqs"]
vfloor = third_octave_velocity_um(fr0, np.full_like(fr0, hp.PHONE_FLOOR_ASD ** 2))
ax.fill_between(cf, 1, vfloor, color=MUTED, alpha=0.18, lw=0)
ax.plot(cf, vfloor, color=MUTED, lw=1.2)
ax.text(1.05, 6.3, "grey = phone self-noise: vibration in here is invisible to the phone", fontsize=9, color=INK2, style="italic")
vc = {}
for k in SITES:
    r = RES[k]
    v = np.nanmax([third_octave_velocity_um(r["psd_freqs"], r["psd_median"][c]) for c in range(3)], axis=0)
    vc[k] = v
    ax.plot(cf, v, color=COLOR[k], lw=1.6, marker="o", ms=4, label=LABEL[k])
ax.set(xscale="log", yscale="log", xlim=(0.9, 40), ylim=(5, 1000), xlabel="1/3-octave band centre (Hz)",
       ylabel="RMS velocity, worst axis (µm/s)")
ax.set_xticks([1, 2, 4, 8, 16, 31.5], ["1", "2", "4", "8", "16", "31.5"])
ax.legend(loc="upper right", fontsize=9, ncol=3, bbox_to_anchor=(1.0, 1.0))
ax.set_title("What the phone CAN say: an upper bound. In quiet periods all 5 sites are below the ISO operating-theatre limit")
caption(fig, "Generic criteria (ISO 2631-2 multiples; Gordon/Amick VC curves); screening only, not certification. "
             "Median of accepted quiet windows, so transients are excluded.\nA site sitting ON the grey line means its true level is at or below it. "
             "The phone cannot check VC-A below ~3 Hz, or VC-B and stricter at all.")
save(fig, "11_vibration_criteria_screening.png")
insights["vc_max_um_s"] = {k: round(float(np.nanmax(v)), 1) for k, v in vc.items()}
insights["phone_floor_um_s_at_1_4_16hz"] = [round(float(np.interp(x, cf, vfloor)), 1) for x in (1, 4, 16)]

# ---- 12 AI check: can a classifier tell sites apart? ---------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score

edges = np.logspace(np.log10(0.8), np.log10(45), 15)  # every band wider than the 0.2 Hz PSD bin
X, y, g = [], [], []
for si, k in enumerate(SITES):
    a, fs = ACC[k], FS[k]
    keep = a.sta_lta_quiet.values
    n = int(10 * fs)
    chunks = [s for s in range(0, len(a) - n, n) if keep[s:s + n].mean() > 0.9]
    rms = np.array([np.sqrt(sum(np.var(a[c].values[s:s + n]) for c in ("ax", "ay", "az"))) for s in chunks])
    chunks = [s for s, rr in zip(chunks, rms) if rr <= hp.MAX_RMS_RATIO * np.median(rms)]
    for j, s in enumerate(chunks):
        feats = []
        for c in ("ax", "ay", "az"):
            fr, P = welch(detrend(a[c].values[s:s + n]), fs=fs, nperseg=512)
            feats += [np.log10(P[(fr >= lo) & (fr < hi)].mean()) for lo, hi in zip(edges[:-1], edges[1:])]
        X.append(feats)
        y.append(si)
        g.append(int(4 * j / len(chunks)))  # 4 contiguous time blocks per site -> no temporal leakage
X, y, g = np.array(X), np.array(y), np.array(g)
centres = np.sqrt(edges[:-1] * edges[1:])
low = np.tile(centres < 8, 3)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
cv = GroupKFold(n_splits=4)
pred_all = cross_val_predict(model, X, y, groups=g, cv=cv)
pred_low = cross_val_predict(model, X[:, low], y, groups=g, cv=cv)
acc_all, acc_low = accuracy_score(y, pred_all), accuracy_score(y, pred_low)
insights["classifier"] = {"n_windows": int(len(y)), "chance": round(1 / len(SITES), 2),
                          "acc_all_bands": round(acc_all, 2), "acc_0p8_8hz_only": round(acc_low, 2),
                          "per_site_recall_all": {SITES[i]: round(float(np.mean(pred_all[y == i] == i)), 2) for i in range(len(SITES))},
                          "per_site_recall_low": {SITES[i]: round(float(np.mean(pred_low[y == i] == i)), 2) for i in range(len(SITES))}}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, pred, ttl in ((axes[0], pred_all, f"All bands 0.8–45 Hz: accuracy {acc_all:.0%}"),
                      (axes[1], pred_low, f"Only 0.8–8 Hz (the HVSR band): accuracy {acc_low:.0%}")):
    cm = confusion_matrix(y, pred, labels=range(len(SITES)), normalize="true")
    ax.imshow(cm, cmap=BLUES, vmin=0, vmax=1)
    for i in range(len(SITES)):
        for j in range(len(SITES)):
            ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center", fontsize=9, color=SURF if cm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(SITES)), [LABEL[k] for k in SITES], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(SITES)), [LABEL[k] for k in SITES], fontsize=9)
    ax.grid(False)
    ax.set_title(ttl, fontsize=11)
    ax.set_xlabel("predicted site")
axes[0].set_ylabel("true site")
fig.suptitle(f"AI as a lie-detector for our own data: can a model tell the sites apart? (chance = {1 / len(SITES):.0%})",
             x=0.01, ha="left", fontsize=13, fontweight="semibold")
caption(fig, f"Logistic regression on per-axis log-PSD in 14 bands; {len(y)} quiet 10 s windows; 4-fold CV split by contiguous time "
             "blocks (no temporal leakage).\nIf ground resonance differed between sites, the 0.8–8 Hz band should still identify them. "
             "Only Barak–Umiam (machinery above 15 Hz) is recognisable.", y=-0.1)
save(fig, "12_site_fingerprint_classifier.png")

# ---- key numbers -----------------------------------------------------------------------------
insights.update({
    "n_sites": len(SITES),
    "n_recordings": int(len(PROV)),
    "n_accel_samples": int(S.n_samples.sum()),
    "n_gps_rows": int(sum(len(x) for x in R["gps"])),
    "n_inclination_rows": int(sum(len(x) for x in R["incl"])),
    "accel_minutes_total": round(float(S.duration_s.sum() / 60), 1),
    "collection_window_ist": [str(PROV.start_ist.min()), str(PROV.end_ist.max())],
    "windows_kept": f"{int(S.n_windows_used.sum())}/{int(S.n_windows_total.sum())}",
    "sites_resolved": int((S.confidence != "unresolved_sensor_floor").sum()),
    "measured_floor_mm_s2_rtHz": {k: [round(float(np.sqrt(np.median(RES[k]["psd_median"][c][(RES[k]["psd_freqs"] > 0.5) &
                                        (RES[k]["psd_freqs"] < 8)])) * 1e3), 3) for c in range(3)] for k in SITES},
})
with open("outputs/insights.json", "w", encoding="utf-8") as fh:
    json.dump(insights, fh, indent=2, default=str)
print(json.dumps(insights, indent=2, default=str))
