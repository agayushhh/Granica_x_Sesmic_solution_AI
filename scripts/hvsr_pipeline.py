"""
HVSR (Horizontal-to-Vertical Spectral Ratio) pipeline for the IITG ambient
ground-vibration citizen survey.

Ingests phyphox zip exports (one location = up to 3 zips: Acceleration without
g, Location/GPS, Inclination), rejects transient-contaminated windows with an
STA/LTA filter, computes Konno-Ohmachi-smoothed H/V per window, stacks in log
space, runs the SESAME (2004) peak criteria, and -- before any f0 is reported --
checks that the signal actually sits above the phone accelerometer's own
self-noise floor.

Writes: outputs/summary.{csv,parquet}, outputs/hvsr_curves.parquet,
outputs/windows.parquet, outputs/<location_id>_hvsr.png, and tidy
data/processed/{accel,gps,inclination}.parquet + recordings.csv (provenance).

See README.md for full context, data schema, and interpretation guidance.
"""
import os
import re
import zipfile
import hashlib
import numpy as np
import pandas as pd
from scipy.signal import detrend, welch
from scipy.signal.windows import tukey
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# phyphox exports get renamed by hand in the field, so match on keywords, not
# exact suffixes (seen so far: "accelaration", "ocation (GPS)", double spaces).
KINDS = {
    "accel": re.compile(r"acc[a-z]*ration", re.I),
    "gps": re.compile(r"l?ocation|gps", re.I),
    "incl": re.compile(r"inclination", re.I),
}

# ---- Tunable analysis parameters (see README "Analysis method" section) ----
WINDOW_SEC = 20.0          # HVSR window length
OVERLAP = 0.5              # fractional overlap between windows
TRIM_SEC = 5.0             # always drop first/last seconds (phone placement / pickup)
BAND_HZ = (0.5, 15.0)      # f0 search band; 0.5 Hz = SESAME 10/lw limit for 20 s windows
KO_B = 40                  # Konno-Ohmachi smoothing bandwidth (SESAME/Geopsy default)
FREQ_GRID = np.logspace(np.log10(0.2), np.log10(25), 160)
STA_SEC = 1.0              # STA/LTA short-term window
LTA_SEC = 10.0             # STA/LTA long-term window
STA_LTA_THRESH = 2.5       # ratio above which a sample is "transient"
MAX_CONTAM_FRAC = 0.10     # reject an HVSR window if >10% of it is transient
MAX_RMS_RATIO = 3.0        # ...or if its RMS is >3x the site's median window RMS. STA/LTA
                           # misses long (>LTA) bursts because the LTA rises with them.

# Phone self-noise (amplitude spectral density). Note: estimated from THIS
# dataset -- the flat, axis-identical level at the 4 quiet sites (0.93-1.09
# mm/s^2/rtHz on a vivo V2545 / TDK icm42607). Replace with a proper self-noise
# test (phone on foam in a quiet room) and re-measure per phone model.
PHONE_FLOOR_ASD = 1.0e-3   # m/s^2/sqrt(Hz)
FLOOR_MARGIN = 2.0         # signal must beat floor PSD by 2x (3 dB) to count
MIN_FRAC_ABOVE_FLOOR = 0.5 # share of BAND_HZ where ALL 3 axes must clear the floor
MAX_TILT_DEG = 5.0


def site_id(prefix):
    return re.sub(r"[^a-z0-9]+", "_", prefix.lower()).strip("_")


def find_locations(raw_dir):
    """Group zip files in raw_dir by location_id = normalised text before the kind keyword."""
    locs = {}
    for f in sorted(os.listdir(raw_dir)):
        if not f.lower().endswith(".zip"):
            continue
        for kind, rx in KINDS.items():
            m = rx.search(f)
            if m:
                locs.setdefault(site_id(f[:m.start()]), {})[kind] = os.path.join(raw_dir, f)
                break
    return locs


def read_csv_from_zip(zip_path, inner_name):
    with zipfile.ZipFile(zip_path) as z:
        with z.open(inner_name) as fh:
            return pd.read_csv(fh)


def recording_meta(zip_path):
    """Provenance for one phyphox export: wall-clock window + device + checksum."""
    t = read_csv_from_zip(zip_path, "meta/time.csv")
    dev = read_csv_from_zip(zip_path, "meta/device.csv").set_index("property")["value"]
    with open(zip_path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    start = pd.to_datetime(t["system time"].iloc[0], unit="s", utc=True).tz_convert("Asia/Kolkata")
    end = pd.to_datetime(t["system time"].iloc[-1], unit="s", utc=True).tz_convert("Asia/Kolkata")
    return {"source_file": os.path.basename(zip_path), "sha256": sha,
            "start_ist": start, "end_ist": end,
            "device": f"{dev.get('deviceBrand')} {dev.get('deviceModel')}",
            "android_release": dev.get("deviceRelease"),
            "phyphox_version": dev.get("version"),
            "accelerometer": f"{dev.get('accelerometer Vendor')} {dev.get('accelerometer Name')}"}


def load_accel(zip_path):
    df = read_csv_from_zip(zip_path, "Raw Data.csv")
    df.columns = ["t", "ax", "ay", "az", "a_abs"]
    df = df.drop_duplicates(subset="t").sort_values("t").reset_index(drop=True)
    fs = 1.0 / np.median(np.diff(df["t"].values))
    return df, fs


def load_gps(zip_path):
    if zip_path is None:
        return None
    df = read_csv_from_zip(zip_path, "Raw Data.csv")
    df.columns = ["t", "lat", "lon", "alt", "alt_wgs84", "speed",
                  "direction", "distance", "h_acc", "v_acc", "satellites"]
    return df


def load_incl(zip_path):
    if zip_path is None:
        return None
    try:
        df = read_csv_from_zip(zip_path, "Flat.csv")
    except KeyError:
        return None
    df.columns = ["t", "tilt_updown", "tilt_leftright"]
    return df.drop_duplicates(subset="t")


def sta_lta_keep_mask(df, fs):
    """True where the sample sits in a 'quiet' (non-transient) stretch.
    Energy uses de-meaned axes: the fused linear-acceleration channel carries a
    small DC bias (up to 0.03 m/s^2) that would otherwise mute the trigger."""
    e = sum((df[c] - df[c].median()) ** 2 for c in ("ax", "ay", "az"))
    sta_n = max(1, int(STA_SEC * fs))
    lta_n = max(sta_n + 1, int(LTA_SEC * fs))
    sta = e.rolling(sta_n, min_periods=1, center=True).mean()
    lta = e.rolling(lta_n, min_periods=1, center=True).mean()
    keep = (sta / lta.clip(lower=1e-12)).values < STA_LTA_THRESH
    t = df["t"].values
    keep &= (t > t[0] + TRIM_SEC) & (t < t[-1] - TRIM_SEC)
    return keep


def konno_ohmachi_matrix(freqs, fc, b=KO_B):
    """Rows = smoothing weights for each centre frequency in fc (Konno & Ohmachi, 1998)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        x = b * np.log10(freqs[None, :] / fc[:, None])
        w = (np.sin(x) / x) ** 4
    w[x == 0] = 1.0
    w[:, freqs <= 0] = 0.0
    w[np.abs(x) > 3 * np.pi] = 0.0  # beyond the 3rd side-lobe: negligible
    return w / w.sum(axis=1, keepdims=True)


def iter_windows(n, win_n, step):
    return range(0, max(1, n - win_n + 1), step)


def compute_hvsr(df, fs):
    keep = sta_lta_keep_mask(df, fs)
    win_n = min(int(WINDOW_SEC * fs), len(df))
    step = max(1, int(win_n * (1 - OVERLAP)))
    freqs = np.fft.rfftfreq(win_n, 1 / fs)
    W = konno_ohmachi_matrix(freqs, FREQ_GRID)
    taper = tukey(win_n, 0.1)

    starts = list(iter_windows(len(df), win_n, step))
    rms = np.array([np.sqrt(sum(np.var(df[c].values[s:s + win_n]) for c in ("ax", "ay", "az")))
                    for s in starts])
    curves, psds, win_rows = [], [], []
    for i, start in enumerate(starts):
        seg = slice(start, start + win_n)
        contam = 1 - keep[seg].mean()
        loud = rms[i] > MAX_RMS_RATIO * np.median(rms)
        ok = contam <= MAX_CONTAM_FRAC and not loud
        win_rows.append({"window": i, "t_start": df["t"].values[start],
                         "t_end": df["t"].values[start + win_n - 1],
                         "rms_ms2": rms[i], "contam_frac": contam,
                         "reject_reason": "sta_lta" if contam > MAX_CONTAM_FRAC else ("rms" if loud else ""),
                         "kept": ok})
        if not ok:
            continue
        sig = {c: detrend(df[c].values[seg]) for c in ("ax", "ay", "az")}
        amp = {c: W @ np.abs(np.fft.rfft(v * taper)) for c, v in sig.items()}
        h = np.sqrt((amp["ax"] ** 2 + amp["ay"] ** 2) / 2)  # quadratic-mean horizontal
        curves.append(h / np.maximum(amp["az"], 1e-20))
        f_w, P = None, []
        for c in ("ax", "ay", "az"):
            f_w, p = welch(sig[c], fs=fs, nperseg=min(1024, win_n))
            P.append(p)
        psds.append(P)

    windows = pd.DataFrame(win_rows)
    if not curves:
        return None, windows

    hv = np.clip(np.array(curves), 1e-12, None)
    # Stack in LOG space (geometric mean): H/V is ~log-normal across windows, so
    # an arithmetic mean lets a few loud windows dominate. Band is multiplicative.
    log_hv = np.log(hv)
    log_mean, log_std = log_hv.mean(axis=0), log_hv.std(axis=0)
    band = (FREQ_GRID >= BAND_HZ[0]) & (FREQ_GRID <= BAND_HZ[1])
    idx = int(np.argmax(np.where(band, log_mean, -np.inf)))

    # per-window peak frequency -> SESAME sigma_f
    win_f0 = FREQ_GRID[np.argmax(np.where(band, log_hv, -np.inf), axis=1)]
    windows.loc[windows["kept"], "f0_window_hz"] = win_f0
    windows.loc[windows["kept"], "a0_window"] = hv[np.arange(len(hv)), np.argmax(np.where(band, log_hv, -np.inf), axis=1)]

    psd_med = np.median(np.array(psds), axis=0)  # (3 axes, nfreq), median across windows
    return {
        "freqs": FREQ_GRID, "hv_mean": np.exp(log_mean), "log_std": log_std,
        "hv_upper": np.exp(log_mean + log_std), "hv_lower": np.exp(log_mean - log_std),
        "f0_hz": float(FREQ_GRID[idx]), "a0": float(np.exp(log_mean[idx])),
        "log_std_at_f0": float(log_std[idx]), "sigma_f_hz": float(np.std(win_f0)),
        "n_windows_used": len(curves), "n_windows_total": len(windows),
        "psd_freqs": f_w, "psd_median": psd_med,
    }, windows


def sesame_criteria(r, window_sec=WINDOW_SEC):
    """SESAME (2004) HVSR guideline checks. Returns dict of booleans.
    3 reliability criteria (curve is statistically usable) + 6 clarity criteria
    (peak is clear); SESAME calls a peak clear when >=5 of 6 hold."""
    f, A, sA = r["freqs"], r["hv_mean"], np.exp(r["log_std"])
    f0, A0 = r["f0_hz"], r["a0"]
    eps, theta = next((e, t) for lim, e, t in [(0.2, .25, 3.0), (0.5, .20, 2.5), (1.0, .15, 2.0),
                                              (2.0, .10, 1.78), (np.inf, .05, 1.58)] if f0 < lim)
    near = (f >= 0.5 * f0) & (f <= 2 * f0)
    lo, hi = (f >= f0 / 4) & (f <= f0), (f >= f0) & (f <= 4 * f0)
    f_up, f_dn = f[np.argmax(A * sA)], f[np.argmax(A / sA)]
    return {
        "R1_f0_gt_10_over_lw": f0 > 10 / window_sec,
        "R2_nc_gt_200": window_sec * r["n_windows_used"] * f0 > 200,
        "R3_sigmaA_lt_2_near_f0": bool(np.all(sA[near] < (2 if f0 > 0.5 else 3))),
        "C1_trough_below_f0": bool(np.any(A[lo] < A0 / 2)),
        "C2_trough_above_f0": bool(np.any(A[hi] < A0 / 2)),
        "C3_A0_gt_2": A0 > 2,
        "C4_peak_stable_5pct": abs(f_up - f0) < 0.05 * f0 and abs(f_dn - f0) < 0.05 * f0,
        "C5_sigma_f_lt_eps": r["sigma_f_hz"] < eps * f0,
        "C6_sigmaA_lt_theta": float(sA[np.argmin(abs(f - f0))]) < theta,
    }


def floor_check(r):
    """Is there ground signal above the phone's self-noise? H/V of pure sensor
    noise is just the ratio of per-axis noise levels (~1): it says nothing about
    the ground. Returns (fraction of band where all 3 axes clear the floor,
    min-axis SNR in dB at f0)."""
    f, P = r["psd_freqs"], r["psd_median"]
    snr = P.min(axis=0) / PHONE_FLOOR_ASD ** 2
    band = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1])
    frac = float(np.mean(snr[band] > FLOOR_MARGIN))
    snr_f0 = float(10 * np.log10(np.interp(r["f0_hz"], f, snr)))
    return frac, snr_f0


def classify(r, crit, frac_above, tilt):
    """Verdict ladder. The floor gate comes first: without it, H/V of sensor
    noise still yields a 'peak' (the old pipeline reported f0=5.2 Hz this way)."""
    if r is None:
        return "no_data"
    if frac_above < MIN_FRAC_ABOVE_FLOOR:
        return "unresolved_sensor_floor"
    reliable = all(v for k, v in crit.items() if k.startswith("R"))
    clear = sum(v for k, v in crit.items() if k.startswith("C"))
    tilt_ok = np.isnan(tilt) or tilt < MAX_TILT_DEG
    if reliable and clear >= 5 and tilt_ok:
        return "high"
    if reliable and clear >= 3:
        return "medium"
    return "low"


def process_location(loc_id, paths):
    accel_df, fs = load_accel(paths["accel"])
    gps_df = load_gps(paths.get("gps"))
    incl_df = load_incl(paths.get("incl"))
    result, windows = compute_hvsr(accel_df, fs)

    lat = lon = h_acc = sats = np.nan
    if gps_df is not None and len(gps_df):
        best = gps_df.loc[gps_df["h_acc"].idxmin()]
        lat, lon, h_acc, sats = (float(best[c]) for c in ("lat", "lon", "h_acc", "satellites"))
    mean_tilt = np.nan
    if incl_df is not None and len(incl_df):
        mean_tilt = float(np.hypot(incl_df["tilt_updown"], incl_df["tilt_leftright"]).mean())

    crit = sesame_criteria(result) if result else {}
    frac_above, snr_f0 = floor_check(result) if result else (np.nan, np.nan)
    row = {
        "location_id": loc_id,
        "start_ist": recording_meta(paths["accel"])["start_ist"].strftime("%Y-%m-%d %H:%M"),
        "sample_rate_hz": round(fs, 2),
        "duration_s": round(float(accel_df["t"].iloc[-1] - accel_df["t"].iloc[0]), 1),
        "n_samples": len(accel_df),
        "f0_hz": result["f0_hz"] if result else np.nan,
        "a0": result["a0"] if result else np.nan,
        "log_std_at_f0": result["log_std_at_f0"] if result else np.nan,
        "sigma_f_hz": result["sigma_f_hz"] if result else np.nan,
        "n_windows_used": result["n_windows_used"] if result else 0,
        "n_windows_total": result["n_windows_total"] if result else len(windows),
        "sesame_reliability_passed": sum(v for k, v in crit.items() if k.startswith("R")),
        "sesame_clarity_passed": sum(v for k, v in crit.items() if k.startswith("C")),
        "band_frac_above_floor": frac_above,
        "snr_db_at_f0": snr_f0,
        "confidence": classify(result, crit, frac_above, mean_tilt),
        "lat": lat, "lon": lon, "gps_horizontal_accuracy_m": h_acc, "gps_satellites": sats,
        "mean_tilt_deg": round(mean_tilt, 2) if not np.isnan(mean_tilt) else np.nan,
        "has_gps": gps_df is not None, "has_inclination": incl_df is not None,
        **crit,
    }
    return row, result, windows, accel_df.assign(sta_lta_quiet=sta_lta_keep_mask(accel_df, fs)), gps_df, incl_df


def plot_site(loc_id, r, conf, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(r["freqs"], r["hv_lower"], r["hv_upper"], color="#2a78d6", alpha=0.15,
                    lw=0, label="±1σ (log-space) across windows")
    ax.plot(r["freqs"], r["hv_mean"], color="#2a78d6", lw=2, label="geometric-mean H/V")
    ax.axhline(2, color="#898781", lw=1, label="SESAME clarity: A0 > 2")
    ax.axvline(r["f0_hz"], color="#d03b3b", lw=1,
               label=f"max in band: {r['f0_hz']:.2f} Hz (A={r['a0']:.2f})")
    ax.set(xscale="log", yscale="log", xlim=(0.2, 25), ylim=(0.2, 5),
           xlabel="Frequency (Hz)", ylabel="H/V ratio",
           title=f"{loc_id} — HVSR  [{conf}]  {r['n_windows_used']}/{r['n_windows_total']} windows kept")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{loc_id}_hvsr.png"), dpi=150)
    plt.close(fig)


def run(raw_dir="data/raw", out_dir="outputs", processed_dir="data/processed",
        manifest="sites_manifest.csv", write=True):
    """Process every discovered site. Returns dict of in-memory results (used by make_figures.py)."""
    locations = find_locations(raw_dir)
    rows, results, win_all, acc_all, gps_all, incl_all, prov = [], {}, [], [], [], [], []
    for loc_id, paths in sorted(locations.items()):
        for kind, p in paths.items():
            prov.append({"location_id": loc_id, "kind": kind, **recording_meta(p)})
        if "accel" not in paths:
            print(f"skip {loc_id}: no acceleration zip found")
            continue
        print(f"processing {loc_id} ({', '.join(sorted(paths))}) ...")
        row, r, w, acc, gps, incl = process_location(loc_id, paths)
        rows.append(row)
        results[loc_id] = r
        win_all.append(w.assign(location_id=loc_id))
        acc_all.append(acc.assign(location_id=loc_id))
        if gps is not None:
            gps_all.append(gps.assign(location_id=loc_id))
        if incl is not None:
            incl_all.append(incl.assign(location_id=loc_id))
        if write and r is not None:
            plot_site(loc_id, r, row["confidence"], out_dir)

    summary = pd.DataFrame(rows)
    if os.path.exists(manifest):
        summary = summary.merge(pd.read_csv(manifest), on="location_id", how="left")
    summary = summary.sort_values("start_ist").reset_index(drop=True)
    prov = pd.DataFrame(prov).sort_values("start_ist")
    windows = pd.concat(win_all, ignore_index=True)

    if write:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
        windows.to_csv(os.path.join(out_dir, "windows.csv"), index=False)
        prov.assign(start_ist=prov.start_ist.astype(str), end_ist=prov.end_ist.astype(str)) \
            .to_csv(os.path.join(processed_dir, "recordings.csv"), index=False)
        curves = pd.concat([pd.DataFrame({"location_id": k, "freq_hz": r["freqs"], "hv_mean": r["hv_mean"],
                                          "hv_lower": r["hv_lower"], "hv_upper": r["hv_upper"]})
                            for k, r in results.items() if r is not None])
        acc_cat = pd.concat(acc_all, ignore_index=True).rename(columns={"t": "t_s"})
        os.makedirs(os.path.join(processed_dir, "sample"), exist_ok=True)
        acc_cat.groupby("location_id").head(200).to_csv(os.path.join(processed_dir, "sample", "accel_sample.csv"), index=False)
        try:
            summary.to_parquet(os.path.join(out_dir, "summary.parquet"), index=False)
            windows.to_parquet(os.path.join(out_dir, "windows.parquet"), index=False)
            curves.to_parquet(os.path.join(out_dir, "hvsr_curves.parquet"), index=False)
            acc_cat.to_parquet(os.path.join(processed_dir, "accel.parquet"), index=False)
            pd.concat(gps_all).rename(columns={"t": "t_s"}).to_parquet(os.path.join(processed_dir, "gps.parquet"), index=False)
            pd.concat(incl_all).rename(columns={"t": "t_s"}).to_parquet(os.path.join(processed_dir, "inclination.parquet"), index=False)
        except ImportError as e:
            print(f"(parquet export skipped: {e})")
        cols = ["location_id", "duration_s", "f0_hz", "a0", "n_windows_used", "n_windows_total",
                "sesame_reliability_passed", "sesame_clarity_passed", "band_frac_above_floor",
                "confidence", "gps_horizontal_accuracy_m", "mean_tilt_deg"]
        print(summary[cols].to_string(index=False))
    return {"summary": summary, "results": results, "windows": windows, "provenance": prov,
            "accel": acc_all, "gps": gps_all, "incl": incl_all}


def _selfcheck():
    # discovery must survive the real-world filenames seen in the field
    names = ["Core 4  Inclination.zip", "old_sac_accelaration(without g).zip",
             "Lecture hall lake ocation (GPS).zip", "Barak_Umium_Acceleration__without_g_.zip"]
    got = {n: next((site_id(n[:rx.search(n).start()]), k) for k, rx in KINDS.items() if rx.search(n)) for n in names}
    assert got[names[0]] == ("core_4", "incl"), got
    assert got[names[1]] == ("old_sac", "accel"), got
    assert got[names[2]] == ("lecture_hall_lake", "gps"), got
    assert got[names[3]] == ("barak_umium", "accel"), got
    # K-O weights: normalised, peak at the centre frequency
    f = np.linspace(0, 50, 1001)
    W = konno_ohmachi_matrix(f, np.array([5.0]))
    assert abs(W.sum() - 1) < 1e-9 and abs(f[np.argmax(W[0])] - 5.0) < 0.06
    # white noise on all 3 axes must NOT pass the floor gate
    rng = np.random.default_rng(0)
    n, fs = 20000, 100.0
    df = pd.DataFrame({"t": np.arange(n) / fs, **{c: rng.normal(0, 0.007, n) for c in ("ax", "ay", "az")}})
    df["a_abs"] = np.sqrt(df.ax ** 2 + df.ay ** 2 + df.az ** 2)
    r, _ = compute_hvsr(df, fs)
    assert floor_check(r)[0] < MIN_FRAC_ABOVE_FLOOR, "sensor-noise-only input slipped past the floor gate"
    assert 0.7 < np.median(r["hv_mean"]) < 1.4
    print("selfcheck ok")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        run()
