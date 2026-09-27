"""Peterson (1993) New Low / High Noise Models: P(dB) = A + B*log10(T), acceleration PSD
rel. 1 (m/s^2)^2/Hz, rows = (period start s, A, B). Tabulated here only for 0.1-10 s."""
import numpy as np

NLNM = [(0.10, -162.36, 5.64), (0.17, -166.7, 0), (0.40, -170.0, -8.30), (0.80, -166.4, 28.90),
        (1.24, -168.6, 52.48), (2.40, -159.98, 29.81), (4.30, -141.1, 0), (5.00, -71.36, -99.77),
        (6.00, -97.26, -66.49), (10.0, -132.18, -31.57)]
NHNM = [(0.10, -108.73, -17.23), (0.22, -150.34, -80.50), (0.32, -122.31, -23.87),
        (0.80, -116.85, 32.51), (3.80, -108.48, 18.08), (4.60, -74.66, -32.95),
        (6.30, 0.66, -127.18), (7.90, -93.37, -22.42)]


def peterson(table, T):
    out = np.full_like(T, np.nan)
    edges = [r[0] for r in table] + [np.inf]
    for (t0, A, B), t1 in zip(table, edges[1:]):
        m = (T >= t0) & (T < t1)
        out[m] = A + B * np.log10(T[m])
    return out


def detect_fraction(minutes, phones=1, seg_s=10.0, k=2.0):
    """Weakest ground signal that stands out, as a fraction of one phone's hiss (amplitude).
    Averaging M independent 10 s spectra steadies the hiss estimate by 1/sqrt(M), so ground power
    of k/sqrt(M) times the hiss power becomes visible; stacking N phones side by side lowers the hiss
    itself by 1/sqrt(N). Conservative: no averaging across neighbouring frequencies."""
    return (k / (minutes * 60 / seg_s) ** 0.5) ** 0.5 / phones ** 0.5
