import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# =========================================================
# Fig.3C | Continuous Golgi residence time (percent + fit curve)
# - Times New Roman
# - Median REMOVED (not informative, often zero)
# =========================================================

# =========================
# 0) 你只改这里
# =========================
FIG3C_DIR = r"E:\Multi-SIM\20251208-SF-copy\474_Golgi\Golgi figure\residence time"
OUT_TIFF  = os.path.join(
    FIG3C_DIR,
    "Fig3C_contiguous_Golgi_residence_time_percent_fit_noMedian.tiff"
)

PHASES = ["2h", "4h", "6h", "8h"]
FILE_PREFIX = "{ph}_track_metrics"

COL_CANDIDATES = [
    "max_contiguous_golgi_proximal_time_s",
    "max_contiguous_golgi_prox_time_s",
    "max_contiguous_Golgi_proximal_time_s",
    "max_contiguous_Golgi_prox_time_s",
    "max_contiguous_golgi_time_s",
    # fallback
    "max_contiguous_contact_time_s",
    "max_contiguous_residence_time_s",
    "continuous_residence_time_s",
    "contiguous_residence_time_s",
    "max_contiguous_time_s",
    "max_contiguous_time",
]

# ====== Plot settings ======
Y_AS_PERCENT = True
PLOT_FIT_CURVE = True
FIT_DIST = "lognormal"
GAUSS_NPTS = 600
FIT_COLOR = "#FFC000"

XMAX = 45.0
BIN_W = 2.0
BINS = np.arange(0, XMAX + BIN_W, BIN_W)

# =========================
# Font sizes (adjust freely)
# =========================
FS_BASE      = 14
FS_TITLE     = 14
FS_SUPTITLE  = 16
FS_AXISLABEL = 15
FS_TICK      = 12
FS_ANNOT     = 12

# =========================
# Global font
# =========================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": FS_BASE,
})

# -------------------------
# Helpers
# -------------------------
def find_files(fig_dir, ph, prefix):
    hits = []
    for ext in [".csv", ".xlsx", ".xls"]:
        hits += glob.glob(os.path.join(fig_dir, prefix.format(ph=ph) + ext))
    return sorted(hits)

def read_table(path):
    return pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)

def pick_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        s = c.lower()
        if ("contig" in s or "continu" in s) and "time" in s:
            return c
    raise KeyError("No continuous residence time column found")

def clean_nonneg(x):
    x = pd.to_numeric(x, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    return x[x >= 0].to_numpy()

def lognormal_pdf(x, mu, sigma):
    pdf = np.zeros_like(x)
    m = x > 0
    pdf[m] = (1 / (x[m] * sigma * np.sqrt(2*np.pi))) * np.exp(
        -0.5 * ((np.log(x[m]) - mu) / sigma) ** 2
    )
    return pdf

def fit_lognormal(x):
    x = x[x > 0]
    if len(x) < 2:
        return None
    z = np.log(x)
    return np.mean(z), np.std(z, ddof=1)

# -------------------------
# Load data
# -------------------------
phase_data = {}
meta = {}

for ph in PHASES:
    files = find_files(FIG3C_DIR, ph, FILE_PREFIX)
    if not files:
        raise FileNotFoundError(f"No files for {ph}")

    xs = []
    for f in files:
        df = read_table(f)
        col = pick_column(df, COL_CANDIDATES)
        xs.append(clean_nonneg(df[col]))

    x = np.concatenate(xs)
    phase_data[ph] = np.clip(x, 0, XMAX)

    meta[ph] = {
        "n": len(x),
        "mean": float(np.mean(x)) if len(x) else np.nan
    }

# unified YMAX
YMAX = max(
    np.max(np.histogram(phase_data[ph], bins=BINS)[0] / max(len(phase_data[ph]),1) * 100)
    for ph in PHASES
) * 1.2

# -------------------------
# Plot
# -------------------------
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.4), dpi=200, sharey=True)

for i, ph in enumerate(PHASES):
    ax = axes[i]
    x = phase_data[ph]
    m = meta[ph]

    weights = np.ones_like(x) * (100 / len(x))
    ax.hist(x, bins=BINS, weights=weights, edgecolor="black", linewidth=0.6)

    # mean line only
    ax.axvline(m["mean"], color="black", linewidth=1.2)

    # fit curve
    params = fit_lognormal(x)
    if params:
        mu, sigma = params
        xx = np.linspace(0, XMAX, GAUSS_NPTS)
        yy = lognormal_pdf(xx, mu, sigma) * BIN_W * 100
        ax.plot(xx, yy, color=FIT_COLOR, linewidth=1.8)

    ax.set_title(ph, fontsize=FS_TITLE)
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.tick_params(labelsize=FS_TICK)

    for s in ax.spines.values():
        s.set_linewidth(0.8)

    if i == 0:
        ax.set_ylabel("Percent of tracks (%)", fontsize=FS_AXISLABEL)

    # annotation: n + mean ONLY
    ax.text(
        0.97, 0.88,
        f"n = {m['n']}\nmean = {m['mean']:.2g} s",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=FS_ANNOT,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6)
    )

fig.suptitle(
    "C  Continuous Golgi residence time",
    x=0.01, ha="left",
    fontsize=FS_SUPTITLE,
    fontweight="bold"
)

fig.supxlabel(
    "Maximum continuous residence time near Golgi (s)",
    y=0.015,
    fontsize=FS_AXISLABEL
)

plt.tight_layout(rect=[0.001, 0.06, 0.99, 0.90])
plt.savefig(OUT_TIFF, dpi=600)
plt.show()

print(f"[OK] Saved -> {OUT_TIFF}")
