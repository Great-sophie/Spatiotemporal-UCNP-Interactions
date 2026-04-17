import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# =========================
# 0) YOU ONLY EDIT HERE
# =========================
BASE_DIR = r"E:\Multi-SIM\20251124-SF-474_aizde lyso\Fig2"   # <- 改成你截图里“1h_track_metrics”等文件夹所在的那个目录
PHASES = ["1h", "2h", "4h", "6h", "8h"]

# column names in each track_metrics table
COL_MEDIAN_D = "median_d_nm"
COL_CONTACT_FRACTION = "contact_fraction"

# thresholds
CONTACT_REF_NM = 100.0
ESC_THR_NM = 300.0
SHOW_CONTACT_REF_LINE = True
SHOW_ESC_THR_LINE = True

# style
FONT_FAMILY = "Times New Roman"
FONT_SIZE = 20
TITLE_SIZE = 22
LABEL_SIZE = 16
TICK_SIZE = 18

POINT_SIZE = 18
POINT_ALPHA = 0.55
SUMMARY_LINEWIDTH = 2.2
SUMMARY_MARKERSIZE = 6

ERRORBAR = "IQR"   # None / "SEM" / "IQR"
ERRORBAR_LINEWIDTH = 1.6
ERRORBAR_CAPSIZE = 4

PHASE_COLORS = {
    "1h":  "#1f77b4",
    "2h":  "#1f77b4",
    "4h":  "#1f77b4",
    "6h":  "#1f77b4",
    "8h":  "#1f77b4",
    }

SHOW_BLACK_FRAME = True
FRAME_LINEWIDTH = 1.8

Y1_LIM = (0, 5500)   # distance
Y2_LIM = (0, 1.02)   # contact
Y3_LIM = (0, 100)    # escape %

SAVE_DPI = 600
SAVE_FMT = "tiff"    # "tiff" or "png"
FIG_W, FIG_H = 6.2, 5.0
JITTER_WIDTH = 0.22
N_LABEL_SIZE = 11


OUT_DIR = os.path.join(BASE_DIR, "_Fig2_panels_out")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 1) helpers
# =========================
def set_global_style():
    mpl.rcParams["font.family"] = FONT_FAMILY
    mpl.rcParams["font.size"] = FONT_SIZE
    mpl.rcParams["axes.titlesize"] = TITLE_SIZE
    mpl.rcParams["axes.labelsize"] = LABEL_SIZE
    mpl.rcParams["xtick.labelsize"] = TICK_SIZE
    mpl.rcParams["ytick.labelsize"] = TICK_SIZE

def style_axes(ax):
    if SHOW_BLACK_FRAME:
        for s in ax.spines.values():
            s.set_linewidth(FRAME_LINEWIDTH)
            s.set_color("black")
    ax.tick_params(width=FRAME_LINEWIDTH if SHOW_BLACK_FRAME else 1.0, length=5)

def read_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)

def summarize_with_error(vals, stat="median", err_mode=None):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, (np.nan, np.nan)

    center = float(np.median(vals)) if stat == "median" else float(np.mean(vals))

    if err_mode is None:
        return center, (np.nan, np.nan)
    if err_mode.upper() == "SEM":
        sem = float(np.std(vals, ddof=1) / np.sqrt(max(vals.size, 1))) if vals.size > 1 else 0.0
        return center, (center - sem, center + sem)
    if err_mode.upper() == "IQR":
        q1, q3 = np.percentile(vals, [25, 75])
        return center, (float(q1), float(q3))
    return center, (np.nan, np.nan)

def infer_phase_from_path(path):
    """
    Try to infer phase token like 1h/2h/4h/6h/8h from:
    - folder name
    - file name
    """
    s = path.replace("\\", "/").lower()
    # match '1h', '2h', '10h' as whole tokens
    m = re.search(r"(^|/|_)(\d{1,2}h)($|/|_)", s)
    if m:
        return m.group(2)
    return None

# =========================
# 2) AUTO-FIND all track_metrics tables (recursive)
# =========================
set_global_style()

candidates = []
for ext in ("csv", "xlsx", "xls"):
    candidates += glob.glob(os.path.join(BASE_DIR, "**", f"*track*metrics*.{ext}"), recursive=True)

print("\n[diagnose] BASE_DIR =", BASE_DIR)
print("[diagnose] found tables =", len(candidates))
for p in candidates[:30]:
    print("  -", p)
if len(candidates) > 30:
    print(f"  ... ({len(candidates)-30} more)")

# group by phase
phase_files = {ph: [] for ph in PHASES}
for p in candidates:
    ph = infer_phase_from_path(p)
    if ph in phase_files:
        phase_files[ph].append(p)

print("\n[diagnose] phase grouping:")
for ph in PHASES:
    print(f"  {ph}: {len(phase_files[ph])} files")

# load first file per phase (or you can change to choose newest)
phase_data = {}
used_file = {}
for ph in PHASES:
    if len(phase_files[ph]) == 0:
        continue
    # choose the first; better: choose newest modified
    chosen = sorted(phase_files[ph], key=lambda x: os.path.getmtime(x))[-1]
    df = read_table(chosen)

    if COL_MEDIAN_D not in df.columns or COL_CONTACT_FRACTION not in df.columns:
        raise KeyError(
            f"[{ph}] missing required columns.\n"
            f"Need: {COL_MEDIAN_D}, {COL_CONTACT_FRACTION}\n"
            f"Found: {list(df.columns)}\n"
            f"File: {chosen}"
        )

    d = pd.to_numeric(df[COL_MEDIAN_D], errors="coerce").to_numpy()
    c = pd.to_numeric(df[COL_CONTACT_FRACTION], errors="coerce").to_numpy()
    d = d[np.isfinite(d)]
    c = c[np.isfinite(c)]

    phase_data[ph] = {"d": d, "c": c}
    used_file[ph] = chosen
    print(f"[load] {ph}: n(d)={len(d)}, n(c)={len(c)} | {os.path.basename(chosen)}")

loaded_phases = [ph for ph in PHASES if ph in phase_data]
if len(loaded_phases) == 0:
    raise RuntimeError(
        "No phase data loaded.\n"
        "1) Check BASE_DIR points to the folder that CONTAINS 1h_track_metrics/2h_track_metrics/... \n"
        "2) Ensure your table filenames contain 'track' and 'metrics' (or adjust glob pattern)\n"
        "3) Ensure phase tokens like '1h' exist in folder/file names.\n"
    )

xpos = np.arange(len(loaded_phases))

# =========================
# 3) Panel 1: Distance
# =========================
def plot_panel1_distance():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=220)

    centers, low, high = [], [], []
    for i, ph in enumerate(loaded_phases):
        d = phase_data[ph]["d"]
        color = PHASE_COLORS.get(ph, "#C00000")

        jitter = (np.random.rand(d.size) - 0.5) * JITTER_WIDTH
        ax.scatter(np.full(d.size, xpos[i]) + jitter, d,
                   s=POINT_SIZE, alpha=POINT_ALPHA,
                   edgecolors="none", color=color)

        center, (l, h) = summarize_with_error(d, stat="median", err_mode=ERRORBAR)
        centers.append(center)
        low.append(l)
        high.append(h)

        # n label
        y_top = Y1_LIM[1] if Y1_LIM else (np.nanmax(d) if d.size else 1)
        ax.text(
            xpos[i], y_top * 0.98, f"n={len(d)}",
            ha="center", va="top", fontsize=N_LABEL_SIZE
        )

    centers = np.array(centers, float)
    low = np.array(low, float)
    high = np.array(high, float)

    ax.plot(
        xpos, centers,
        linewidth=SUMMARY_LINEWIDTH,
        marker="o",
        markersize=SUMMARY_MARKERSIZE,
        color="black"
    )

    if ERRORBAR is not None:
        yerr = np.vstack([centers - low, high - centers])
        ax.errorbar(
            xpos, centers, yerr=yerr,
            fmt="none",
            ecolor="black",
            elinewidth=ERRORBAR_LINEWIDTH,
            capsize=ERRORBAR_CAPSIZE
        )

    # -------------------------
    # Reference lines + Y-axis labels (CT / ET only)
    # -------------------------
    if SHOW_CONTACT_REF_LINE:
        ax.axhline(
            CONTACT_REF_NM,
            linestyle="--",
            linewidth=1.3,
            color="black",
            alpha=0.45
        )
        # CT label on Y-axis side
        ax.text(
            -0.04, CONTACT_REF_NM,
            "CT",
            transform=ax.get_yaxis_transform(),
            ha="right", va="center",
            fontsize=LABEL_SIZE - 6,
            alpha=0.6
        )

    if SHOW_ESC_THR_LINE:
        ax.axhline(
            ESC_THR_NM,
            linestyle="--",
            linewidth=1.3,
            color="black",
            alpha=0.75
        )
        # ET label on Y-axis side
        ax.text(
            -0.04, ESC_THR_NM,
            "ET",
            transform=ax.get_yaxis_transform(),
            ha="right", va="center",
            fontsize=LABEL_SIZE - 6,
            alpha=0.8
        )

    ax.set_ylabel("UCNP–lysosome distance(nm)", fontsize=LABEL_SIZE)
    ax.set_xlabel("Phase (post-incubation)", fontsize=LABEL_SIZE)
    ax.set_xticks(xpos)
    ax.set_xticklabels(loaded_phases)

    if Y1_LIM:
        ax.set_ylim(*Y1_LIM)

    style_axes(ax)

    fig.tight_layout()

    out = os.path.join(OUT_DIR, f"Fig3_Panel1_Distance_Mito.{SAVE_FMT}")
    fig.savefig(out, dpi=SAVE_DPI)
    plt.close(fig)
    print("[save]", out)


# =========================
# 4) Panel 2: Contact
# =========================
def plot_panel2_contact():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=220)

    centers, low, high = [], [], []
    for i, ph in enumerate(loaded_phases):
        c = phase_data[ph]["c"]
        color = PHASE_COLORS.get(ph, "#333333")

        jitter = (np.random.rand(c.size) - 0.5) * 0.22
        ax.scatter(np.full(c.size, xpos[i]) + jitter, c,
                   s=POINT_SIZE, alpha=POINT_ALPHA, edgecolors="none", color=color)

        center, (l, h) = summarize_with_error(c, stat="median", err_mode=ERRORBAR)
        centers.append(center); low.append(l); high.append(h)

        ax.text(xpos[i], 1.02, f"n={len(c)}", ha="center", va="bottom", fontsize=10)

    centers = np.array(centers, float)
    low = np.array(low, float)
    high = np.array(high, float)

    ax.plot(xpos, centers, linewidth=SUMMARY_LINEWIDTH, marker="o",
            markersize=SUMMARY_MARKERSIZE, color="black")

    if ERRORBAR is not None:
        yerr = np.vstack([centers - low, high - centers])
        ax.errorbar(xpos, centers, yerr=yerr, fmt="none", ecolor="black",
                    elinewidth=ERRORBAR_LINEWIDTH, capsize=ERRORBAR_CAPSIZE)

    #ax.set_title("Panel 2 | Contact", pad=12)
    ax.set_ylabel(f"Contact fraction (d ≤ {CONTACT_REF_NM:.0f} nm)",fontsize=15)
    ax.set_xlabel("Phase (post-incubation)",fontsize=15)
    ax.set_xticks(xpos)
    ax.set_xticklabels(loaded_phases)
    if Y2_LIM: ax.set_ylim(*Y2_LIM)

    style_axes(ax)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, f"Fig2_Panel2_Contact.{SAVE_FMT}")
    fig.savefig(out, dpi=SAVE_DPI)
    plt.close(fig)
    print("[save]", out)

# =========================
# 5) Panel 3: Escape
# =========================
def plot_panel3_escape():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=220)

    esc_pct = []
    ns = []
    for ph in loaded_phases:
        d = phase_data[ph]["d"]
        esc = float(np.mean(d > ESC_THR_NM) * 100.0) if d.size else np.nan
        esc_pct.append(esc)
        ns.append(int(d.size))

    esc_pct = np.array(esc_pct, float)

    ax.plot(xpos, esc_pct, linewidth=SUMMARY_LINEWIDTH, marker="o",
            markersize=SUMMARY_MARKERSIZE, color="#1f77b4")

    for i, n in enumerate(ns):
        ax.text(xpos[i], (Y3_LIM[1] if Y3_LIM else np.nanmax(esc_pct))*0.98,
                f"n={n}", ha="center", va="top", fontsize=10)

    #ax.set_title("Panel 3 | Escape", pad=12)
    ax.set_ylabel(f"Escape fraction (%)\n(median d > {ESC_THR_NM:.0f} nm)",fontsize=15)
    ax.set_xlabel("Phase (post-incubation)",fontsize=15)
    ax.set_xticks(xpos)
    ax.set_xticklabels(loaded_phases)
    if Y3_LIM: ax.set_ylim(*Y3_LIM)

    style_axes(ax)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, f"Fig2_Panel3_Escape.{SAVE_FMT}")
    fig.savefig(out, dpi=SAVE_DPI)
    plt.close(fig)
    print("[save]", out)

# =========================
# 6) Run
# =========================
plot_panel1_distance()
plot_panel2_contact()
plot_panel3_escape()

print("\nDone. Output:", OUT_DIR)
