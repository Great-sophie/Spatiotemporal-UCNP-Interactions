# -*- coding: utf-8 -*-
"""
FIGURE 6 (FAST FINAL) | Fig6B + Fig6C (randomization) + SI robustness
✅ Designed for YOUR folder structure (as in screenshot):

BASE_DIR/
  2h_1ER_UCNP_positional/
  2h_1Golgi_UCNP_positional/
  2h_1Mito_UCNP_positional/
  ...
  8h_1Mito_UCNP_positional/

Each positional folder contains:
  - points_distances.csv/xlsx (columns: frame,x,y,mask_id,d_nm,contact(optional))
  - organelle mask stack: er_mask_stack.tif / mito_mask_stack.tif / golgi_mask_stack.tif (recommended)
  - ROI/cell mask: cell_roi_dilated.tif (preferred) or cell_mask.tif

Key optimizations (FAST + robust):
  1) Randomization uses POOLED sampling (match total_N), not per-frame loops → much faster.
  2) If ROI mask is empty or missing, automatically fallback to full FOV (no crash).
  3) Strictly matches masks within the same positional folder (cell-resolved, reviewer-safe).

Outputs:
  - Fig6B_proximity_probability_longTable.csv
  - Fig6B_cross_organelle_Pd100nm.tiff
  - Fig6C_randomization_summary.csv
  - Fig6C_randomization_PD100nm_1x3.tiff
  - Fig6C_threshold_robustness_SI.csv
  - Fig6C_error_log.csv (if any folder fails; analysis continues)

Author: generated for Fei
"""

import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from tifffile import imread
from scipy import ndimage as ndi

# =========================
# 0) ONLY EDIT HERE
# =========================
BASE_DIR = r"E:\Multi-SIM\Figur6\Fig6"   # ← contains 2h_1ER_UCNP_positional ...
OUT_DIR  = r"E:\Multi-SIM\Figur6"        # ← where to save figures/tables

TIMEPOINTS = ["2h", "4h", "6h", "8h"]
N_REP = 3

CONTACT_THR_NM = 100.0
ROBUST_THR_LIST = [50, 100, 150]

pixel_size_nm = 30.0      # ✅ confirmed
N_ITER = 800              # ✅ fast & stable; debug: 200–300; final: 800–1500
RANDOM_SEED = 0
DPI = 600

COLORS = {"Mito": "#C00000", "ER": "#7030A0", "Golgi": "#FFC000"}  # red/purple/gold

# points_distances columns (your header)
FRAME_COL = "frame"
X_COL = "x"
Y_COL = "y"
D_COL = "d_nm"

# =========================
# outputs
# =========================
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV_B = os.path.join(OUT_DIR, "Fig6B_proximity_probability_longTable.csv")
OUT_FIG_B = os.path.join(OUT_DIR, "Fig6B_cross_organelle_Pd100nm.tiff")

OUT_CSV_C = os.path.join(OUT_DIR, "Fig6C_randomization_summary.csv")
OUT_FIG_C = os.path.join(OUT_DIR, "Fig6C_randomization_PD100nm_1x3.tiff")
OUT_SI_C  = os.path.join(OUT_DIR, "Fig6C_threshold_robustness_SI.csv")
OUT_ERR   = os.path.join(OUT_DIR, "Fig6C_error_log.csv")

# =========================
# global style
# =========================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 20

# =========================
# 1) helpers
# =========================
def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")

def ensure_cols(df: pd.DataFrame, required_cols):
    miss = [c for c in required_cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns {miss}. Found: {list(df.columns)}")

def parse_folder_name(folder_path: str):
    """
    From: 2h_1ER_UCNP_positional
    Return: tp='2h', rep=1, org='ER'
    """
    bn = os.path.basename(folder_path)
    m = re.search(r"(2h|4h|6h|8h)_([1-9]\d*)\s*(ER|Mito|Golgi)", bn, flags=re.IGNORECASE)
    if not m:
        return None, None, None
    tp = m.group(1).lower()
    rep = int(m.group(2))
    org_raw = m.group(3).lower()
    if org_raw == "mito":
        org = "Mito"
    elif org_raw == "golgi":
        org = "Golgi"
    else:
        org = "ER"
    return tp, rep, org

def read_mask_2d(path: str) -> np.ndarray:
    """
    Read tif (2D or stack). If stack, do max projection.
    Return boolean.
    """
    arr = imread(path)
    if arr.ndim == 2:
        return (arr > 0)
    arr2 = np.max(arr, axis=0)
    return (arr2 > 0)

def build_distance_map_nm(organelle_mask_bool: np.ndarray, px_nm: float) -> np.ndarray:
    inv = ~organelle_mask_bool
    dist_px = ndi.distance_transform_edt(inv)
    return dist_px * float(px_nm)

def sample_random_points(pool_mask_bool: np.ndarray, n: int, rng: np.random.Generator):
    ys, xs = np.where(pool_mask_bool)
    if len(xs) == 0:
        return None, None
    idx = rng.integers(0, len(xs), size=int(n))
    return ys[idx], xs[idx]

def summarize_null(null_arr, alpha=0.05):
    lo = float(np.quantile(null_arr, alpha/2))
    hi = float(np.quantile(null_arr, 1 - alpha/2))
    return float(null_arr.mean()), lo, hi

def find_points_file(pos_folder: str):
    cands = []
    cands += glob.glob(os.path.join(pos_folder, "*points*distances*.csv"))
    cands += glob.glob(os.path.join(pos_folder, "*points*distances*.xlsx"))
    cands += glob.glob(os.path.join(pos_folder, "*points*distances*.xls"))
    if not cands:
        return None
    cands.sort(key=os.path.getmtime, reverse=True)
    return cands[0]

def find_organelle_mask(pos_folder: str, organelle: str):
    """
    Prefer *mask*stack*.tif ; else any *{org}*mask*.tif excluding ucnp_mask
    """
    org_low = organelle.lower()

    # prefer stack
    stack = glob.glob(os.path.join(pos_folder, f"*{org_low}*mask*stack*.tif")) + \
            glob.glob(os.path.join(pos_folder, f"*{org_low}*mask*stack*.tiff"))
    if stack:
        stack.sort(key=os.path.getmtime, reverse=True)
        return stack[0]

    # fallback
    cands = []
    cands += glob.glob(os.path.join(pos_folder, f"*{org_low}*mask*.tif"))
    cands += glob.glob(os.path.join(pos_folder, f"*{org_low}*mask*.tiff"))
    cands += glob.glob(os.path.join(pos_folder, f"*mask*{org_low}*.tif"))
    cands += glob.glob(os.path.join(pos_folder, f"*mask*{org_low}*.tiff"))
    cands = [p for p in cands if "ucnp_mask" not in os.path.basename(p).lower()]
    if cands:
        cands.sort(key=os.path.getmtime, reverse=True)
        return cands[0]
    return None

def find_roi_mask(pos_folder: str):
    """
    Prefer cell_roi_dilated; else cell_mask; else other roi/pool mask.
    """
    c1 = glob.glob(os.path.join(pos_folder, "*cell_roi_dilated*.tif")) + \
         glob.glob(os.path.join(pos_folder, "*cell_roi_dilated*.tiff"))
    if c1:
        c1.sort(key=os.path.getmtime, reverse=True)
        return c1[0]

    c2 = glob.glob(os.path.join(pos_folder, "*cell_mask*.tif")) + \
         glob.glob(os.path.join(pos_folder, "*cell_mask*.tiff"))
    if c2:
        c2.sort(key=os.path.getmtime, reverse=True)
        return c2[0]

    c3 = glob.glob(os.path.join(pos_folder, "*roi*mask*.tif")) + \
         glob.glob(os.path.join(pos_folder, "*pool*mask*.tif"))
    if c3:
        c3.sort(key=os.path.getmtime, reverse=True)
        return c3[0]
    return None

# =========================
# 2) discover positional folders
# =========================
pos_folders = [p for p in glob.glob(os.path.join(BASE_DIR, "*_UCNP_positional")) if os.path.isdir(p)]
pos_folders += [p for p in glob.glob(os.path.join(BASE_DIR, "*UCNP*positional*")) if os.path.isdir(p)]
pos_folders = sorted(list(set(pos_folders)))

if not pos_folders:
    raise FileNotFoundError(f"No *_UCNP_positional folders found under: {BASE_DIR}")

# index: org -> tp -> rep -> folder
index = {}
for pf in pos_folders:
    tp, rep, org = parse_folder_name(pf)
    if tp is None:
        continue
    if tp not in TIMEPOINTS:
        continue
    if rep > N_REP:
        continue
    index.setdefault(org, {}).setdefault(tp, {})[rep] = pf

# print what found
for org in ["Mito", "ER", "Golgi"]:
    for tp in TIMEPOINTS:
        reps = sorted(index.get(org, {}).get(tp, {}).keys())
        if reps:
            print(f"[FOUND] {org} {tp}: reps {reps}")

rng = np.random.default_rng(RANDOM_SEED)

# =========================
# 3) FIG6B
# =========================
rows_B = []
for org in ["Mito", "ER", "Golgi"]:
    for tp in TIMEPOINTS:
        rep_map = index.get(org, {}).get(tp, {})
        for rep, folder in sorted(rep_map.items()):
            pts_file = find_points_file(folder)
            if pts_file is None:
                print(f"[WARN] points_distances not found: {folder}")
                continue

            df = load_table(pts_file)
            ensure_cols(df, [FRAME_COL, X_COL, Y_COL, D_COL])

            d_nm = df[D_COL].to_numpy(dtype=float)
            p_obs = float(np.mean(d_nm <= CONTACT_THR_NM))

            rows_B.append(dict(
                organelle=org, timepoint=tp, replicate=rep,
                thr_nm=float(CONTACT_THR_NM),
                proximity_prob=p_obs,
                points_n=int(len(df)),
                positional_folder=folder,
                points_file=pts_file
            ))

df_B = pd.DataFrame(rows_B)
df_B.to_csv(OUT_CSV_B, index=False)
print("Saved:", OUT_CSV_B)

# plot Fig6B
plt.figure(figsize=(7.8, 5.2))
x = np.arange(len(TIMEPOINTS), dtype=float)

for org in ["Mito", "ER", "Golgi"]:
    sub = df_B[df_B["organelle"] == org]
    if sub.empty:
        continue
    grp = sub.groupby("timepoint", sort=False).agg(
        mean=("proximity_prob", "mean"),
        sem=("proximity_prob", lambda s: np.std(s, ddof=1)/np.sqrt(len(s)) if len(s) > 1 else 0.0),
    ).reindex(TIMEPOINTS)

    plt.errorbar(
        x, grp["mean"].to_numpy(),
        yerr=grp["sem"].to_numpy(),
        fmt="o-", lw=2.5, capsize=4,
        color=COLORS[org],
        label=org
    )

plt.xticks(x, TIMEPOINTS)
plt.ylim(0, 1.0)
plt.xlabel("Post-incubation time (h)")
plt.ylabel(f"Proximity probability (d ≤ {int(CONTACT_THR_NM)} nm)")
plt.legend(frameon=False, fontsize=14)
plt.tight_layout()
plt.savefig(OUT_FIG_B, dpi=DPI, format="tiff")
plt.close()
print("Saved:", OUT_FIG_B)

# =========================
# 4) FIG6C (FAST pooled randomization) + SI robustness
# =========================
rows_C = []
rows_SI = []
err_rows = []

for org in ["Mito", "ER", "Golgi"]:
    for tp in TIMEPOINTS:
        rep_map = index.get(org, {}).get(tp, {})
        for rep, folder in sorted(rep_map.items()):
            try:
                pts_file = find_points_file(folder)
                if pts_file is None:
                    raise FileNotFoundError("points_distances not found")

                df = load_table(pts_file)
                ensure_cols(df, [FRAME_COL, X_COL, Y_COL, D_COL])

                org_mask_path = find_organelle_mask(folder, org)
                if org_mask_path is None:
                    raise FileNotFoundError(f"{org} organelle mask not found")

                roi_mask_path = find_roi_mask(folder)

                org_m = read_mask_2d(org_mask_path)
                if roi_mask_path is not None:
                    pool_m = read_mask_2d(roi_mask_path)
                else:
                    pool_m = np.ones_like(org_m, dtype=bool)

                if pool_m.shape != org_m.shape:
                    raise ValueError(f"Mask shape mismatch: org={org_m.shape}, roi={pool_m.shape}")

                # build distance map
                dist_map_nm = build_distance_map_nm(org_m, pixel_size_nm)

                # total points (pooled across frames)
                total_N = int(len(df))
                d_nm = df[D_COL].to_numpy(dtype=float)

                # observed
                obs_p = float(np.mean(d_nm <= CONTACT_THR_NM))

                # ROI empty -> fallback to full FOV (no crash)
                ry1, rx1 = sample_random_points(pool_m, 1, rng)
                if ry1 is None:
                    print(f"[WARN] ROI empty -> full FOV | {org} {tp} rep{rep}")
                    pool_m = np.ones_like(org_m, dtype=bool)

                # ----- FAST pooled null distribution -----
                null_ps = np.zeros(N_ITER, dtype=float)
                for i in range(N_ITER):
                    ry, rx = sample_random_points(pool_m, total_N, rng)
                    if ry is None:
                        raise ValueError("Pool/ROI mask empty (even after fallback).")
                    rd = dist_map_nm[ry, rx]
                    null_ps[i] = float(np.mean(rd <= CONTACT_THR_NM))

                p_one = float(np.mean(null_ps >= obs_p))
                null_mean, null_lo, null_hi = summarize_null(null_ps, alpha=0.05)

                rows_C.append(dict(
                    organelle=org, timepoint=tp, replicate=rep,
                    thr_nm=float(CONTACT_THR_NM),
                    P_obs=obs_p,
                    P_null_mean=null_mean,
                    P_null_CI95_lo=null_lo,
                    P_null_CI95_hi=null_hi,
                    p_one_sided=p_one,
                    points_n=total_N,
                    used_roi_mask=bool(roi_mask_path is not None),
                    positional_folder=folder,
                    points_file=pts_file,
                    organelle_mask=org_mask_path,
                    roi_mask=(roi_mask_path if roi_mask_path else "")
                ))

                # ----- SI robustness (same pooled randomization) -----
                for thr in ROBUST_THR_LIST:
                    obs_thr = float(np.mean(d_nm <= thr))
                    null_thr = np.zeros(N_ITER, dtype=float)
                    for i in range(N_ITER):
                        ry, rx = sample_random_points(pool_m, total_N, rng)
                        rd = dist_map_nm[ry, rx]
                        null_thr[i] = float(np.mean(rd <= thr))

                    p_thr = float(np.mean(null_thr >= obs_thr))
                    m_thr, lo_thr, hi_thr = summarize_null(null_thr, alpha=0.05)

                    rows_SI.append(dict(
                        organelle=org, timepoint=tp, replicate=rep,
                        thr_nm=float(thr),
                        P_obs=obs_thr,
                        P_null_mean=m_thr,
                        P_null_CI95_lo=lo_thr,
                        P_null_CI95_hi=hi_thr,
                        p_one_sided=p_thr,
                        points_n=total_N,
                        positional_folder=folder
                    ))

                print(f"[OK] Fig6C {org} {tp} rep{rep} | mask={os.path.basename(org_mask_path)} | ROI={'Y' if roi_mask_path else 'N'}")

            except Exception as e:
                err_rows.append({
                    "organelle": org, "timepoint": tp, "replicate": rep,
                    "folder": folder, "error": repr(e)
                })
                print(f"[SKIP] {org} {tp} rep{rep} | {repr(e)}")

# save C tables
df_C = pd.DataFrame(rows_C)
df_SI = pd.DataFrame(rows_SI)
df_C.to_csv(OUT_CSV_C, index=False)
df_SI.to_csv(OUT_SI_C, index=False)
print("Saved:", OUT_CSV_C)
print("Saved:", OUT_SI_C)

# save error log if exists
if err_rows:
    pd.DataFrame(err_rows).to_csv(OUT_ERR, index=False)
    print("Saved:", OUT_ERR)

# =========================
# 5) Plot Fig6C 1×3 panel
# =========================
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharey=True)

for ax, org in zip(axes, ["Mito", "ER", "Golgi"]):
    sub = df_C[df_C["organelle"] == org]
    if sub.empty:
        ax.set_axis_off()
        continue

    grp = sub.groupby("timepoint", sort=False).agg(
        P_obs_mean=("P_obs", "mean"),
        P_obs_sem=("P_obs", lambda s: np.std(s, ddof=1)/np.sqrt(len(s)) if len(s) > 1 else 0.0),
        P_null_mean=("P_null_mean", "mean"),
        P_null_lo=("P_null_CI95_lo", "mean"),
        P_null_hi=("P_null_CI95_hi", "mean"),
        p_med=("p_one_sided", "median"),
    ).reindex(TIMEPOINTS)

    x = np.arange(len(TIMEPOINTS), dtype=float)

    ax.fill_between(x, grp["P_null_lo"].to_numpy(), grp["P_null_hi"].to_numpy(), alpha=0.25)
    ax.plot(x, grp["P_null_mean"].to_numpy(), lw=2.0, color="black", label="Randomized (mean)")
    ax.errorbar(
        x, grp["P_obs_mean"].to_numpy(),
        yerr=grp["P_obs_sem"].to_numpy(),
        fmt="o-", lw=2.5, capsize=4,
        color=COLORS[org],
        label="Observed"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(TIMEPOINTS)
    ax.set_ylim(0, 1.0)
    ax.set_title(org)
    ax.set_xlabel("Post-incubation time (h)")

    # annotate p-values (median across replicates)
    for i, tp in enumerate(TIMEPOINTS):
        pv = grp.loc[tp, "p_med"]
        if pd.isna(pv):
            continue
        ytxt = min(0.98, float(grp.loc[tp, "P_obs_mean"]) + 0.07)
        ax.text(i, ytxt, f"p={pv:.3g}", ha="center", va="bottom", fontsize=12)

    if org == "Mito":
        ax.set_ylabel(f"Proximity probability (d ≤ {int(CONTACT_THR_NM)} nm)")

handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=14)
fig.tight_layout(rect=[0, 0, 0.93, 1])
fig.savefig(OUT_FIG_C, dpi=DPI, format="tiff")
plt.close(fig)
print("Saved:", OUT_FIG_C)

print("\nALL DONE.")
