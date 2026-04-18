# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 13:00:45 2026

@author: sophi
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tifffile import imread, imwrite
from skimage import filters, morphology, measure
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


# =========================
# 0) 参数区（只改这里）
# =========================
img_dir = r"E:\Multi-SIM\20251208-SF-copy\474_Golgi\474 Golgi 4+6h\474_Golgi_6h_002_20251210_163508\Golgi"

single_multich_tif = None  # 若用单个多通道tif，填路径；否则None

# 默认：Golgi=488；UCNP=980
golgi_glob = os.path.join(img_dir, "*488*.tif")
ucnp_glob  = os.path.join(img_dir, "*980*.tif")

# 仅在 single_multich_tif 模式用
GOLGI_CH = 0
UCNP_CH  = 1

pixel_size_nm = 30.0
frame_interval_s = 5.4
CONTACT_THR_NM = 100.0

FRAME_COL = "frame"
X_COL = "x"
Y_COL = "y"

# Debug：保存原图+mask叠加
SAVE_DEBUG_OVERLAY = True
DEBUG_FRAME = 0  # 保存第几帧的叠加图

# 随机对照：不再用 cell mask
# 方案1（默认）：全图随机，但排除边缘一圈（避免黑边/漂移）
USE_FULL_FOV_RANDOM = True
RANDOM_BORDER_PX = 10
N_RANDOM = 200

# ========== Golgi mask（强度阈值） ==========
# 背景扣除：用 opening(rolling-ball近似) —— 半径别太大，避免算得慢/崩
GOLGI_BG_RADIUS = 20          # 10~30（SIM常用）
GOLGI_THR_MODE  = "pctl"      # "pctl" 或 "robust"
GOLGI_PCTL      = 99.2        # 98.5~99.7（越大越“贴亮结构”）
GOLGI_K_MAD     = 6.0         # robust：median + K*MAD（4~10）
GOLGI_MIN_ABS   = 0.0         # 背景扣除后阈值绝对下限（0~50）

# 形态学细化（让 mask 更像原图亮结构）
GOLGI_MIN_OBJ = 30            # 去小碎片（你可以 20~100）
GOLGI_CLOSE_RADIUS = 1        # closing 连通亮斑（0/1/2）
GOLGI_ERODE_RADIUS = 0        # 想更细就 1；先建议 0（避免吃掉信号）
GOLGI_THIN_ITERS = 0          # 细线化（慎用，一般不需要）

# 保存 Golgi mask
SAVE_GOLGI_MASK_STACK = True
SAVE_GOLGI_MASK_PER_FRAME = True

# ========== UCNP mask（强度阈值分割） ==========
SAVE_UCNP_MASK_STACK = True
SAVE_UCNP_MASK_PER_FRAME = True
SAVE_UCNP_LABEL_STACK = True
SAVE_UCNP_LABEL_PER_FRAME = False

UCNP_TOPHAT_RADIUS = 10
UCNP_MASK_PCTL = 99.5
UCNP_MIN_OBJ_PX = 8
UCNP_CLOSE_RADIUS = 1
UCNP_FILL_HOLES = True

# ========== 对象跨帧链接成 mask_id ==========
AUTO_TRACK_OBJECTS = True
MAX_LINK_DIST_PX = 6.0
ALLOW_GAP_FRAMES = 0
MIN_TRACK_LEN = 2

out_dir = os.path.join(img_dir, "_Golgi_UCNP_positional")
os.makedirs(out_dir, exist_ok=True)


# =========================
# 1) 读入图像（Golgi + UCNP）
# =========================
def load_series():
    if single_multich_tif is not None:
        arr = imread(single_multich_tif)
        if arr.ndim != 4:
            raise ValueError(f"single_multich_tif 需要 4D (T,C,H,W). 当前: {arr.shape}")
        golgi = arr[:, GOLGI_CH, :, :].astype(np.float32)
        ucnp  = arr[:, UCNP_CH,  :, :].astype(np.float32)
        return golgi, ucnp

    golgi_files = sorted(glob.glob(golgi_glob))
    u_files     = sorted(glob.glob(ucnp_glob))
    if len(golgi_files) == 0 or len(u_files) == 0:
        raise FileNotFoundError("没有找到Golgi或UCNP序列文件。请检查 golgi_glob / ucnp_glob。")
    if len(golgi_files) != len(u_files):
        raise ValueError(f"Golgi文件数({len(golgi_files)})与UCNP文件数({len(u_files)})不一致。")

    golgi_stack, u_stack = [], []
    for gf, uf in zip(golgi_files, u_files):
        golgi_stack.append(imread(gf))
        u_stack.append(imread(uf))
    golgi = np.stack(golgi_stack, axis=0).astype(np.float32)
    ucnp  = np.stack(u_stack, axis=0).astype(np.float32)
    return golgi, ucnp

GOLGI, UCNP = load_series()
T, H, W = GOLGI.shape
print(f"Loaded Golgi/UCNP stacks: T={T}, H={H}, W={W}")
print(f"UCNP raw min/max: {UCNP.min():.1f} / {UCNP.max():.1f}")


# =========================
# 2) Golgi 强度阈值分割（无 cell mask）
# =========================
def preprocess_opening_subtract(im, radius):
    """rolling-ball近似：opening 估计背景，然后相减（radius 不要太离谱）"""
    im = im.astype(np.float32)
    if radius and radius > 0:
        se = morphology.disk(int(radius))
        bg = morphology.opening(im, se)
        im = im - bg
        im = np.clip(im, 0, None)
    return im

def golgi_mask_from_frame(img_golgi):
    im = preprocess_opening_subtract(img_golgi, GOLGI_BG_RADIUS)

    vals = im[im > 0]
    if vals.size < 50:
        return np.zeros_like(im, dtype=bool)

    if GOLGI_THR_MODE.lower() == "pctl":
        thr = np.percentile(vals, GOLGI_PCTL)
    elif GOLGI_THR_MODE.lower() == "robust":
        med = np.median(vals)
        mad = np.median(np.abs(vals - med)) + 1e-6
        thr = med + GOLGI_K_MAD * mad
    else:
        raise ValueError("GOLGI_THR_MODE must be 'pctl' or 'robust'")

    thr = max(float(thr), float(GOLGI_MIN_ABS))
    m = im >= thr

    # 形态学：尽量“贴近亮结构”
    m = morphology.remove_small_objects(m, GOLGI_MIN_OBJ)

    if GOLGI_CLOSE_RADIUS and GOLGI_CLOSE_RADIUS > 0:
        m = morphology.binary_closing(m, morphology.disk(int(GOLGI_CLOSE_RADIUS)))

    if GOLGI_ERODE_RADIUS and GOLGI_ERODE_RADIUS > 0:
        m = morphology.binary_erosion(m, morphology.disk(int(GOLGI_ERODE_RADIUS)))

    if GOLGI_THIN_ITERS and GOLGI_THIN_ITERS > 0:
        m = morphology.thin(m, max_num_iter=int(GOLGI_THIN_ITERS))

    return m.astype(bool)

golgi_masks = np.zeros((T, H, W), dtype=bool)
dist_to_golgi_px = np.zeros((T, H, W), dtype=np.float32)

print("Building golgi masks (intensity threshold) + distance transforms ...")
for t in range(T):
    m = golgi_mask_from_frame(GOLGI[t])
    golgi_masks[t] = m
    dist_to_golgi_px[t] = ndi.distance_transform_edt(~m).astype(np.float32)

print("Saving Golgi masks ...")
golgi_masks_u8 = golgi_masks.astype(np.uint8)

if SAVE_GOLGI_MASK_PER_FRAME:
    for t in range(T):
        imwrite(os.path.join(out_dir, f"golgi_mask_t{t:03d}.tif"), golgi_masks_u8[t])

if SAVE_GOLGI_MASK_STACK:
    imwrite(os.path.join(out_dir, "golgi_mask_stack.tif"), golgi_masks_u8, imagej=True)

print("Golgi masks saved.")


# =========================
# 3) UCNP 强度阈值分割 -> UCNPmask + label + 保存
# =========================
def preprocess_ucnp_for_mask(img):
    im = img.astype(np.float32)
    if UCNP_TOPHAT_RADIUS and UCNP_TOPHAT_RADIUS > 0:
        se = morphology.disk(int(UCNP_TOPHAT_RADIUS))
        bg = morphology.opening(im, se)
        im = im - bg
        im = np.clip(im, 0, None)
    return im

def ucnp_mask_from_frame(img_ucnp):
    im = preprocess_ucnp_for_mask(img_ucnp)
    if np.all(im <= 0):
        return np.zeros_like(im, dtype=bool)

    thr = np.percentile(im, UCNP_MASK_PCTL)
    m = im >= thr

    m = morphology.remove_small_objects(m, UCNP_MIN_OBJ_PX)

    if UCNP_CLOSE_RADIUS and UCNP_CLOSE_RADIUS > 0:
        m = morphology.binary_closing(m, morphology.disk(int(UCNP_CLOSE_RADIUS)))

    if UCNP_FILL_HOLES:
        m = ndi.binary_fill_holes(m)

    return m.astype(bool)

print("Building UCNP masks (intensity-threshold segmentation) ...")
ucnp_masks_bool = np.zeros((T, H, W), dtype=bool)
ucnp_label_u16  = np.zeros((T, H, W), dtype=np.uint16)

obj_rows = []
for t in range(T):
    m = ucnp_mask_from_frame(UCNP[t])
    ucnp_masks_bool[t] = m

    lab, nlab = ndi.label(m, structure=np.ones((3, 3), dtype=np.int8))
    ucnp_label_u16[t] = lab.astype(np.uint16)

    if nlab > 0:
        props = measure.regionprops(lab)
        for p in props:
            cy, cx = p.centroid
            obj_rows.append({
                FRAME_COL: int(t),
                X_COL: float(cx),
                Y_COL: float(cy),
                "frame_label": int(p.label),
            })

print("Saving UCNP masks ...")
ucnp_masks_u8 = ucnp_masks_bool.astype(np.uint8)

if SAVE_UCNP_MASK_PER_FRAME:
    for t in range(T):
        imwrite(os.path.join(out_dir, f"ucnp_mask_t{t:03d}.tif"), ucnp_masks_u8[t])

if SAVE_UCNP_MASK_STACK:
    imwrite(os.path.join(out_dir, "ucnp_mask_stack.tif"), ucnp_masks_u8, imagej=True)

if SAVE_UCNP_LABEL_PER_FRAME:
    for t in range(T):
        imwrite(os.path.join(out_dir, f"ucnp_label_t{t:03d}.tif"), ucnp_label_u16[t])

if SAVE_UCNP_LABEL_STACK:
    imwrite(os.path.join(out_dir, "ucnp_label_stack.tif"), ucnp_label_u16, imagej=True)

print("UCNP masks saved.")

obj_df = pd.DataFrame(obj_rows)
print(f"Total UCNP objects (frame-wise components) = {len(obj_df)}")


# =========================
# 3a) 把每帧mask连通域对象跨帧链接 -> mask_id
# =========================
def build_tracks_nearest_neighbor(df_in, T, max_link_dist_px=6.0, allow_gap=0):
    df = df_in.copy()
    df["mask_id"] = -1

    frame_idx = {t: df.index[df[FRAME_COL] == t].to_numpy() for t in range(T)}
    coords = {}
    for t in range(T):
        idx = frame_idx[t]
        if len(idx) == 0:
            coords[t] = np.zeros((0, 2), dtype=float)
        else:
            coords[t] = np.vstack([
                df.loc[idx, X_COL].to_numpy(dtype=float),
                df.loc[idx, Y_COL].to_numpy(dtype=float)
            ]).T

    next_tid = 0
    active = {}  # mask_id -> (last_frame, last_xy)

    for t in range(T):
        idx_t = frame_idx[t]
        xy_t = coords[t]

        drop = [tid for tid, (lf, _) in active.items() if (t - lf > allow_gap + 1)]
        for tid in drop:
            active.pop(tid, None)

        if xy_t.shape[0] == 0:
            continue

        tree = cKDTree(xy_t)
        used = np.zeros(len(idx_t), dtype=bool)

        candidates = []
        for tid, (lf, lxy) in active.items():
            if t - lf > allow_gap + 1:
                continue
            dist, j = tree.query(lxy, k=1)
            candidates.append((dist, tid, j))
        candidates.sort(key=lambda x: x[0])

        new_active = dict(active)

        for dist, tid, j in candidates:
            if dist <= max_link_dist_px and (not used[j]):
                used[j] = True
                df.loc[idx_t[j], "mask_id"] = tid
                new_active[tid] = (t, xy_t[j])

        active = new_active

        for j in range(len(idx_t)):
            if not used[j]:
                tid = next_tid
                next_tid += 1
                df.loc[idx_t[j], "mask_id"] = tid
                active[tid] = (t, xy_t[j])

    return df

if AUTO_TRACK_OBJECTS:
    if len(obj_df) == 0:
        raise RuntimeError("UCNPmask 分割后没有任何连通域对象。请降低 UCNP_MASK_PCTL 或减小 UCNP_MIN_OBJ_PX。")

    print("Linking UCNP objects across frames -> mask_id ...")
    obj_df = build_tracks_nearest_neighbor(
        obj_df, T,
        max_link_dist_px=MAX_LINK_DIST_PX,
        allow_gap=ALLOW_GAP_FRAMES
    )

    track_len = obj_df.groupby("mask_id").size()
    keep_ids = track_len[track_len >= MIN_TRACK_LEN].index
    obj_df = obj_df[obj_df["mask_id"].isin(keep_ids)].copy()
    obj_df.reset_index(drop=True, inplace=True)

    print(f"Object tracking done. mask_id kept={len(keep_ids)}; object-frames kept={len(obj_df)}")


# =========================
# 4) Debug：保存叠加图（Golgi+Golgi mask, UCNP+mask）
# =========================
if SAVE_DEBUG_OVERLAY:
    t0 = int(np.clip(DEBUG_FRAME, 0, T - 1))

    # (1) Golgi + Golgi mask
    plt.figure(figsize=(6, 6), dpi=220)
    plt.imshow(GOLGI[t0], cmap="gray")
    gm = golgi_masks[t0]
    if np.any(gm):
        b = gm ^ morphology.binary_erosion(gm, morphology.disk(1))
        yy, xx = np.where(b)
        plt.scatter(xx, yy, s=1, c="cyan", label="Golgi mask")
    plt.legend(loc="lower right", fontsize=8)
    plt.title(f"Golgi frame {t0}: golgi mask (cyan)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"debug_golgi_mask_overlay_t{t0:03d}.png"))
    plt.close()

    # (2) UCNP + UCNP mask
    plt.figure(figsize=(6, 6), dpi=220)
    plt.imshow(UCNP[t0], cmap="gray")
    m = ucnp_masks_bool[t0]
    if np.any(m):
        b = m ^ morphology.binary_erosion(m, morphology.disk(1))
        yy, xx = np.where(b)
        plt.scatter(xx, yy, s=1, c="red")
    plt.title(f"UCNP frame {t0}: ucnp mask boundary (red)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"debug_ucnp_mask_overlay_t{t0:03d}.png"))
    plt.close()

    print(f"Saved debug overlays for frame {t0}.")


# =========================
# 5) 基于 UCNPmask 统计每个 mask_id 每帧的距离/接触（输出 pts_df）
# =========================
rows = []
for _, r in obj_df.iterrows():
    t = int(r[FRAME_COL])
    flab = int(r["frame_label"])
    mid = int(r["mask_id"])

    lab = ucnp_label_u16[t]
    m = (lab == flab)
    if not np.any(m):
        continue

    dmin_px = float(np.min(dist_to_golgi_px[t][m]))
    d_nm = dmin_px * pixel_size_nm
    contact = int(d_nm <= CONTACT_THR_NM)

    rows.append({
        FRAME_COL: t,
        X_COL: float(r[X_COL]),
        Y_COL: float(r[Y_COL]),
        "mask_id": mid,
        "d_nm": float(d_nm),
        "contact": int(contact),
    })

pts_df = pd.DataFrame(rows)
if len(pts_df) == 0:
    raise RuntimeError("pts_df 为空：mask对象距离采样失败。请检查 UCNPmask/label 是否正常。")

pts_df.to_csv(os.path.join(out_dir, "points_distances.csv"), index=False)
print(f"Distance sampling done. object-frame rows={len(pts_df)}")


# =========================
# 6) 轨迹级别 track_metrics（mask_id）
# =========================
def runs_of_ones(b):
    out = []
    run = 0
    for v in b:
        if v == 1:
            run += 1
        else:
            if run > 0:
                out.append(run)
                run = 0
    if run > 0:
        out.append(run)
    return out

track_metrics = []
print("Computing mask-level metrics ...")
df_sorted = pts_df.sort_values(["mask_id", FRAME_COL]).copy()

for mid, g in df_sorted.groupby("mask_id", sort=False):
    contact = g["contact"].to_numpy(dtype=np.int8)

    if contact.size >= 2:
        enter = int(np.sum((contact[1:] == 1) & (contact[:-1] == 0)))
    else:
        enter = 0

    n_contact = int(contact.sum())
    runs = runs_of_ones(contact)
    max_run = int(max(runs)) if len(runs) else 0

    track_metrics.append({
        "mask_id": mid,
        "n_points": int(len(g)),
        "median_d_nm": float(np.median(g["d_nm"])),
        "mean_d_nm": float(np.mean(g["d_nm"])),
        "contact_fraction": float(n_contact / max(len(g), 1)),
        "contact_time_s": float(n_contact * frame_interval_s),
        "max_contiguous_contact_time_s": float(max_run * frame_interval_s),
        "n_contact_entries": int(enter),
    })

tm = pd.DataFrame(track_metrics).sort_values("mask_id")
tm.to_csv(os.path.join(out_dir, "track_metrics.csv"), index=False)
print(f"Saved track_metrics.csv (n_masks={tm.shape[0]})")


# =========================
# 7) 随机对照（无 cell mask：全图随机 + 去边缘）
# =========================
if USE_FULL_FOV_RANDOM:
    valid = np.ones((H, W), dtype=bool)
    if RANDOM_BORDER_PX and RANDOM_BORDER_PX > 0:
        b = int(RANDOM_BORDER_PX)
        valid[:b, :] = False
        valid[-b:, :] = False
        valid[:, :b] = False
        valid[:, -b:] = False
    ys_mask, xs_mask = np.where(valid)
else:
    # 兜底：仍用全图
    ys_mask, xs_mask = np.where(np.ones((H, W), dtype=bool))

if len(xs_mask) < 100:
    raise ValueError("随机区域太小，无法做随机对照。")

def random_points(n):
    idx = np.random.randint(0, len(xs_mask), size=n)
    return xs_mask[idx], ys_mask[idx]

real_p = float((pts_df["d_nm"] <= CONTACT_THR_NM).mean())

rand_ps = []
print("Randomization test ...")
for _ in range(N_RANDOM):
    ps = []
    for t in range(T):
        n = int((pts_df[FRAME_COL] == t).sum())
        if n == 0:
            continue
        rx, ry = random_points(n)
        d_nm_r = dist_to_golgi_px[t, ry, rx] * pixel_size_nm
        ps.append(np.mean(d_nm_r <= CONTACT_THR_NM))
    rand_ps.append(float(np.mean(ps) if len(ps) else 0.0))

rand_ps = np.array(rand_ps)
p_value = float((np.sum(rand_ps >= real_p) + 1) / (len(rand_ps) + 1))

rand_ps = np.array(rand_ps)
p_value = float((np.sum(rand_ps >= real_p) + 1) / (len(rand_ps) + 1))

# =========================
# 7b) 保存 random test 关键数据（用于阶段汇总）
# =========================
import json

np.save(os.path.join(out_dir, "random_p.npy"), rand_ps.astype(np.float32))

with open(os.path.join(out_dir, "p_obs.txt"), "w", encoding="utf-8") as f:
    f.write(f"{real_p:.10f}\n")

meta = {
    "T": int(T),
    "H": int(H),
    "W": int(W),
    "pixel_size_nm": float(pixel_size_nm),
    "frame_interval_s": float(frame_interval_s),
    "CONTACT_THR_NM": float(CONTACT_THR_NM),
    "real_p": float(real_p),
    "p_value_one_sided": float(p_value),
    "rand_mean": float(rand_ps.mean()),
    "rand_std": float(rand_ps.std(ddof=0)),
    "N_RANDOM": int(N_RANDOM),
    "random_control": {
        "USE_FULL_FOV_RANDOM": bool(USE_FULL_FOV_RANDOM),
        "RANDOM_BORDER_PX": int(RANDOM_BORDER_PX),
    },
    "golgi_mask_params": {
        "GOLGI_BG_RADIUS": int(GOLGI_BG_RADIUS),
        "GOLGI_THR_MODE": str(GOLGI_THR_MODE),
        "GOLGI_PCTL": float(GOLGI_PCTL),
        "GOLGI_K_MAD": float(GOLGI_K_MAD),
        "GOLGI_MIN_ABS": float(GOLGI_MIN_ABS),
        "GOLGI_MIN_OBJ": int(GOLGI_MIN_OBJ),
        "GOLGI_CLOSE_RADIUS": int(GOLGI_CLOSE_RADIUS),
        "GOLGI_ERODE_RADIUS": int(GOLGI_ERODE_RADIUS),
        "GOLGI_THIN_ITERS": int(GOLGI_THIN_ITERS),
    },
    "ucnp_mask_params": {
        "UCNP_TOPHAT_RADIUS": int(UCNP_TOPHAT_RADIUS),
        "UCNP_MASK_PCTL": float(UCNP_MASK_PCTL),
        "UCNP_MIN_OBJ_PX": int(UCNP_MIN_OBJ_PX),
        "UCNP_CLOSE_RADIUS": int(UCNP_CLOSE_RADIUS),
        "UCNP_FILL_HOLES": bool(UCNP_FILL_HOLES),
    },
    "linking_params": {
        "AUTO_TRACK_OBJECTS": bool(AUTO_TRACK_OBJECTS),
        "MAX_LINK_DIST_PX": float(MAX_LINK_DIST_PX),
        "ALLOW_GAP_FRAMES": int(ALLOW_GAP_FRAMES),
        "MIN_TRACK_LEN": int(MIN_TRACK_LEN),
    },
    "counts": {
        "total_object_frame_rows_final": int(len(pts_df)),
        "n_masks_after_tracking": int(tm.shape[0]) if "tm" in globals() else None,
        "n_ucnp_objects_framewise": int(len(obj_df)) if "obj_df" in globals() else None,
    },
}

with open(os.path.join(out_dir, "random_test_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

metrics_row = pd.DataFrame([{
    "real_p": real_p,
    "p_value_one_sided": p_value,
    "rand_mean": rand_ps.mean(),
    "rand_std": rand_ps.std(ddof=0),
    "N_RANDOM": N_RANDOM,
    "T": T,
    "pixel_size_nm": pixel_size_nm,
    "frame_interval_s": frame_interval_s,
    "CONTACT_THR_NM": CONTACT_THR_NM,
    "RANDOM_BORDER_PX": RANDOM_BORDER_PX,
    "USE_FULL_FOV_RANDOM": USE_FULL_FOV_RANDOM,
    "total_object_frame_rows_final": len(pts_df),
}])
metrics_row.to_csv(os.path.join(out_dir, "random_test_metrics.csv"), index=False)

print("Saved random test core outputs: random_p.npy, p_obs.txt, random_test_meta.json, random_test_metrics.csv")



# =========================
# 8) 作图 + 总结
# =========================
plt.figure(figsize=(6.2, 4.4), dpi=160)
plt.hist(pts_df["d_nm"], bins=40)
plt.axvline(CONTACT_THR_NM, linestyle="--")
plt.xlabel("UCNP→Golgi nearest distance (nm)")
plt.ylabel("Count")
plt.title("Distance distribution")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "distance_hist.png"))
plt.close()

d = np.sort(pts_df["d_nm"].to_numpy())
cdf = np.arange(1, len(d) + 1) / len(d)
plt.figure(figsize=(6.2, 4.4), dpi=160)
plt.plot(d, cdf)
plt.axvline(CONTACT_THR_NM, linestyle="--")
plt.xlabel("UCNP→Golgi nearest distance (nm)")
plt.ylabel("CDF")
plt.title("Distance CDF")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "distance_cdf.png"))
plt.close()

plt.figure(figsize=(6.2, 4.4), dpi=160)
plt.hist(rand_ps, bins=25)
plt.axvline(real_p, linestyle="--")
plt.xlabel(f"Random P(d ≤ {CONTACT_THR_NM:.0f} nm)")
plt.ylabel("Count")
plt.title(f"Randomization test (p={p_value:.3g})")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "random_test.png"))
plt.close()

summary = []
summary.append(f"T={T}, pixel_size_nm={pixel_size_nm}, dt_s={frame_interval_s}")
summary.append(f"CONTACT_THR_NM={CONTACT_THR_NM}")
summary.append(f"Total object-frame rows (final)={len(pts_df)}")
summary.append(f"Real P(d≤thr)={real_p:.4f}")
summary.append(f"Random mean P(d≤thr)={rand_ps.mean():.4f} ± {rand_ps.std():.4f}")
summary.append(f"One-sided p (real > random)={p_value:.4g}")
summary.append("")
summary.append("Golgi mask (intensity-only):")
summary.append(f"  BG_RADIUS={GOLGI_BG_RADIUS}, MODE={GOLGI_THR_MODE}, PCTL={GOLGI_PCTL}, K_MAD={GOLGI_K_MAD}, MIN_ABS={GOLGI_MIN_ABS}")
summary.append(f"  MIN_OBJ={GOLGI_MIN_OBJ}, CLOSE_R={GOLGI_CLOSE_RADIUS}, ERODE_R={GOLGI_ERODE_RADIUS}, THIN={GOLGI_THIN_ITERS}")
summary.append("UCNP mask segmentation:")
summary.append(f"  TOPHAT_RADIUS={UCNP_TOPHAT_RADIUS}, MASK_PCTL={UCNP_MASK_PCTL}, MIN_OBJ={UCNP_MIN_OBJ_PX}, CLOSE_R={UCNP_CLOSE_RADIUS}, FILL={UCNP_FILL_HOLES}")
summary.append("Object linking:")
summary.append(f"  MAX_LINK_DIST_PX={MAX_LINK_DIST_PX}, GAP={ALLOW_GAP_FRAMES}, MIN_TRACK_LEN={MIN_TRACK_LEN}")
summary.append("Random control:")
summary.append(f"  FULL_FOV_RANDOM={USE_FULL_FOV_RANDOM}, BORDER_PX={RANDOM_BORDER_PX}, N_RANDOM={N_RANDOM}")

with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary))

print("\n".join(summary))
print(f"\nSaved to: {out_dir}")

