# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 14:37:16 2026
@author: sophi
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tifffile import imread, imwrite
from skimage import filters, morphology, feature, measure
from skimage.draw import disk
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


# =========================
# 0) 参数区（只改这里）
# =========================
img_dir = r"E:\Multi-SIM\20251128-474_Mito\474_2h_002_20251128_101933-980WF\Mito"

single_multich_tif = None  # 若用单个多通道tif，填路径；否则None

# 默认：Mito=488；UCNP=980
mito_glob = os.path.join(img_dir, "*488*.tif")
ucnp_glob = os.path.join(img_dir, "*980*.tif")

# 仅在 single_multich_tif 模式用
MITO_CH = 0
UCNP_CH = 1

pixel_size_nm = 30.0
frame_interval_s = 5.8
CONTACT_THR_NM = 100.0

# 外部 tracking CSV（如有则优先用）——【本版不再使用】
tracking_csv = None
FRAME_COL = "frame"
X_COL = "x"
Y_COL = "y"
TRACK_COL = "track_id"

# Debug：保存原图+mask叠加
SAVE_DEBUG_OVERLAY = True
DEBUG_FRAME = 0  # 保存第几帧的叠加图

# 细胞mask（用于随机对照），若没有就用 Mito+UCNP 的联合粗mask
cell_mask_path = None
N_RANDOM = 200

# ===== Random control settings（后面 meta / 随机检验要用）=====
USE_FULL_FOV_RANDOM = False   # False: 在 cell_mask 内随机；True: 全视野随机（去边缘）
RANDOM_BORDER_PX = 0          # USE_FULL_FOV_RANDOM=True 时，排除边缘像素宽度

# ===== Mito mask 变细参数 =====
MITO_MIN_OBJ = 60
MITO_CLOSE_RADIUS = 1
MITO_ERODE_RADIUS = 1
MITO_THIN_ITERS = 0

# --- 为了 meta 兼容：旧版可能用到的 mito 参数（本脚本不参与计算）---
MITO_BG_RADIUS = 0
MITO_THR_MODE = "otsu"   # 记录用途
MITO_PCTL = 0.0
MITO_K_MAD = 0.0
MITO_MIN_ABS = 0.0

# Mito mask 保存
SAVE_MITO_MASK_STACK = True
SAVE_MITO_MASK_PER_FRAME = True

# UCNP mask 保存（强度阈值分割得到）
SAVE_UCNP_MASK_STACK = True
SAVE_UCNP_MASK_PER_FRAME = True
SAVE_UCNP_LABEL_STACK = True
SAVE_UCNP_LABEL_PER_FRAME = False

# ====== UCNP 强度阈值分割参数（核心：让mask更像原图亮斑）======
UCNP_TOPHAT_RADIUS = 10
UCNP_MASK_PCTL = 99.5
UCNP_MIN_OBJ_PX = 8
UCNP_CLOSE_RADIUS = 1
UCNP_FILL_HOLES = True

# ====== 把每帧连通域作为对象后，跨帧链接成 mask_id 的参数 ======
AUTO_TRACK_OBJECTS = True
MAX_LINK_DIST_PX = 6.0
ALLOW_GAP_FRAMES = 0
MIN_TRACK_LEN = 2

out_dir = os.path.join(img_dir, "_Mito_UCNP_positional")
os.makedirs(out_dir, exist_ok=True)


# =========================
# 1) 读入图像（Mito + UCNP）
# =========================
def load_series():
    if single_multich_tif is not None:
        arr = imread(single_multich_tif)
        if arr.ndim != 4:
            raise ValueError(f"single_multich_tif 需要 4D (T,C,H,W). 当前: {arr.shape}")
        T_, C_, H_, W_ = arr.shape
        mito = arr[:, MITO_CH, :, :].astype(np.float32)
        ucnp = arr[:, UCNP_CH, :, :].astype(np.float32)
        return mito, ucnp

    mito_files = sorted(glob.glob(mito_glob))
    u_files = sorted(glob.glob(ucnp_glob))
    if len(mito_files) == 0 or len(u_files) == 0:
        raise FileNotFoundError("没有找到Mito或UCNP序列文件。请检查 mito_glob / ucnp_glob。")
    if len(mito_files) != len(u_files):
        raise ValueError(f"Mito文件数({len(mito_files)})与UCNP文件数({len(u_files)})不一致。")

    mito_stack, u_stack = [], []
    for mf, uf in zip(mito_files, u_files):
        mito_stack.append(imread(mf))
        u_stack.append(imread(uf))
    mito = np.stack(mito_stack, axis=0).astype(np.float32)   # T,H,W
    ucnp = np.stack(u_stack, axis=0).astype(np.float32)      # T,H,W
    return mito, ucnp

MITO, UCNP = load_series()
T, H, W = MITO.shape
print(f"Loaded Mito/UCNP stacks: T={T}, H={H}, W={W}")
print(f"UCNP raw min/max: {UCNP.min():.1f} / {UCNP.max():.1f}")


# =========================
# 2) 分割 Mito（mask）+ 距离变换
# =========================
def mito_mask_from_frame(img_mito):
    img = img_mito.astype(np.float32)

    thr = filters.threshold_otsu(img)
    m = img > thr

    m = morphology.remove_small_objects(m, MITO_MIN_OBJ)

    if MITO_CLOSE_RADIUS and MITO_CLOSE_RADIUS > 0:
        m = morphology.binary_closing(m, morphology.disk(int(MITO_CLOSE_RADIUS)))

    if MITO_ERODE_RADIUS and MITO_ERODE_RADIUS > 0:
        m = morphology.binary_erosion(m, morphology.disk(int(MITO_ERODE_RADIUS)))

    if MITO_THIN_ITERS and MITO_THIN_ITERS > 0:
        m = morphology.thin(m, max_num_iter=int(MITO_THIN_ITERS))

    return m

mito_masks = np.zeros((T, H, W), dtype=bool)
dist_to_mito_px = np.zeros((T, H, W), dtype=np.float32)

print("Building mito masks + distance transforms ...")
for t in range(T):
    m = mito_mask_from_frame(MITO[t])
    mito_masks[t] = m
    dist_to_mito_px[t] = ndi.distance_transform_edt(~m).astype(np.float32)

print("Saving Mito masks ...")
mito_masks_u8 = mito_masks.astype(np.uint8)

if SAVE_MITO_MASK_PER_FRAME:
    for t in range(T):
        imwrite(os.path.join(out_dir, f"mito_mask_t{t:03d}.tif"), mito_masks_u8[t])

if SAVE_MITO_MASK_STACK:
    imwrite(os.path.join(out_dir, "mito_mask_stack.tif"), mito_masks_u8, imagej=True)

print("Mito masks saved.")


# =========================
# 3) 细胞mask（用于随机对照）
# =========================
def build_cell_mask_fallback():
    mito_max = MITO.max(axis=0)
    ucnp_max = UCNP.max(axis=0)
    m1 = mito_max > filters.threshold_otsu(mito_max)
    m2 = ucnp_max > filters.threshold_otsu(ucnp_max)
    m = m1 | m2
    m = morphology.remove_small_objects(m, 200)
    m = morphology.binary_closing(m, morphology.disk(5))
    m = ndi.binary_fill_holes(m)
    return m.astype(bool)

if cell_mask_path and os.path.exists(cell_mask_path):
    cell_mask = imread(cell_mask_path).astype(bool)
    if cell_mask.ndim == 3:
        cell_mask = cell_mask.any(axis=0)
else:
    cell_mask = build_cell_mask_fallback()


# =========================
# 4) UCNP 强度阈值分割 -> UCNPmask + 保存
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
ucnp_label_u16 = np.zeros((T, H, W), dtype=np.uint16)

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
# 4a) 跨帧链接 -> mask_id
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
    active = {}

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
# 4b) Debug overlay
# =========================
if SAVE_DEBUG_OVERLAY:
    t0 = int(np.clip(DEBUG_FRAME, 0, T - 1))

    plt.figure(figsize=(6, 6), dpi=220)
    plt.imshow(UCNP[t0], cmap="gray")

    m = ucnp_masks_bool[t0]
    if np.any(m):
        b = m ^ morphology.binary_erosion(m, morphology.disk(1))
        yy, xx = np.where(b)
        plt.scatter(xx, yy, s=1, c="red")

    plt.title(f"UCNP frame {t0}: intensity-mask boundary overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"debug_overlay_mask_frame{t0:03d}.png"))
    plt.close()
    print(f"Saved debug overlay: debug_overlay_mask_frame{t0:03d}.png")


# =========================
# 5) 基于 UCNPmask 统计每个 mask_id 每帧的距离/接触
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

    dmin_px = float(np.min(dist_to_mito_px[t][m]))
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

tm = pd.DataFrame(track_metrics)
tm = tm[[
    "mask_id", "n_points", "median_d_nm", "mean_d_nm",
    "contact_fraction", "contact_time_s",
    "max_contiguous_contact_time_s", "n_contact_entries"
]].sort_values("mask_id")

tm.to_csv(os.path.join(out_dir, "track_metrics.csv"), index=False)
print(f"Saved track_metrics.csv (n_masks={tm.shape[0]})")


# =========================
# 7a) Randomization test（补齐：生成 real_p / rand_ps / p_value）
# =========================
print("Running randomization test ...")

# 真实：用与后续一致的定义（object-frame 里 contact 的比例）
real_p = float(np.mean(pts_df["contact"].to_numpy(dtype=np.float32)))

# 每帧真实对象数量（用于随机时匹配采样数）
n_obj_per_frame = pts_df.groupby(FRAME_COL).size().reindex(range(T), fill_value=0).to_numpy()

def get_random_pool_coords():
    """返回可随机抽样的坐标池（row, col）"""
    if USE_FULL_FOV_RANDOM:
        m = np.ones((H, W), dtype=bool)
        if RANDOM_BORDER_PX and RANDOM_BORDER_PX > 0:
            b = int(RANDOM_BORDER_PX)
            m[:b, :] = False
            m[-b:, :] = False
            m[:, :b] = False
            m[:, -b:] = False
        return np.column_stack(np.where(m))
    else:
        return np.column_stack(np.where(cell_mask))

pool_rc = get_random_pool_coords()
if pool_rc.shape[0] == 0:
    raise RuntimeError("随机采样池为空：请检查 cell_mask 或 USE_FULL_FOV_RANDOM/RANDOM_BORDER_PX 设置。")

thr_px = CONTACT_THR_NM / float(pixel_size_nm)
rand_ps = np.zeros(int(N_RANDOM), dtype=np.float32)

rng = np.random.default_rng()

for i in range(int(N_RANDOM)):
    total = 0
    hit = 0
    for t in range(T):
        n = int(n_obj_per_frame[t])
        if n <= 0:
            continue

        # 从池中有放回抽样 n 个点
        idx = rng.integers(0, pool_rc.shape[0], size=n, endpoint=False)
        rc = pool_rc[idx]  # (n,2) -> (r,c)
        rr = rc[:, 0]
        cc = rc[:, 1]

        dpx = dist_to_mito_px[t][rr, cc]
        hit += int(np.sum(dpx <= thr_px))
        total += n

    rand_ps[i] = (hit / total) if total > 0 else 0.0

# one-sided: P(random >= real)
p_value = float((np.sum(rand_ps >= real_p) + 1.0) / (len(rand_ps) + 1.0))
print(f"Random test done: real_p={real_p:.4f}, rand_mean={rand_ps.mean():.4f}, p={p_value:.4g}")


# =========================
# 7b) 保存 random test 关键数据（用于阶段汇总，Mito version）
# =========================
import json

np.save(os.path.join(out_dir, "random_p.npy"), np.asarray(rand_ps, dtype=np.float32))

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

    "mito_mask_params": {
        "MITO_BG_RADIUS": int(MITO_BG_RADIUS),
        "MITO_THR_MODE": str(MITO_THR_MODE),
        "MITO_PCTL": float(MITO_PCTL),
        "MITO_K_MAD": float(MITO_K_MAD),
        "MITO_MIN_ABS": float(MITO_MIN_ABS),
        "MITO_MIN_OBJ": int(MITO_MIN_OBJ),
        "MITO_CLOSE_RADIUS": int(MITO_CLOSE_RADIUS),
        "MITO_ERODE_RADIUS": int(MITO_ERODE_RADIUS),
        "MITO_THIN_ITERS": int(MITO_THIN_ITERS),
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
    "rand_mean": float(rand_ps.mean()),
    "rand_std": float(rand_ps.std(ddof=0)),
    "N_RANDOM": N_RANDOM,
    "T": T,
    "pixel_size_nm": pixel_size_nm,
    "frame_interval_s": frame_interval_s,
    "CONTACT_THR_NM": CONTACT_THR_NM,
    "RANDOM_BORDER_PX": RANDOM_BORDER_PX,
    "USE_FULL_FOV_RANDOM": USE_FULL_FOV_RANDOM,
    "total_object_frame_rows_final": len(pts_df),
}])

metrics_row.to_csv(
    os.path.join(out_dir, "random_test_metrics.csv"),
    index=False
)

print(
    "Saved random test core outputs (Mito): "
    "random_p.npy, p_obs.txt, random_test_meta.json, random_test_metrics.csv"
)


# =========================
# 8) 作图 + 总结（逻辑不变）
# =========================
plt.figure(figsize=(6.2, 4.4), dpi=160)
plt.hist(pts_df["d_nm"], bins=40)
plt.axvline(CONTACT_THR_NM, linestyle="--")
plt.xlabel("UCNP→Mito nearest distance (nm)")
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
plt.xlabel("UCNP→Mito nearest distance (nm)")
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
summary.append("UCNP mask segmentation:")
summary.append(
    f"  TOPHAT_RADIUS={UCNP_TOPHAT_RADIUS}, MASK_PCTL={UCNP_MASK_PCTL}, "
    f"MIN_OBJ={UCNP_MIN_OBJ_PX}, CLOSE_R={UCNP_CLOSE_RADIUS}, FILL={UCNP_FILL_HOLES}"
)
summary.append("Object linking:")
summary.append(f"  MAX_LINK_DIST_PX={MAX_LINK_DIST_PX}, GAP={ALLOW_GAP_FRAMES}, MIN_TRACK_LEN={MIN_TRACK_LEN}")

if tm is not None and len(tm) > 0:
    summary.append("")
    summary.append("Mask-level (median across masks):")
    summary.append(f"  median(mean_d_nm)={tm['mean_d_nm'].median():.1f} nm")
    summary.append(f"  median(median_d_nm)={tm['median_d_nm'].median():.1f} nm")
    summary.append(f"  median(contact_fraction)={tm['contact_fraction'].median():.3f}")
    summary.append(f"  median(contact_time_s)={tm['contact_time_s'].median():.1f} s")
    summary.append(f"  median(max_contiguous_contact_time_s)={tm['max_contiguous_contact_time_s'].median():.1f} s")
    summary.append(f"  median(n_contact_entries)={tm['n_contact_entries'].median():.1f}")

with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary))

print("\n".join(summary))
print(f"\nSaved to: {out_dir}")
