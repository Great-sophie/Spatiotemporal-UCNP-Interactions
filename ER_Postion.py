import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tifffile import imread, imwrite
from skimage import filters, morphology, measure, exposure
from skimage.filters import frangi, threshold_sauvola
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


# =========================
# 0) 参数区（只改这里）
# =========================
img_dir = r"E:\Multi-SIM\20251201-474_ER\BT474_2h_002_20251201_102509\ER"

single_multich_tif = None  # 若用单个多通道tif，填路径；否则None

# ✅ ER=488；UCNP=980
er_glob   = os.path.join(img_dir, "*488*.tif")
ucnp_glob = os.path.join(img_dir, "*980*.tif")

# 仅在 single_multich_tif 模式用
ER_CH   = 0
UCNP_CH = 1

pixel_size_nm = 30.0
frame_interval_s = 5.4
CONTACT_THR_NM = 100.0

FRAME_COL = "frame"
X_COL = "x"
Y_COL = "y"

SAVE_DEBUG_OVERLAY = True
DEBUG_FRAME = 0

cell_mask_path = None
N_RANDOM = 200

# ====== ER mask：强力兜底版（保证不空）======
ER_MIN_OBJ = 20                 # ER细丝：不要太大
ER_CLOSE_RADIUS = 1
ER_FILL_HOLES = False           # ER通常不建议fill（会变块），先False
ER_THIN_ITERS = 0               # 可选：1~2（慎用）
ER_FORCE_MIN_FG_FRAC = 0.0008   # 前景占比太小则触发fallback
ER_TOP_PERCENT_FALLBACK = 0.8   # fallback：取最亮 top 0.8% 像素（0.5~2%）

# CLAHE
ER_CLAHE = True
ER_CLAHE_KERNEL = (32, 32)      # 16~64
ER_CLAHE_CLIP = 0.01            # 0.005~0.03

# Frangi（增强细丝/管状结构）
ER_USE_FRANGI = True
ER_FRANGI_SCALE_RANGE = (1, 3)  # 1~4 (像素尺度)
ER_FRANGI_SCALE_STEP = 1

# Sauvola 局部阈值
ER_SAUVOLA_WINDOW = 41          # 31~71（越大越平滑）
ER_SAUVOLA_K = 0.2              # 0.1~0.5

# 保存 ER masks
SAVE_ER_MASK_STACK = True
SAVE_ER_MASK_PER_FRAME = True

# UCNP mask 保存
SAVE_UCNP_MASK_STACK = True
SAVE_UCNP_MASK_PER_FRAME = True
SAVE_UCNP_LABEL_STACK = True
SAVE_UCNP_LABEL_PER_FRAME = False

# UCNP 强度阈值分割参数
UCNP_TOPHAT_RADIUS = 10
UCNP_MASK_PCTL = 99.5
UCNP_MIN_OBJ_PX = 8
UCNP_CLOSE_RADIUS = 1
UCNP_FILL_HOLES = True

# 对象跨帧链接
AUTO_TRACK_OBJECTS = True
MAX_LINK_DIST_PX = 6.0
ALLOW_GAP_FRAMES = 0
MIN_TRACK_LEN = 2

out_dir = os.path.join(img_dir, "_ER_UCNP_positional")
os.makedirs(out_dir, exist_ok=True)


# =========================
# 1) 读入图像（ER + UCNP）
# =========================
def load_series():
    if single_multich_tif is not None:
        arr = imread(single_multich_tif)
        if arr.ndim != 4:
            raise ValueError(f"single_multich_tif 需要 4D (T,C,H,W). 当前: {arr.shape}")
        er = arr[:, ER_CH, :, :].astype(np.float32)
        ucnp = arr[:, UCNP_CH, :, :].astype(np.float32)
        return er, ucnp

    er_files = sorted(glob.glob(er_glob))
    u_files  = sorted(glob.glob(ucnp_glob))
    if len(er_files) == 0 or len(u_files) == 0:
        raise FileNotFoundError("没有找到ER或UCNP序列文件。请检查 er_glob / ucnp_glob。")
    if len(er_files) != len(u_files):
        raise ValueError(f"ER文件数({len(er_files)})与UCNP文件数({len(u_files)})不一致。")

    er_stack, u_stack = [], []
    for ef, uf in zip(er_files, u_files):
        er_stack.append(imread(ef))
        u_stack.append(imread(uf))
    er = np.stack(er_stack, axis=0).astype(np.float32)
    ucnp = np.stack(u_stack, axis=0).astype(np.float32)
    return er, ucnp

ER, UCNP = load_series()
T, H, W = ER.shape
print(f"Loaded ER/UCNP stacks: T={T}, H={H}, W={W}")
print(f"ER raw min/max:   {ER.min():.1f} / {ER.max():.1f}")
print(f"UCNP raw min/max: {UCNP.min():.1f} / {UCNP.max():.1f}")


# =========================
# 2) ER 分割（保证不空） + 距离变换
# =========================
def robust_norm(im, p_lo=1, p_hi=99):
    im = im.astype(np.float32)
    lo, hi = np.percentile(im, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(im, dtype=np.float32)
    out = (im - lo) / (hi - lo)
    return np.clip(out, 0, 1).astype(np.float32)

def er_mask_from_frame(img_er):
    # 0) robust normalize
    im = robust_norm(img_er, 1, 99)
    if np.max(im) <= 0:
        return np.zeros_like(im, dtype=bool)

    # 1) CLAHE enhance (local contrast)
    if ER_CLAHE:
        im = exposure.equalize_adapthist(
            im, kernel_size=ER_CLAHE_KERNEL, clip_limit=ER_CLAHE_CLIP
        ).astype(np.float32)

    # 2) vesselness (Frangi) to enhance ER filaments
    if ER_USE_FRANGI:
        v = frangi(
            im,
            scale_range=ER_FRANGI_SCALE_RANGE,
            scale_step=ER_FRANGI_SCALE_STEP,
            black_ridges=False
        ).astype(np.float32)
        # normalize vesselness
        v = robust_norm(v, 1, 99.5)
    else:
        v = im

    # 3) Sauvola local threshold
    try:
        thr = threshold_sauvola(v, window_size=ER_SAUVOLA_WINDOW, k=ER_SAUVOLA_K)
        m = v > thr
    except Exception:
        # fallback to simple percentile if Sauvola fails
        m = v > np.percentile(v, 95)

    # 4) clean
    m = morphology.remove_small_objects(m, ER_MIN_OBJ)
    if ER_CLOSE_RADIUS and ER_CLOSE_RADIUS > 0:
        m = morphology.binary_closing(m, morphology.disk(int(ER_CLOSE_RADIUS)))

    if ER_FILL_HOLES:
        m = ndi.binary_fill_holes(m)

    # 5) if too sparse -> fallback: brightest top X%
    fg_frac = float(m.mean())
    if fg_frac < ER_FORCE_MIN_FG_FRAC:
        p = 100.0 - ER_TOP_PERCENT_FALLBACK  # e.g. top 0.8% => percentile 99.2
        thr2 = np.percentile(v, p)
        m2 = v >= thr2
        m2 = morphology.remove_small_objects(m2, max(5, ER_MIN_OBJ // 2))
        if ER_CLOSE_RADIUS and ER_CLOSE_RADIUS > 0:
            m2 = morphology.binary_closing(m2, morphology.disk(int(ER_CLOSE_RADIUS)))
        # 仍然不空才替换
        if m2.any():
            m = m2

    # 6) optional thin
    if ER_THIN_ITERS and ER_THIN_ITERS > 0:
        m2 = morphology.thin(m, max_num_iter=int(ER_THIN_ITERS))
        if m2.any():
            m = m2

    return m.astype(bool)

er_masks = np.zeros((T, H, W), dtype=bool)
dist_to_er_px = np.zeros((T, H, W), dtype=np.float32)

print("Building ER masks + distance transforms ...")
for t in range(T):
    m = er_mask_from_frame(ER[t])
    er_masks[t] = m
    dist_to_er_px[t] = ndi.distance_transform_edt(~m).astype(np.float32)

print(f"ER mask foreground fraction (median across frames): {np.median(er_masks.mean(axis=(1,2))):.6f}")

# 保存 ER masks
print("Saving ER masks ...")
er_masks_u8 = er_masks.astype(np.uint8)
if SAVE_ER_MASK_PER_FRAME:
    for t in range(T):
        imwrite(os.path.join(out_dir, f"er_mask_t{t:03d}.tif"), er_masks_u8[t])
if SAVE_ER_MASK_STACK:
    imwrite(os.path.join(out_dir, "er_mask_stack.tif"), er_masks_u8, imagej=True)
print("ER masks saved.")


# =========================
# 3) 细胞mask（用于随机对照）
# =========================
def build_cell_mask_fallback():
    er_max = ER.max(axis=0)
    ucnp_max = UCNP.max(axis=0)
    m1 = er_max > filters.threshold_otsu(er_max)
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

print("Building UCNP masks ...")
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

obj_df = pd.DataFrame(obj_rows)
print(f"Total UCNP objects (frame-wise components) = {len(obj_df)}")
if len(obj_df) == 0:
    raise RuntimeError("UCNPmask 分割后没有任何对象。请降低 UCNP_MASK_PCTL 或减小 UCNP_MIN_OBJ_PX。")


# =========================
# 4a) 跨帧链接对象 -> mask_id
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
    print("Linking UCNP objects across frames -> mask_id ...")
    obj_df = build_tracks_nearest_neighbor(obj_df, T, MAX_LINK_DIST_PX, ALLOW_GAP_FRAMES)

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
    plt.title(f"UCNP frame {t0}: mask boundary overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"debug_overlay_UCNPmask_frame{t0:03d}.png"))
    plt.close()

    plt.figure(figsize=(6, 6), dpi=220)
    plt.imshow(ER[t0], cmap="gray")
    m = er_masks[t0]
    if np.any(m):
        b = m ^ morphology.binary_erosion(m, morphology.disk(1))
        yy, xx = np.where(b)
        plt.scatter(xx, yy, s=1, c="lime")
    plt.title(f"ER frame {t0}: ER-mask boundary overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"debug_overlay_ERmask_frame{t0:03d}.png"))
    plt.close()

    print(f"Saved debug overlays (UCNP/ER) at frame {t0}")


# =========================
# 5) 基于 UCNPmask 统计每个 mask_id 每帧距离/接触（输出 pts_df）
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

    dmin_px = float(np.min(dist_to_er_px[t][m]))
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
# 6) 轨迹级别 metrics（mask_id）
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
# 7) 随机对照
# =========================
ys_mask, xs_mask = np.where(cell_mask)
if len(xs_mask) < 100:
    raise ValueError("cell_mask太小或为空，无法做随机对照。")

def random_points(n):
    idx = np.random.randint(0, len(xs_mask), size=n)
    return xs_mask[idx], ys_mask[idx]

real_p = float((pts_df["d_nm"] <= CONTACT_THR_NM).mean())

rand_ps = []
for _ in range(N_RANDOM):
    ps = []
    for t in range(T):
        n = int((pts_df[FRAME_COL] == t).sum())
        if n == 0:
            continue
        rx, ry = random_points(n)
        d_nm_r = dist_to_er_px[t, ry, rx] * pixel_size_nm
        ps.append(np.mean(d_nm_r <= CONTACT_THR_NM))
    rand_ps.append(float(np.mean(ps) if len(ps) else 0.0))

rand_ps = np.array(rand_ps)
p_value = float((np.sum(rand_ps >= real_p) + 1) / (len(rand_ps) + 1))


# =========================
# 8) 作图 + 总结
# =========================
plt.figure(figsize=(6.2, 4.4), dpi=160)
plt.hist(pts_df["d_nm"], bins=40)
plt.axvline(CONTACT_THR_NM, linestyle="--")
plt.xlabel("UCNP→ER nearest distance (nm)")
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
plt.xlabel("UCNP→ER nearest distance (nm)")
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
summary.append("ER mask segmentation (forced robust):")
summary.append(f"  CLAHE={ER_CLAHE}, kernel={ER_CLAHE_KERNEL}, clip={ER_CLAHE_CLIP}")
summary.append(f"  FRANGI={ER_USE_FRANGI}, scales={ER_FRANGI_SCALE_RANGE}, step={ER_FRANGI_SCALE_STEP}")
summary.append(f"  SAUVOLA window={ER_SAUVOLA_WINDOW}, k={ER_SAUVOLA_K}")
summary.append(f"  MIN_OBJ={ER_MIN_OBJ}, CLOSE_R={ER_CLOSE_RADIUS}, THIN_ITERS={ER_THIN_ITERS}")
summary.append(f"  FORCE_MIN_FG_FRAC={ER_FORCE_MIN_FG_FRAC}, TOP_PERCENT_FALLBACK={ER_TOP_PERCENT_FALLBACK}%")
summary.append("")
summary.append("UCNP mask segmentation:")
summary.append(f"  TOPHAT_RADIUS={UCNP_TOPHAT_RADIUS}, MASK_PCTL={UCNP_MASK_PCTL}, MIN_OBJ={UCNP_MIN_OBJ_PX}, CLOSE_R={UCNP_CLOSE_RADIUS}, FILL={UCNP_FILL_HOLES}")
summary.append("Object linking:")
summary.append(f"  MAX_LINK_DIST_PX={MAX_LINK_DIST_PX}, GAP={ALLOW_GAP_FRAMES}, MIN_TRACK_LEN={MIN_TRACK_LEN}")

if len(tm) > 0:
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

