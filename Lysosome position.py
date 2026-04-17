# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 18:12:06 2026

@author: sophi
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tifffile import imread, imwrite
from skimage import morphology, measure, exposure
from skimage.feature import blob_log   # ✅ NEW: LoG blob detection
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


# =========================================================
# Lyso(561)–UCNP positional analysis (cell-mask constrained)
# + Robust load_series(): auto-harmonize frame shapes (crop/pad)
# + Only segment/track/stat inside cell ROI
# Outputs:
#   - shape_report.csv
#   - cell_mask.tif / cell_roi_dilated.tif
#   - lyso_mask_*.tif / lyso_mask_stack.tif
#   - ucnp_mask_*.tif / ucnp_mask_stack.tif
#   - ucnp_label_stack.tif
#   - points_distances.csv / track_metrics.csv / summary.txt
#   - debug overlays
# =========================================================


# =========================
# 0) 参数区（只改这里）
# =========================
img_dir = r"E:\Multi-SIM\20251124-SF-474_aizde lyso\10h\10_1_001_20251124_151514+++"

single_multich_tif = None  # 若用单个多通道tif (T,C,H,W)，填路径；否则None

# ✅ Lyso=561；UCNP=980 (two separate tif sequences)
lyso_glob = os.path.join(img_dir, "*561*.tif")
ucnp_glob = os.path.join(img_dir, "*980*.tif")

# 仅在 single_multich_tif 模式用（按你自己的通道顺序改）
LYSO_CH = 0
UCNP_CH = 1

pixel_size_nm = 30.0
frame_interval_s = 5.4
CONTACT_THR_NM = 100.0

FRAME_COL = "frame"
X_COL = "x"
Y_COL = "y"

SAVE_DEBUG_OVERLAY = True
DEBUG_FRAME = 0

cell_mask_path = None      # 可选：外部cell mask路径（2D或3D），否则自动生成
N_RANDOM = 200

# ✅ Cell ROI dilation (avoid cutting cell edge)
CELL_DILATE_RADIUS = 3     # 0~6


# =========================================================
# ✅ Lyso(561) mask: LoG blob → draw disks → dilate 1–2 px
#    (replaces the old Frangi/Sauvola logic)
# =========================================================
LYSO_BG_SUBTRACT = True
LYSO_BG_RADIUS = 18          # 12~30：越大扣背景越强
LYSO_GAUSS_SIGMA = 1.0       # 0.5~1.5：轻微去噪

# LoG blob detection (pixel unit)
LYSO_BLOB_MIN_SIGMA = 1.0
LYSO_BLOB_MAX_SIGMA = 3.2    # 大团块可到 4~5
LYSO_BLOB_NUM_SIGMA = 10
LYSO_BLOB_THRESHOLD = 0.06   # 0.03~0.12：越小检出越多
LYSO_BLOB_OVERLAP = 0.5

# mask generation
LYSO_DRAW_RADIUS_FACTOR = 1.0   # radius = factor * sigma*sqrt(2)
LYSO_DILATE_PX = 1              # ✅ 你要的膨胀 1–2 px
LYSO_MIN_OBJ = 5                # 去小噪点
LYSO_MAX_BLOBS_PER_FRAME = 4000 # 防止异常爆炸

# 保存 Lyso masks
SAVE_LYSO_MASK_STACK = True
SAVE_LYSO_MASK_PER_FRAME = True

# UCNP mask 保存
SAVE_UCNP_MASK_STACK = True
SAVE_UCNP_MASK_PER_FRAME = True
SAVE_UCNP_LABEL_STACK = True
SAVE_UCNP_LABEL_PER_FRAME = False

# UCNP 分割参数
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

out_dir = os.path.join(img_dir, "_Lyso_UCNP_positional")
os.makedirs(out_dir, exist_ok=True)


# =========================
# 1) 通用：robust normalize
# =========================
def robust_norm(im, p_lo=1, p_hi=99):
    im = im.astype(np.float32)
    lo, hi = np.percentile(im, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(im, dtype=np.float32)
    out = (im - lo) / (hi - lo)
    return np.clip(out, 0, 1).astype(np.float32)


# =========================
# 2) Robust loader: unify shapes + report
# =========================
def _to_2d(img, name=""):
    """
    Ensure image is 2D.
    - If 2D: return as is
    - If 3D: max projection along axis 0 (works for Z-stack or multi-page)
    """
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return img.max(axis=0)
    raise ValueError(f"{name} unsupported ndim={img.ndim}, shape={img.shape}")

def _center_crop_or_pad(im, target_hw):
    """Center-crop or zero-pad image to target_hw=(H,W)."""
    th, tw = target_hw
    h, w = im.shape

    # center crop
    if h > th:
        y0 = (h - th) // 2
        im = im[y0:y0+th, :]
    if w > tw:
        x0 = (w - tw) // 2
        im = im[:, x0:x0+tw]

    # center pad
    h2, w2 = im.shape
    if h2 < th or w2 < tw:
        out = np.zeros((th, tw), dtype=im.dtype)
        y0 = (th - h2) // 2
        x0 = (tw - w2) // 2
        out[y0:y0+h2, x0:x0+w2] = im
        im = out
    return im

def load_series():
    """
    Supports:
    - single_multich_tif: 4D (T,C,H,W)
    - separate 561/980 sequences: each frame can be 2D or 3D (auto max-proj)
    Then harmonize all frames to same (H,W) by center crop/pad to median size.
    """
    if single_multich_tif is not None:
        arr = imread(single_multich_tif)
        if arr.ndim != 4:
            raise ValueError(f"single_multich_tif 需要 4D (T,C,H,W). 当前: {arr.shape}")
        lyso = arr[:, LYSO_CH, :, :].astype(np.float32)
        ucnp = arr[:, UCNP_CH, :, :].astype(np.float32)
        return lyso, ucnp

    lyso_files = sorted(glob.glob(lyso_glob))
    u_files    = sorted(glob.glob(ucnp_glob))

    if len(lyso_files) == 0 or len(u_files) == 0:
        raise FileNotFoundError("没有找到Lyso(561)或UCNP(980)序列文件。请检查 lyso_glob / ucnp_glob。")
    if len(lyso_files) != len(u_files):
        raise ValueError(f"Lyso文件数({len(lyso_files)})与UCNP文件数({len(u_files)})不一致。")

    lyso_raw, u_raw = [], []
    rep_rows = []

    for i, (lf, uf) in enumerate(zip(lyso_files, u_files)):
        lyso_img = _to_2d(imread(lf), name=f"Lyso[{i}]")
        u_img    = _to_2d(imread(uf), name=f"UCNP[{i}]")
        lyso_raw.append(lyso_img)
        u_raw.append(u_img)

        rep_rows.append({
            "i": i,
            "lyso_file": os.path.basename(lf),
            "ucnp_file": os.path.basename(uf),
            "lyso_shape": str(lyso_img.shape),
            "ucnp_shape": str(u_img.shape),
        })

    lyso_shapes = np.array([im.shape for im in lyso_raw], dtype=int)
    u_shapes    = np.array([im.shape for im in u_raw], dtype=int)
    all_shapes  = np.vstack([lyso_shapes, u_shapes])

    target_h = int(np.median(all_shapes[:, 0]))
    target_w = int(np.median(all_shapes[:, 1]))
    target_hw = (target_h, target_w)

    lyso_stack = [_center_crop_or_pad(im, target_hw) for im in lyso_raw]
    u_stack    = [_center_crop_or_pad(im, target_hw) for im in u_raw]

    rep = pd.DataFrame(rep_rows)
    rep["target_hw"] = str(target_hw)
    rep.to_csv(os.path.join(out_dir, "shape_report.csv"), index=False)
    print(f"[load_series] target_hw={target_hw}. Saved shape_report.csv")

    lyso = np.stack(lyso_stack, axis=0).astype(np.float32)
    ucnp = np.stack(u_stack, axis=0).astype(np.float32)
    return lyso, ucnp


# =========================
# 3) Load Lyso + UCNP
# =========================
LYSO, UCNP = load_series()
T, H, W = LYSO.shape
print(f"Loaded Lyso/UCNP stacks: T={T}, H={H}, W={W}")
print(f"Lyso raw min/max:  {LYSO.min():.1f} / {LYSO.max():.1f}")
print(f"UCNP raw min/max:  {UCNP.min():.1f} / {UCNP.max():.1f}")


# =========================
# 4) Build cell_mask (ROI for ALL stats)
# =========================
def build_cell_mask_fallback():
    lyso_max = LYSO.max(axis=0)
    u_max    = UCNP.max(axis=0)

    lyso_n = robust_norm(lyso_max, 1, 99)
    u_n    = robust_norm(u_max, 1, 99)

    m1 = lyso_n > np.percentile(lyso_n, 70)
    m2 = u_n    > np.percentile(u_n, 85)

    m = m1 | m2
    m = morphology.remove_small_objects(m, 500)
    m = morphology.binary_closing(m, morphology.disk(7))
    m = ndi.binary_fill_holes(m)
    return m.astype(bool)

def _load_external_mask(path):
    m = imread(path)
    if m.ndim == 3:
        m = m.any(axis=0)
    m = (m > 0)
    if m.shape != (H, W):
        m = _center_crop_or_pad(m.astype(np.uint8), (H, W)).astype(bool)
    return m.astype(bool)

if cell_mask_path and os.path.exists(cell_mask_path):
    cell_mask = _load_external_mask(cell_mask_path)
else:
    cell_mask = build_cell_mask_fallback()

cell_roi = cell_mask.copy()
if CELL_DILATE_RADIUS and CELL_DILATE_RADIUS > 0:
    cell_roi = morphology.binary_dilation(cell_roi, morphology.disk(int(CELL_DILATE_RADIUS)))

imwrite(os.path.join(out_dir, "cell_mask.tif"), cell_mask.astype(np.uint8))
imwrite(os.path.join(out_dir, "cell_roi_dilated.tif"), cell_roi.astype(np.uint8))
print(f"cell_mask frac={cell_mask.mean():.3f}, cell_roi frac={cell_roi.mean():.3f}")


# =========================
# 5) Lyso segmentation (LoG blobs) (AND cell_roi) + distance transform
# =========================
def lyso_mask_from_frame(img_lyso):
    """
    Lysosome (561) vesicle mask:
    1) denoise + background subtraction
    2) normalize
    3) LoG blob detection inside cell_roi
    4) draw disks for blobs + dilate 1–2 px
    5) cleanup and restrict to cell_roi
    """
    im = img_lyso.astype(np.float32)

    # (1) light denoise
    if LYSO_GAUSS_SIGMA and LYSO_GAUSS_SIGMA > 0:
        im = ndi.gaussian_filter(im, sigma=float(LYSO_GAUSS_SIGMA))

    # (2) background subtraction (rolling-ball like)
    if LYSO_BG_SUBTRACT and LYSO_BG_RADIUS and LYSO_BG_RADIUS > 0:
        se = morphology.disk(int(LYSO_BG_RADIUS))
        bg = morphology.opening(im, se)
        im = im - bg
        im = np.clip(im, 0, None)

    # normalize AFTER bg subtraction
    imn = robust_norm(im, 1, 99.8)
    if np.max(imn) <= 0:
        return np.zeros_like(imn, dtype=bool)

    # (3) blob detection only inside ROI
    im_roi = imn.copy()
    im_roi[~cell_roi] = 0.0

    blobs = blob_log(
        im_roi,
        min_sigma=float(LYSO_BLOB_MIN_SIGMA),
        max_sigma=float(LYSO_BLOB_MAX_SIGMA),
        num_sigma=int(LYSO_BLOB_NUM_SIGMA),
        threshold=float(LYSO_BLOB_THRESHOLD),
        overlap=float(LYSO_BLOB_OVERLAP),
        exclude_border=False,
    )

    if blobs is None or len(blobs) == 0:
        return np.zeros_like(imn, dtype=bool)

    # safety cap
    if len(blobs) > int(LYSO_MAX_BLOBS_PER_FRAME):
        ys = np.clip(blobs[:, 0].astype(int), 0, H-1)
        xs = np.clip(blobs[:, 1].astype(int), 0, W-1)
        score = imn[ys, xs] * (blobs[:, 2] + 1e-6)  # intensity * sigma
        keep = np.argsort(score)[-int(LYSO_MAX_BLOBS_PER_FRAME):]
        blobs = blobs[keep]

    # (4) draw disks
    m = np.zeros((H, W), dtype=bool)
    for (y, x, sigma) in blobs:
        iy, ix = int(round(y)), int(round(x))
        if iy < 0 or iy >= H or ix < 0 or ix >= W:
            continue
        if not cell_roi[iy, ix]:
            continue

        r = float(LYSO_DRAW_RADIUS_FACTOR) * float(sigma) * np.sqrt(2.0)
        r = max(1.0, r)
        rr = int(np.ceil(r))

        y0 = max(0, iy - rr)
        y1 = min(H, iy + rr + 1)
        x0 = max(0, ix - rr)
        x1 = min(W, ix + rr + 1)

        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (yy - iy) ** 2 + (xx - ix) ** 2 <= r ** 2
        m[y0:y1, x0:x1] |= disk

    # (5) dilate 1–2 px to form vesicle mask
    if LYSO_DILATE_PX and LYSO_DILATE_PX > 0:
        m = morphology.binary_dilation(m, morphology.disk(int(LYSO_DILATE_PX)))

    # cleanup + restrict to ROI
    m = m & cell_roi
    m = morphology.remove_small_objects(m, int(LYSO_MIN_OBJ))

    return m.astype(bool)

lyso_masks = np.zeros((T, H, W), dtype=bool)
dist_to_lyso_px = np.zeros((T, H, W), dtype=np.float32)

print("Building Lyso masks + distance transforms ...")
for t in range(T):
    m = lyso_mask_from_frame(LYSO[t])
    lyso_masks[t] = m
    dist_to_lyso_px[t] = ndi.distance_transform_edt(~m).astype(np.float32)

print(f"Lyso mask foreground fraction (median across frames): {np.median(lyso_masks.mean(axis=(1,2))):.6f}")

print("Saving Lyso masks ...")
lyso_masks_u8 = lyso_masks.astype(np.uint8)
if SAVE_LYSO_MASK_PER_FRAME:
    for t in range(T):
        imwrite(os.path.join(out_dir, f"lyso_mask_t{t:03d}.tif"), lyso_masks_u8[t])
if SAVE_LYSO_MASK_STACK:
    imwrite(os.path.join(out_dir, "lyso_mask_stack.tif"), lyso_masks_u8, imagej=True)
print("Lyso masks saved.")


# =========================
# 6) UCNP segmentation (AND cell_roi) + labels + objects list
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
    m = (im >= thr)

    # ✅ restrict UCNP mask to cell ROI
    m = m & cell_roi

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
            iy, ix = int(round(cy)), int(round(cx))
            iy = int(np.clip(iy, 0, H-1))
            ix = int(np.clip(ix, 0, W-1))

            if not cell_roi[iy, ix]:
                continue

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
print(f"Total UCNP objects (frame-wise components, in cell ROI) = {len(obj_df)}")
if len(obj_df) == 0:
    raise RuntimeError("UCNPmask 分割后（且限制cell内）没有任何对象。请降低 UCNP_MASK_PCTL 或减小 UCNP_MIN_OBJ_PX。")


# =========================
# 7) Link objects across frames -> mask_id
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
# 8) Debug overlays
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
    plt.title(f"UCNP frame {t0}: mask boundary overlay (cell ROI only)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"debug_overlay_UCNPmask_frame{t0:03d}.png"))
    plt.close()

    plt.figure(figsize=(6, 6), dpi=220)
    plt.imshow(LYSO[t0], cmap="gray")
    m = lyso_masks[t0]
    if np.any(m):
        b = m ^ morphology.binary_erosion(m, morphology.disk(1))
        yy, xx = np.where(b)
        plt.scatter(xx, yy, s=1, c="cyan")
    plt.title(f"Lyso frame {t0}: Lyso-mask boundary overlay (cell ROI only)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"debug_overlay_LysoMask_frame{t0:03d}.png"))
    plt.close()

    plt.figure(figsize=(6, 6), dpi=220)
    plt.imshow(cell_roi, cmap="gray")
    plt.title("cell ROI (dilated)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "debug_overlay_cell_roi.png"))
    plt.close()

    print(f"Saved debug overlays at frame {t0}")


# =========================
# 9) Distance sampling (STRICTLY inside cell_roi)
# =========================
rows = []
for _, r in obj_df.iterrows():
    t = int(r[FRAME_COL])
    flab = int(r["frame_label"])
    mid = int(r["mask_id"])

    lab = ucnp_label_u16[t]
    m_obj = (lab == flab)
    if not np.any(m_obj):
        continue

    m_use = m_obj & cell_roi
    if not np.any(m_use):
        continue

    dmin_px = float(np.min(dist_to_lyso_px[t][m_use]))
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
    raise RuntimeError("pts_df 为空：限制 cell_roi 后对象距离采样失败。请检查 cell_mask 是否过小 / UCNP阈值是否过高。")

pts_df.to_csv(os.path.join(out_dir, "points_distances.csv"), index=False)
print(f"Distance sampling done (cell ROI only). object-frame rows={len(pts_df)}")


# =========================
# 10) Track-level metrics
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
print(f"Saved track_metrics.csv (cell ROI only) (n_masks={tm.shape[0]})")


# =========================
# 11) Random control (sample within cell_mask)
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
        d_nm_r = dist_to_lyso_px[t, ry, rx] * pixel_size_nm
        ps.append(np.mean(d_nm_r <= CONTACT_THR_NM))
    rand_ps.append(float(np.mean(ps) if len(ps) else 0.0))

rand_ps = np.array(rand_ps)
p_value = float((np.sum(rand_ps >= real_p) + 1) / (len(rand_ps) + 1))


# =========================
# 12) Plots + summary
# =========================
plt.figure(figsize=(6.2, 4.4), dpi=160)
plt.hist(pts_df["d_nm"], bins=40)
plt.axvline(CONTACT_THR_NM, linestyle="--")
plt.xlabel("UCNP→Lyso nearest distance (nm)")
plt.ylabel("Count")
plt.title("Distance distribution (cell ROI only)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "distance_hist.png"))
plt.close()

d = np.sort(pts_df["d_nm"].to_numpy())
cdf = np.arange(1, len(d) + 1) / len(d)
plt.figure(figsize=(6.2, 4.4), dpi=160)
plt.plot(d, cdf)
plt.axvline(CONTACT_THR_NM, linestyle="--")
plt.xlabel("UCNP→Lyso nearest distance (nm)")
plt.ylabel("CDF")
plt.title("Distance CDF (cell ROI only)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "distance_cdf.png"))
plt.close()

plt.figure(figsize=(6.2, 4.4), dpi=160)
plt.hist(rand_ps, bins=25)
plt.axvline(real_p, linestyle="--")
plt.xlabel(f"Random P(d ≤ {CONTACT_THR_NM:.0f} nm) (cell_mask)")
plt.ylabel("Count")
plt.title(f"Randomization test (p={p_value:.3g})")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "random_test.png"))
plt.close()

summary = []
summary.append(f"T={T}, pixel_size_nm={pixel_size_nm}, dt_s={frame_interval_s}")
summary.append(f"CONTACT_THR_NM={CONTACT_THR_NM}")
summary.append(f"cell_mask frac={cell_mask.mean():.4f}; cell_roi(dilated) frac={cell_roi.mean():.4f}")
summary.append(f"Total object-frame rows (cell ROI only)={len(pts_df)}")
summary.append(f"Real P(d≤thr)={real_p:.4f}")
summary.append(f"Random mean P(d≤thr)={rand_ps.mean():.4f} ± {rand_ps.std():.4f}")
summary.append(f"One-sided p (real > random)={p_value:.4g}")
summary.append("")
summary.append("Lyso (LoG blob) params:")
summary.append(f"  BG_SUBTRACT={LYSO_BG_SUBTRACT}, BG_RADIUS={LYSO_BG_RADIUS}, GAUSS_SIGMA={LYSO_GAUSS_SIGMA}")
summary.append(f"  min_sigma={LYSO_BLOB_MIN_SIGMA}, max_sigma={LYSO_BLOB_MAX_SIGMA}, num_sigma={LYSO_BLOB_NUM_SIGMA}")
summary.append(f"  threshold={LYSO_BLOB_THRESHOLD}, overlap={LYSO_BLOB_OVERLAP}")
summary.append(f"  draw_factor={LYSO_DRAW_RADIUS_FACTOR}, dilate_px={LYSO_DILATE_PX}, min_obj={LYSO_MIN_OBJ}")
summary.append(f"  max_blobs_per_frame={LYSO_MAX_BLOBS_PER_FRAME}")
summary.append("")
summary.append("UCNP params:")
summary.append(f"  TOPHAT_RADIUS={UCNP_TOPHAT_RADIUS}, MASK_PCTL={UCNP_MASK_PCTL}, MIN_OBJ={UCNP_MIN_OBJ_PX}, CLOSE_R={UCNP_CLOSE_RADIUS}, FILL={UCNP_FILL_HOLES}")
summary.append("Tracking:")
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
