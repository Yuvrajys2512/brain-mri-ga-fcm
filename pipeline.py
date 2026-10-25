"""
Segmentation pipeline shared by the app and the evaluation.

    image -> FLAIR channel -> brain mask (simple skull strip) -> features per pixel
          -> clustering (GA-FCM / random-init FCM / k-means)
          -> tumour cluster = brightest centre (tumours are hyperintense on FLAIR)
          -> identical post-processing for every method -> crisp mask
"""

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from fcm import fcm, partition_coefficient, partition_entropy
from ga import GAConfig, ga_init_centers
from kmeans import kmeans


# ------------------------------------------------------------------ preprocessing
def to_flair(img):
    """Accepts a grayscale image, or the 3-channel LGG composite (pre-contrast,
    FLAIR, post-contrast) where FLAIR is the middle channel."""
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., :3]
        chans = img.astype(np.float32)
        if np.abs(chans[..., 0] - chans[..., 1]).mean() < 1 and np.abs(chans[..., 1] - chans[..., 2]).mean() < 1:
            img = img[..., 0]              # ordinary grayscale saved as RGB
        else:
            img = img[..., 1]              # LGG composite: FLAIR
    return img.astype(np.float32)


def _largest_component(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    return lab == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def _fill_holes(mask):
    m = mask.astype(np.uint8)
    h, w = m.shape
    flood = np.pad(m, 1).copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    holes = flood[1:-1, 1:-1] == 0
    return mask | holes


def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def brain_mask(gray):
    """Skull stripping from the radial intensity profile.

    Walking inwards from the edge of the head, an MRI slice goes
        bright scalp  ->  dark skull bone  ->  brain.
    We measure the mean intensity at each distance from the head's edge, find the
    dark bone band after the scalp peak, and keep everything deeper than it.
    """
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    g8 = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    t, _ = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    head = _fill_holes(_largest_component(g8 > 0.5 * t))
    scale = max(np.sqrt(head.sum() / np.pi) / 90.0, 0.3)       # ~1.0 for a typical 256 px slice

    dist = cv2.distanceTransform(head.astype(np.uint8), cv2.DIST_L2, 5)
    max_d = int(30 * scale)
    profile = np.array([g[(dist >= d) & (dist < d + 1)].mean() if ((dist >= d) & (dist < d + 1)).any() else 0
                        for d in range(max_d)])
    peak = int(np.argmax(profile[:max(2, int(12 * scale))]))                 # scalp
    valley = peak + int(np.argmin(profile[peak:peak + int(16 * scale) + 1]))  # skull bone
    brain = dist > valley + 1.5 * scale
    brain = cv2.morphologyEx(brain.astype(np.uint8), cv2.MORPH_OPEN, _disk(max(1, int(3 * scale)))).astype(bool)
    if not brain.any():
        return head
    return _fill_holes(_largest_component(brain))


def features(gray, mask):
    """Per-pixel features: normalised intensity and its 5x5 local mean (adds spatial context,
    which makes the clustering robust to single-pixel noise)."""
    ref = np.percentile(gray[mask], 99.5) + 1e-6
    I = np.clip(gray / ref, 0, 1)
    local = cv2.blur(I, (5, 5))
    return np.stack([I[mask], local[mask]], axis=1), I


# ------------------------------------------------------------------ result
@dataclass
class Segmentation:
    method: str
    gray: np.ndarray            # normalised FLAIR 0..1
    brain: np.ndarray           # bool mask
    labels: np.ndarray          # crisp cluster label per pixel (-1 outside brain)
    centers: np.ndarray
    tumour_cluster: int
    mask: np.ndarray            # final crisp tumour mask
    U: np.ndarray = None        # (c, H, W) memberships (fuzzy methods only)
    iterations: int = 0
    objective: float = 0.0
    seconds: float = 0.0
    ga: object = None
    info: dict = field(default_factory=dict)

    @property
    def tumour_membership(self):
        return None if self.U is None else self.U[self.tumour_cluster]

    @property
    def fuzziness(self):
        """Tumour-boundary uncertainty 4u(1-u) of the tumour membership u:
        0 where the pixel is clearly tumour (u=1) or clearly not (u=0), 1 where u=0.5."""
        if self.U is None:
            return None
        u = self.tumour_membership
        return (4.0 * u * (1.0 - u)) * self.brain

    @property
    def entropy(self):
        """Normalised entropy over all tissue classes (uncertainty between any tissues)."""
        if self.U is None:
            return None
        H = np.zeros(self.brain.shape, np.float32)
        H[self.brain] = partition_entropy(self.U[:, self.brain])
        return H


def segment(img, method="ga-fcm", c=5, m=2.0, seed=0, ga_cfg=None):
    t0 = time.time()
    gray = to_flair(img)
    brain = brain_mask(gray)
    X, I = features(gray, brain)
    ga_res, U_full = None, None

    if method == "kmeans":
        V, lab, obj, iters = kmeans(X, c, seed=seed)
    else:
        init = None
        if method == "ga-fcm":
            ga_res = ga_init_centers(X, c, m, ga_cfg or GAConfig(), seed=seed)
            init = ga_res.centers
        V, U, hist, iters = fcm(X, c, m, init_centers=init, seed=seed)
        lab = U.argmax(0)
        obj = hist[-1]
        U_full = np.zeros((c,) + gray.shape, np.float32)
        U_full[:, brain] = U

    labels = np.full(gray.shape, -1, np.int32)
    labels[brain] = lab
    tumour = int(np.argmax(V[:, 0]))
    mask, info = postprocess(labels == tumour, brain, V, tumour)
    if U_full is not None:
        info["partition_coefficient"] = partition_coefficient(U_full[:, brain])
    return Segmentation(method, I, brain, labels, V, tumour, mask, U_full, iters, obj,
                        time.time() - t0, ga_res, info)


def postprocess(raw, brain, V, tumour, min_contrast=1.12):
    """Identical for every method: remove noise, skull remnants and tiny blobs."""
    info = {}
    others = np.delete(V[:, 0], tumour)
    contrast = V[tumour, 0] / (others.max() + 1e-6)
    info["contrast"] = float(contrast)
    if contrast < min_contrast:        # no clearly hyperintense tissue class
        info["note"] = "no distinct hyperintense region"
        return np.zeros_like(raw), info
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_OPEN, k3)
    rim = brain & ~cv2.erode(brain.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = []
    for i in range(1, n):
        comp = lab == i
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 30 or (comp & rim).sum() > 0.3 * area:   # tiny, or mostly on the skull rim
            continue
        keep.append((area, i))
    if not keep:
        return np.zeros_like(raw), info
    biggest = max(a for a, _ in keep)
    out = np.isin(lab, [i for a, i in keep if a >= 0.25 * biggest])
    out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, k3).astype(bool)
    return _fill_holes(out) & brain, info


# ------------------------------------------------------------------ metrics
def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u


# ------------------------------------------------------------------ visualisation (RGB uint8)
def _gray_rgb(I):
    g = (np.clip(I, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


def _contour(img, mask, color, thick=1):
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnt, -1, color, thick, cv2.LINE_AA)


def _colormap(values, brain, cmap=cv2.COLORMAP_INFERNO):
    v = (np.clip(values, 0, 1) * 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(v, cmap), cv2.COLOR_BGR2RGB)
    rgb[~brain] = 0
    return rgb


def upscale(img, size=512):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST if img.dtype == bool else cv2.INTER_LINEAR)


def overlay(seg, gt=None):
    """Tumour membership as a red tint (fuzzy) or solid fill (crisp), crisp boundary in yellow,
    expert boundary in green."""
    base = _gray_rgb(seg.gray).astype(np.float32)
    alpha = seg.tumour_membership if seg.U is not None else seg.mask.astype(np.float32) * 0.6
    if seg.U is not None:
        alpha = alpha * 0.75 * seg.brain
    red = np.array([255, 40, 40], np.float32)
    out = base * (1 - alpha[..., None]) + red * alpha[..., None]
    out = upscale(out.astype(np.uint8))
    _contour(out, upscale(seg.mask.astype(np.uint8)) > 0, (255, 230, 0), 2)
    if gt is not None:
        _contour(out, upscale(gt.astype(np.uint8)) > 0, (0, 255, 0), 1)
    return out


def fuzziness_heatmap(seg, gt=None):
    out = upscale(_colormap(seg.fuzziness, seg.brain))
    _contour(out, upscale(seg.mask.astype(np.uint8)) > 0, (255, 255, 255), 1)
    if gt is not None:
        _contour(out, upscale(gt.astype(np.uint8)) > 0, (0, 255, 0), 1)
    return out


def membership_map(seg):
    return upscale(_colormap(seg.tumour_membership, seg.brain, cv2.COLORMAP_JET))


def cluster_maps(seg):
    """One membership image per cluster, labelled with its centre intensity."""
    tiles = []
    order = np.argsort(seg.centers[:, 0])
    for rank, i in enumerate(order):
        t = _colormap(seg.U[i], seg.brain, cv2.COLORMAP_VIRIDIS)
        t = cv2.resize(t, (256, 256))
        name = "TUMOUR" if i == seg.tumour_cluster else f"tissue {rank + 1}"
        cv2.putText(t, f"{name}  v={seg.centers[i, 0]:.2f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append((t, name))
    return tiles
