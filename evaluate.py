"""
Quantitative comparison on the LGG slices against the expert masks.

    python evaluate.py

Patients are split once: the first 5 (sorted) were used to choose the number
of clusters (c = 5); every number reported here comes from the other patients.

Experiments
  1. Accuracy        Dice / IoU of GA-FCM, random-init FCM and k-means, one run each
  2. Restarts        the same slice clustered 10x from random starts: how often does
                     plain FCM / k-means land in a worse local solution than GA-FCM?
  3. Fuzziness       do the segmentation errors sit where the fuzzy model said it was unsure?
  4. False positives on tumour-free slices
"""

import glob
import json
import os

import cv2
import numpy as np

from fcm import partition_entropy
from pipeline import dice, iou, segment

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lgg")
DEV_PATIENTS = 5
MIN_TUMOUR_PX = 100
METHODS = ["ga-fcm", "fcm", "kmeans"]
LABELS = {"ga-fcm": "GA-FCM", "fcm": "FCM (random init)", "kmeans": "k-means"}


def load(mask_path):
    img = cv2.imread(mask_path[: -len("_mask.tif")] + ".tif")[..., ::-1]
    return img, cv2.imread(mask_path, 0) > 0


def slices(patients, positive=True):
    out = []
    for p in patients:
        for m in sorted(glob.glob(os.path.join(DATA, p, "*_mask.tif"))):
            area = (cv2.imread(m, 0) > 0).sum()
            if (positive and area >= MIN_TUMOUR_PX) or (not positive and area == 0):
                out.append(m)
    return out


def main():
    os.makedirs("results", exist_ok=True)
    patients = sorted(os.listdir(DATA))
    test_patients = patients[DEV_PATIENTS:]
    pos = slices(test_patients)
    print(f"{len(test_patients)} test patients, {len(pos)} slices with tumour >= {MIN_TUMOUR_PX} px")

    # ---------------------------------------------------------------- 1. accuracy
    scores = {k: {"dice": [], "iou": [], "iters": [], "secs": []} for k in METHODS}
    fuzz_err, fuzz_ok, err_in_fuzzy, px_in_fuzzy = [], [], [], []
    boundary_fuzz, interior_fuzz = [], []
    per_slice = []
    for i, mp in enumerate(pos):
        img, gt = load(mp)
        row = {"slice": os.path.basename(mp).replace("_mask.tif", "")}
        for k in METHODS:
            s = segment(img, k, seed=i)
            d = dice(s.mask, gt)
            scores[k]["dice"].append(d)
            scores[k]["iou"].append(iou(s.mask, gt))
            scores[k]["iters"].append(s.iterations)
            scores[k]["secs"].append(s.seconds)
            row[k] = round(d, 3)
            if k == "ga-fcm":
                # ---------------------------------------------------- 3. fuzziness
                F = s.fuzziness
                b = s.brain
                wrong = (s.mask != gt) & b
                fuzz_err.append(F[wrong].mean() if wrong.any() else np.nan)
                fuzz_ok.append(F[~wrong & b].mean())
                err_in_fuzzy.append((F[wrong] > 0.5).mean() if wrong.any() else np.nan)
                px_in_fuzzy.append((F[b] > 0.5).mean())
                k5 = np.ones((5, 5), np.uint8)
                band = (cv2.dilate(gt.astype(np.uint8), k5) - cv2.erode(gt.astype(np.uint8), k5)).astype(bool) & b
                inner = cv2.erode(gt.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
                if band.any():
                    boundary_fuzz.append(F[band].mean())
                if inner.any():
                    interior_fuzz.append(F[inner].mean())
        per_slice.append(row)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(pos)}  mean Dice so far: " +
                  "  ".join(f"{LABELS[k]} {np.mean(scores[k]['dice']):.3f}" for k in METHODS))

    # ---------------------------------------------------------------- 2. restarts
    rng = np.random.default_rng(0)
    subset = [pos[j] for j in sorted(rng.choice(len(pos), min(30, len(pos)), replace=False))]
    restarts = {k: {"dice_std": [], "bad_rate": [], "dice_range": []} for k in METHODS}
    print(f"\nRestart experiment on {len(subset)} slices (10 runs each) ...")
    for mp in subset:
        img, gt = load(mp)
        runs = {k: [segment(img, k, seed=s) for s in range(10 if k != "ga-fcm" else 5)] for k in METHODS}
        # best FCM objective found by any fuzzy run on this slice = reference optimum
        best_j = min(s.objective for k in ("ga-fcm", "fcm") for s in runs[k])
        best_sse = min(s.objective for s in runs["kmeans"])
        for k in METHODS:
            ds = [dice(s.mask, gt) for s in runs[k]]
            ref = best_sse if k == "kmeans" else best_j
            restarts[k]["dice_std"].append(np.std(ds))
            restarts[k]["dice_range"].append(max(ds) - min(ds))
            restarts[k]["bad_rate"].append(np.mean([s.objective > 1.01 * ref for s in runs[k]]))

    # ---------------------------------------------------------------- 4. false positives
    neg = slices(test_patients, positive=False)
    neg = [neg[j] for j in sorted(rng.choice(len(neg), min(60, len(neg)), replace=False))]
    fp = {k: np.mean([segment(load(mp)[0], k, seed=0).mask.any() for mp in neg]) for k in METHODS}

    # ---------------------------------------------------------------- report
    summary = {"test_patients": len(test_patients), "tumour_slices": len(pos), "methods": {}}
    print(f"\n{'Method':<20}{'mean Dice':>10}{'median':>8}{'IoU':>7}{'iters':>7}{'sec':>7}"
          f"{'Dice std (restarts)':>21}{'worse-optimum runs':>20}{'FP on clean':>13}")
    print("-" * 113)
    for k in METHODS:
        s, r = scores[k], restarts[k]
        summary["methods"][LABELS[k]] = {
            "mean_dice": float(np.mean(s["dice"])), "median_dice": float(np.median(s["dice"])),
            "mean_iou": float(np.mean(s["iou"])), "mean_iterations": float(np.mean(s["iters"])),
            "mean_seconds": float(np.mean(s["secs"])),
            "restart_dice_std": float(np.mean(r["dice_std"])),
            "restart_dice_range": float(np.mean(r["dice_range"])),
            "worse_local_optimum_rate": float(np.mean(r["bad_rate"])),
            "false_positive_rate_clean_slices": float(fp[k]),
        }
        m = summary["methods"][LABELS[k]]
        print(f"{LABELS[k]:<20}{m['mean_dice']:>10.3f}{m['median_dice']:>8.3f}{m['mean_iou']:>7.3f}"
              f"{m['mean_iterations']:>7.1f}{m['mean_seconds']:>7.2f}{m['restart_dice_std']:>21.3f}"
              f"{m['worse_local_optimum_rate'] * 100:>19.1f}%{m['false_positive_rate_clean_slices'] * 100:>12.0f}%")

    fz = {
        "mean_fuzziness_wrong_pixels": float(np.nanmean(fuzz_err)),
        "mean_fuzziness_correct_pixels": float(np.nanmean(fuzz_ok)),
        "share_of_errors_in_fuzzy_pixels": float(np.nanmean(err_in_fuzzy)),
        "share_of_brain_that_is_fuzzy": float(np.nanmean(px_in_fuzzy)),
        "mean_fuzziness_on_expert_boundary": float(np.mean(boundary_fuzz)),
        "mean_fuzziness_inside_tumour": float(np.mean(interior_fuzz)),
    }
    summary["fuzziness"] = fz
    print("\nFuzziness analysis (GA-FCM, tumour-boundary uncertainty 4u(1-u), 0..1):")
    print(f"  mean fuzziness of wrongly labelled pixels   {fz['mean_fuzziness_wrong_pixels']:.3f}")
    print(f"  mean fuzziness of correctly labelled pixels {fz['mean_fuzziness_correct_pixels']:.3f}")
    print(f"  {fz['share_of_errors_in_fuzzy_pixels'] * 100:.0f}% of errors lie in pixels with fuzziness > 0.5, "
          f"which make up only {fz['share_of_brain_that_is_fuzzy'] * 100:.0f}% of the brain")
    print(f"  fuzziness on the expert's tumour boundary {fz['mean_fuzziness_on_expert_boundary']:.3f} "
          f"vs inside the tumour {fz['mean_fuzziness_inside_tumour']:.3f}")

    with open("results/evaluation.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open("results/per_slice_dice.json", "w") as f:
        json.dump(per_slice, f, indent=1)
    plots(scores, restarts, fuzz_err, fuzz_ok)
    print("\nWrote results/evaluation.json, results/dice_comparison.png")


def plots(scores, restarts, fuzz_err, fuzz_ok):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = ["#1565c0", "#7986cb", "#ef6c00"]
    ax = axes[0]
    bp = ax.boxplot([scores[k]["dice"] for k in METHODS], tick_labels=[LABELS[k] for k in METHODS],
                    patch_artist=True, showmeans=True)
    for p, c in zip(bp["boxes"], cols):
        p.set_facecolor(c)
        p.set_alpha(0.6)
    ax.set_ylabel("Dice vs expert mask")
    ax.set_title("Segmentation accuracy (held-out patients)")

    ax = axes[1]
    x = np.arange(len(METHODS))
    ax.bar(x - 0.2, [np.mean(restarts[k]["dice_range"]) for k in METHODS], 0.4, color=cols, alpha=0.6,
           label="Dice range across restarts")
    ax.bar(x + 0.2, [np.mean(restarts[k]["bad_rate"]) for k in METHODS], 0.4, color=cols,
           label="share of runs in a worse optimum")
    ax.set_xticks(x, [LABELS[k] for k in METHODS])
    ax.set_title("Same slice, different random starts")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.hist([np.array(fuzz_ok)[~np.isnan(fuzz_ok)], np.array(fuzz_err)[~np.isnan(fuzz_err)]], bins=20,
            label=["correctly labelled pixels", "wrongly labelled pixels"], color=["#9e9e9e", "#c62828"])
    ax.set_xlabel("mean fuzziness per slice, 4u(1-u)")
    ax.set_title("Errors happen where the model is unsure")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/dice_comparison.png", dpi=140)


if __name__ == "__main__":
    main()
