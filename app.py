"""
Web demo: upload a brain MRI slice -> GA-initialised Fuzzy C-Means segmentation,
fuzziness heatmap, per-tissue membership maps, and a comparison with k-means.

    python app.py        (opens http://127.0.0.1:7860 in the browser)
"""

import glob
import os

import cv2
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fcm import fcm
from ga import GAConfig
from pipeline import (cluster_maps, dice, features, fuzziness_heatmap, membership_map, overlay, segment, to_flair,
                      brain_mask, upscale, _contour, _gray_rgb)

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples")

# Example slices from patients that were NOT used to tune the method.
EXAMPLE_SLICES = [
    "TCGA_FG_6691_20020405/TCGA_FG_6691_20020405_24",   # clear tumour: every method does well
    "TCGA_DU_7304_19930325/TCGA_DU_7304_19930325_19",   # k-means unstable: 0.86 or 0.00 depending on its start
    "TCGA_HT_7879_19981009/TCGA_HT_7879_19981009_15",   # FCM clearly better than k-means
    "TCGA_DU_7304_19930325/TCGA_DU_7304_19930325_20",   # heatmap lights up the part that was missed
    "TCGA_HT_7882_19970125/TCGA_HT_7882_19970125_25",
    "TCGA_DU_6405_19851005/TCGA_DU_6405_19851005_48",   # failure case: tumour not brighter than tissue
]


def ensure_examples():
    os.makedirs(EXAMPLES, exist_ok=True)
    out = []
    for name in EXAMPLE_SLICES:
        src = os.path.join(HERE, "data", "lgg", name + ".tif")
        base = os.path.basename(name)
        img_png = os.path.join(EXAMPLES, base + ".png")
        mask_png = os.path.join(EXAMPLES, base + "_mask.png")
        if not os.path.exists(img_png) and os.path.exists(src):
            cv2.imwrite(img_png, cv2.imread(src))
            cv2.imwrite(mask_png, cv2.imread(src.replace(".tif", "_mask.tif"), 0))
        if os.path.exists(img_png):
            out.append([img_png, mask_png])
    return out


def _objective_plot(img, c, m, ga_seg, seeds=6):
    """FCM objective vs iteration: GA-initialised run against several random starts."""
    gray = to_flair(img)
    X, _ = features(gray, brain_mask(gray))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    g = ga_seg.ga
    ax.plot(g.best_history, color="#1565c0", lw=2, label="best chromosome")
    ax.plot(g.mean_history, color="#90caf9", lw=1.5, label="population mean")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness  R_m (lower = better)")
    ax.set_title(f"Genetic algorithm ({g.evaluations} fitness evaluations)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    ax = axes[1]
    finals = []
    for s in range(seeds):
        _, _, hist, _ = fcm(X, c, m, seed=100 + s)
        ax.plot(hist, color="#ef6c00", alpha=0.5, lw=1, label="random start" if s == 0 else None)
        finals.append(hist[-1])
    _, _, hist, _ = fcm(X, c, m, init_centers=g.centers)
    ax.plot(hist, color="#1565c0", lw=2.5, label="GA start")
    ax.set_xlabel("FCM iteration")
    ax.set_ylabel("objective J_m")
    ax.set_yscale("log")
    worse = sum(f > 1.01 * hist[-1] for f in finals)
    ax.set_title(f"FCM convergence ({worse}/{seeds} random starts end worse)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def run(img, gt_img, c, m, generations, population, seed):
    if img is None:
        raise gr.Error("Upload an MRI slice or pick an example first.")
    gt = None
    if gt_img is not None:
        g = np.asarray(gt_img)
        gt = (g[..., 0] if g.ndim == 3 else g) > 127
        if gt.shape != img.shape[:2]:
            gt = cv2.resize(gt.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    cfg = GAConfig(pop_size=int(population), generations=int(generations))
    c, seed = int(c), int(seed)

    ga = segment(img, "ga-fcm", c=c, m=float(m), seed=seed, ga_cfg=cfg)
    rnd = segment(img, "fcm", c=c, m=float(m), seed=seed)
    km = segment(img, "kmeans", c=c, seed=seed)

    km_view = upscale(_gray_rgb(km.gray))
    km_fill = upscale(km.mask.astype(np.uint8)) > 0
    km_view[km_fill] = (0.4 * km_view[km_fill] + 0.6 * np.array([255, 40, 40])).astype(np.uint8)
    _contour(km_view, km_fill, (255, 230, 0), 2)
    if gt is not None:
        _contour(km_view, upscale(gt.astype(np.uint8)) > 0, (0, 255, 0), 1)

    F = ga.fuzziness[ga.brain]
    rows = [("GA-FCM", ga), ("FCM, random init", rnd), ("k-means (crisp)", km)]
    table = "| Method | Dice | Iterations | Time | Tumour area |\n|---|---|---|---|---|\n"
    for name, s in rows:
        d = f"{dice(s.mask, gt):.3f}" if gt is not None else "upload mask"
        table += f"| {name} | {d} | {s.iterations} | {s.seconds:.2f} s | {int(s.mask.sum())} px |\n"
    note = ga.info.get("note", "")
    summary = (
        f"{table}\n"
        f"**Fuzziness:** {np.mean(F > 0.5) * 100:.1f}% of brain pixels are on an uncertain tumour boundary "
        f"(tumour membership between 0.15 and 0.85); "
        f"partition coefficient {ga.info.get('partition_coefficient', 0):.3f} (1 = crisp, {1 / c:.2f} = totally fuzzy).  \n"
        f"**Tumour cluster centre:** {ga.centers[ga.tumour_cluster, 0]:.2f} "
        f"({ga.info['contrast']:.2f}x brighter than the next tissue class). {note}  \n"
        f"Yellow = crisp tumour boundary, green = expert mask (if uploaded)."
    )
    gallery = cluster_maps(ga)
    return (overlay(ga, gt), fuzziness_heatmap(ga, gt), membership_map(ga), km_view,
            gallery, _objective_plot(img, c, float(m), ga), summary)


def build():
    examples = ensure_examples()
    with gr.Blocks(title="GA-FCM Brain MRI Segmentation") as demo:
        gr.Markdown(
            "# Brain MRI tumour segmentation with GA-initialised Fuzzy C-Means\n"
            "Every pixel gets a **membership degree** for each tissue class instead of a hard label. "
            "A **genetic algorithm** picks the starting cluster centres so FCM doesn't settle in a poor "
            "local optimum. **k-means** (crisp) is the baseline.")
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(label="MRI slice (FLAIR, or the 3-channel LGG .tif/.png)", type="numpy", height=300)
                gt = gr.Image(label="Expert mask (optional, for Dice)", type="numpy", height=150)
                with gr.Accordion("Parameters", open=False):
                    c = gr.Slider(3, 7, value=5, step=1, label="clusters c")
                    m = gr.Slider(1.3, 3.0, value=2.0, step=0.1, label="fuzzifier m (1 = crisp)")
                    gens = gr.Slider(5, 100, value=40, step=5, label="GA generations")
                    pop = gr.Slider(10, 80, value=30, step=2, label="GA population")
                    seed = gr.Number(value=0, precision=0, label="random seed")
                btn = gr.Button("Segment", variant="primary")
                if examples:
                    gr.Examples(examples, inputs=[img, gt], label="Example slices (held-out patients)")
            with gr.Column(scale=2):
                with gr.Row():
                    out_overlay = gr.Image(label="GA-FCM: tumour membership (red) + boundary", height=360)
                    out_fuzzy = gr.Image(label="Fuzziness heatmap: 4u(1-u), bright = uncertain tumour boundary", height=360)
                with gr.Row():
                    out_member = gr.Image(label="Tumour membership degree 0..1", height=360)
                    out_km = gr.Image(label="k-means: crisp, no notion of uncertainty", height=360)
                out_md = gr.Markdown()
        out_plot = gr.Plot(label="Genetic algorithm and FCM convergence")
        out_gallery = gr.Gallery(label="Membership map of every tissue cluster", columns=5, height=260)
        btn.click(run, [img, gt, c, m, gens, pop, seed],
                  [out_overlay, out_fuzzy, out_member, out_km, out_gallery, out_plot, out_md])
    return demo


if __name__ == "__main__":
    build().launch(inbrowser=True)
