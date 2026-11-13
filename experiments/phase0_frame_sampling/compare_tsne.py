"""Side-by-side t-SNE of two encoders, raw and with per-video mean subtraction.

Rows = video sets, columns = (encoder x raw/centred). Each panel is coloured by video and titled
with the kNN same-video rate (lower = more mixing; chance shown in the row label).

Usage:
  python experiments/phase0_frame_sampling/compare_tsne.py data/processed/lemon_chole data/processed/lemon_sample \
      --encoders emb_lemonfm emb_resnet50 --out data/processed/compare_tsne.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from explore_embeddings import center_per_video, load, metrics

NAMES = {"emb_lemonfm": "LemonFM", "emb_resnet50": "ResNet-50"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--encoders", nargs="+", default=["emb_lemonfm", "emb_resnet50"])
    ap.add_argument("--out", default="data/processed/compare_tsne.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cols = [(e, c) for e in args.encoders for c in (False, True)]
    fig, axes = plt.subplots(len(args.sets), len(cols), figsize=(4.2 * len(cols), 4.4 * len(args.sets)),
                             squeeze=False)
    for r, s in enumerate(args.sets):
        for c, (enc, centred) in enumerate(cols):
            X, vids, ts = load(Path(s) / enc)
            X = normalize(X)
            if centred:
                X = normalize(center_per_video(X, vids))
            _, _, knn = metrics(X, vids, ts, 20, args.seed)
            Z = TSNE(2, perplexity=30, init="pca", random_state=args.seed).fit_transform(X)
            ax = axes[r, c]
            for v in sorted(set(vids)):
                m = vids == v
                ax.scatter(Z[m, 0], Z[m, 1], s=2, label=v)
            ax.set_title(f"{NAMES.get(enc, enc)}{' (per-video centred)' if centred else ''}\n"
                         f"kNN same-video {knn:.2f}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                chance = float((pd.Series(vids).value_counts(normalize=True) ** 2).sum())
                ax.set_ylabel(f"{Path(s).name}\n(chance {chance:.2f})", fontsize=10)
                ax.legend(markerscale=5, fontsize=6, loc="best")
            print(s, enc, "centred" if centred else "raw", f"knn={knn:.2f}")
    fig.suptitle("t-SNE coloured by video: LemonFM vs ResNet-50 (lower kNN same-video = more mixing)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print("saved", args.out)


if __name__ == "__main__":
    main()
