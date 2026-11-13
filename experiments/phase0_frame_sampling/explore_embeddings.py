"""Do the embeddings cluster by content, or just by video? (cluster check A)

Loads per-video .npz files (from embed_frames.py) and, for each variant
(full dims, then PCA to each of --dims, optionally also with per-video mean subtraction):
  - k-means (k = --k) purity: share of each cluster from its top video (1.0 = cluster is one video)
  - kNN same-video rate: share of each frame's 10 nearest neighbours from the same video
    (ignoring same-video frames within +-30 s, which are near-duplicates), with the chance level
    if neighbours were picked at random
  - a t-SNE panel coloured by video
Low purity / kNN rate near chance = mixing across videos (what we want).

Usage:
  python experiments/phase0_frame_sampling/explore_embeddings.py data/processed/lemon_chole/emb_lemonfm --dims 512 256 128 64 --center
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

KNN = 10
WINDOW = 30  # seconds


def load(emb_dir):
    X, vids, ts = [], [], []
    for f in sorted(Path(emb_dir).glob("*.npz")):
        d = np.load(f)
        X.append(d["emb"].astype(np.float32))
        vids += [f.stem] * len(d["t"])
        ts.append(d["t"])
    return np.concatenate(X), np.array(vids), np.concatenate(ts)


def center_per_video(X, vids):
    X = X.copy()
    for v in np.unique(vids):
        X[vids == v] -= X[vids == v].mean(0)
    return X


def metrics(X, vids, ts, k, seed):
    lab = KMeans(k, n_init=10, random_state=seed).fit_predict(X)
    ct = pd.crosstab(lab, vids)
    purity = ct.max(axis=1) / ct.sum(axis=1)
    # Neighbouring seconds of the same video are near-duplicates and would always win, so
    # ignore same-video neighbours within +-WINDOW s and keep the first KNN that remain.
    nn = NearestNeighbors(n_neighbors=KNN + 2 * WINDOW + 1).fit(X).kneighbors(
        X, return_distance=False)[:, 1:]
    same = vids[nn] == vids[:, None]
    near = same & (np.abs(ts[nn] - ts[:, None]) <= WINDOW)
    keep = np.cumsum(~near, axis=1) <= KNN
    keep &= ~near
    knn_same = (same & keep).sum() / keep.sum()
    return purity.mean(), int((purity > 0.9).sum()), knn_same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emb_dir")
    ap.add_argument("--dims", type=int, nargs="*", default=[512, 256, 128, 64])
    ap.add_argument("--center", action="store_true", help="also try per-video mean subtraction")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.emb_dir)

    X0, vids, ts = load(args.emb_dir)
    X0 = normalize(X0)  # L2-normalise, so Euclidean distance tracks cosine distance
    videos = sorted(set(vids))
    p = pd.Series(vids).value_counts(normalize=True)
    chance = float((p ** 2).sum())
    print(f"{len(X0)} frames, {X0.shape[1]}-d, {len(videos)} videos; "
          f"kNN same-video chance level = {chance:.2f}")

    cum = np.cumsum(PCA().fit(X0).explained_variance_ratio_)
    variants = []  # (name, X)
    for centred in ([False, True] if args.center else [False]):
        Xb = normalize(center_per_video(X0, vids)) if centred else X0
        tag = " centred" if centred else ""
        variants.append((f"{Xb.shape[1]}-d (full){tag}", Xb))
        for d in args.dims:
            Xd = PCA(d, random_state=args.seed).fit_transform(Xb)
            variants.append((f"PCA {d}{tag}", Xd))

    rows = []
    for name, X in variants:
        pur, n90, knn = metrics(X, vids, ts, args.k, args.seed)
        var = f"{cum[X.shape[1] - 1]:.1%}" if "PCA" in name and "centred" not in name else ""
        rows.append({"variant": name, "var_kept": var, "kmeans_purity": round(pur, 2),
                     f"clusters_>90%_one_video (of {args.k})": n90,
                     "knn_same_video": round(knn, 2), "chance": round(chance, 2)})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(out / "dims_comparison.csv", index=False)

    n = len(variants)
    cols = len(args.dims) + 1
    fig, axes = plt.subplots(n // cols, cols, figsize=(4 * cols, 4 * (n // cols)), squeeze=False)
    for ax, (name, X), r in zip(axes.flat, variants, rows):
        Z = TSNE(2, perplexity=30, init="pca", random_state=args.seed).fit_transform(X)
        for v in videos:
            m = vids == v
            ax.scatter(Z[m, 0], Z[m, 1], s=2, label=v)
        ax.set_title(f"{name}\npurity {r['kmeans_purity']}, kNN same-video {r['knn_same_video']}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes.flat[0].legend(markerscale=5, fontsize=6, loc="best")
    fig.suptitle(f"t-SNE coloured by video ({out.parent.name}); kNN chance = {chance:.2f}")
    fig.tight_layout()
    fig.savefig(out / "tsne_dims.png", dpi=110)
    print("saved", out / "tsne_dims.png")


if __name__ == "__main__":
    main()
