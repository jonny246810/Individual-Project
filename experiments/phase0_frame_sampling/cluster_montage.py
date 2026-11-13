"""Cluster check B by eye: show real frames from each k-means cluster.

Clusters the embeddings exactly as explore_embeddings.py does (L2-normalise, optional per-video
centring, k-means, seed 0), then draws one row per cluster with N randomly chosen frames.
Each thumbnail has a coloured strip along the bottom showing its video (same colours as the t-SNE),
so you can see at a glance whether a cluster mixes videos and whether its frames share content.

Frames are decoded once at 1 fps into a small cache (thumbs.npz in the set folder).

Usage:
  python experiments/phase0_frame_sampling/cluster_montage.py data/raw/lemon_chole data/processed/lemon_chole/emb_resnet50 --center
"""
import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from explore_embeddings import center_per_video, load

TW, TH = 192, 108  # thumbnail size


def decode_thumbs(video_dir, cache):
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    thumbs = {}
    for p in sorted(Path(video_dir).glob("*.mp4")):
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(p), "-vf", f"fps=1,scale={TW}:{TH}",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
        thumbs[p.stem] = np.frombuffer(raw, np.uint8).reshape(-1, TH, TW, 3)
    np.savez(cache, **thumbs)
    return thumbs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("emb_dir")
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n", type=int, default=10, help="frames shown per cluster")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    emb_dir = Path(args.emb_dir)

    X, vids, ts = load(emb_dir)
    X = normalize(X)
    if args.center:
        X = normalize(center_per_video(X, vids))
    lab = KMeans(args.k, n_init=10, random_state=args.seed).fit_predict(X)

    thumbs = decode_thumbs(args.video_dir, emb_dir.parent / "thumbs.npz")
    videos = sorted(set(vids))
    colours = {v: np.array(plt.cm.tab10(i)[:3]) * 255 for i, v in enumerate(videos)}
    rng = np.random.default_rng(args.seed)

    fig, axes = plt.subplots(args.k, 1, figsize=(args.n * 1.9, args.k * 1.25))
    for c in range(args.k):
        idx = np.flatnonzero(lab == c)
        pick = np.sort(rng.choice(idx, min(args.n, len(idx)), replace=False))
        tiles = []
        for i in pick:
            im = thumbs[vids[i]][min(int(ts[i]), len(thumbs[vids[i]]) - 1)].copy()
            im[-10:] = colours[vids[i]]  # video colour strip
            tiles.append(np.pad(im, ((0, 0), (0, 4), (0, 0)), constant_values=255))
        row = np.concatenate(tiles, axis=1)
        counts = pd.Series(vids[idx]).value_counts()
        ax = axes[c]
        ax.imshow(row)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_ylabel(f"c{c}  n={len(idx)}\n{len(counts)} videos\ntop {counts.iloc[0] / len(idx):.0%}",
                      rotation=0, ha="right", va="center", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[v] / 255) for v in videos]
    name = f"{emb_dir.name}{'_centred' if args.center else ''}"
    fig.suptitle(f"{emb_dir.parent.name} / {name}: k={args.k}, {args.n} random frames per cluster",
                 y=0.998, fontsize=10)
    fig.legend(handles, videos, loc="upper center", bbox_to_anchor=(0.5, 0.988), ncol=len(videos),
               fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = emb_dir / f"montage_{name}_k{args.k}.png"
    fig.savefig(out, dpi=90)
    print("saved", out)


if __name__ == "__main__":
    main()
