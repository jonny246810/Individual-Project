"""Step 3: choose which videos to use, N in total, split 70/15/15, with a cap per channel.

Walks each split's videos in their seeded `rank` order (from make_splits.py) and keeps a video unless
its channel already has --cap videos in the selection. Because the order is fixed, a larger N
(same cap) always contains the smaller selection, so experiments can scale up without reshuffling.

Output: data/meta/selected_videos_n{N}_cap{cap}.csv. Prints channel and procedure spread.

Usage: python scripts/select_videos.py --n 100 --cap 5
"""
import argparse
from pathlib import Path

import pandas as pd

SPLITS = Path("data/meta/splits.csv")
SHARES = {"train": 0.70, "val": 0.15, "test": 0.15}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--cap", type=int, default=5, help="max videos per channel within each split")
    args = ap.parse_args()

    df = pd.read_csv(SPLITS, keep_default_na=False)
    # A video with no known channel is its own group.
    df["group"] = df.channel_url.where(df.channel_url != "", "unknown:" + df.youtube_id)

    picked = []
    for split, share in SHARES.items():
        want = round(args.n * share)
        per_group = {}
        for _, r in df[df.split == split].sort_values("rank").iterrows():
            if len(picked) and sum(p.split == split for p in picked) >= want:
                break
            if per_group.get(r.group, 0) < args.cap:
                per_group[r.group] = per_group.get(r.group, 0) + 1
                picked.append(r)
    sel = pd.DataFrame(picked).drop(columns="group")
    out = Path(f"data/meta/selected_videos_n{args.n}_cap{args.cap}.csv")
    sel.to_csv(out, index=False)
    print("saved", out)

    sel["group"] = sel.channel_url.where(sel.channel_url != "", "unknown:" + sel.youtube_id)
    print(sel.groupby("split").agg(videos=("youtube_id", "size"), channels=("group", "nunique"),
                                   max_per_channel=("group", lambda g: g.value_counts().max()),
                                   robotic=("robotic", "mean"), procedures=("procedure", "nunique")))
    print("\nTop channels in the selection:")
    print(sel.groupby(["split", "channel"]).size().sort_values(ascending=False).head(8))


if __name__ == "__main__":
    main()
