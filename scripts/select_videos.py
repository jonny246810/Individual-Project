"""Step 4: choose which videos to use: a number per split, with a cap per channel.

Walks each split's videos in their seeded `rank` order (from make_splits.py) and keeps a video unless
it is shorter than --min-duration (from probe_videos.py) or its channel already has --cap videos in
the selection. Because the order is fixed, a larger count (same cap and minimum) always contains the
smaller selection, so train can grow later without changing val or test.

Output: data/meta/selected_videos_tr{train}_va{val}_te{test}_cap{cap}_min{min}.csv.
Prints channel and procedure spread.

Usage: python scripts/select_videos.py --train 500 --val 107 --test 107 --cap 5 --min-duration 30
"""
import argparse
from pathlib import Path

import pandas as pd

from common import SPLITS_PATH, VIDEO_INFO_PATH



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=500, help="number of train videos")
    ap.add_argument("--val", type=int, default=107, help="number of val videos")
    ap.add_argument("--test", type=int, default=107, help="number of test videos")
    ap.add_argument("--cap", type=int, default=5, help="max videos per channel within each split")
    ap.add_argument("--min-duration", type=float, default=30, help="skip videos shorter than this (s)")
    args = ap.parse_args()

    df = pd.read_csv(SPLITS_PATH, keep_default_na=False)
    info = pd.read_csv(VIDEO_INFO_PATH).set_index("youtube_id")
    df["duration_s"] = df.youtube_id.map(info.duration_s)
    df = df[(info.loc[df.youtube_id, "status"].values == "ok") & (df.duration_s >= args.min_duration)]
    # A video with no known channel is its own group.
    df["group"] = df.channel_url.where(df.channel_url != "", "unknown:" + df.youtube_id)

    picked = []
    for split, want in {"train": args.train, "val": args.val, "test": args.test}.items():
        per_group, n = {}, 0
        for _, r in df[df.split == split].sort_values("rank").iterrows():
            if n >= want:
                break
            if per_group.get(r.group, 0) < args.cap:
                per_group[r.group] = per_group.get(r.group, 0) + 1
                picked.append(r)
                n += 1
        if n < want:
            print(f"warning: only {n} {split} videos available (wanted {want})")
    sel = pd.DataFrame(picked).drop(columns="group")
    out = Path(f"data/meta/selected_videos_tr{args.train}_va{args.val}_te{args.test}"
               f"_cap{args.cap}_min{args.min_duration:g}.csv")
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
