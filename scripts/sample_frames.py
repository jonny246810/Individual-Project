"""Step 5: choose which frames to take from each selected video (stratified sampling in time).

For a video of length D seconds, the number of frames is k = min(--cap, floor(D / --interval)).
The video is cut into k equal windows and one frame is picked uniformly at random inside each.
This gives even coverage of the whole video (no gaps, no near-duplicate clumps) while staying unbiased.

Each video gets its own random generator seeded from (--seed, video id), so its frames don't depend on
which other videos are in the selection. Frames are stored as frame numbers (0-based decode order),
which is exact; the timestamp column is for reading only.

Output: data/meta/frames_<selection>_int{interval}_cap{cap}.csv

Usage: python scripts/sample_frames.py data/meta/selected_videos_tr500_va107_te107_cap5_min30.csv --interval 15 --cap 2
"""
import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from common import VIDEO_INFO_PATH



def sample_video(video_id, duration, fps, n_frames, interval, cap, seed):
    k = int(min(cap, duration // interval))
    if k == 0:
        return []
    rng = np.random.default_rng([seed, zlib.crc32(video_id.encode())])
    edges = np.linspace(0, n_frames, k + 1)
    frames = np.floor(edges[:-1] + rng.random(k) * np.diff(edges)).astype(int)
    frames = np.clip(frames, 0, n_frames - 1)
    return [(int(f), round(f / fps, 3)) for f in frames]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("selection")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds per window (at most 1 frame each)")
    ap.add_argument("--cap", type=int, default=300, help="max frames per video")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sel = pd.read_csv(args.selection, keep_default_na=False)
    info = pd.read_csv(VIDEO_INFO_PATH).set_index("youtube_id")
    rows = []
    for _, v in sel.iterrows():
        i = info.loc[v.youtube_id]
        for frame, t in sample_video(v.youtube_id, i.duration_s, i.fps, int(i.n_frames),
                                     args.interval, args.cap, args.seed):
            rows.append({"youtube_id": v.youtube_id, "split": v.split, "frame": frame, "t_s": t})
    df = pd.DataFrame(rows)
    stem = Path(args.selection).stem.replace("selected_videos_", "")
    out = Path(f"data/meta/frames_{stem}_int{args.interval:g}_cap{args.cap}.csv")
    df.to_csv(out, index=False)
    print("saved", out)
    per = df.groupby("youtube_id").size()
    print(df.groupby("split").agg(videos=("youtube_id", "nunique"), frames=("frame", "size")))
    print(f"frames per video: min {per.min()}, median {int(per.median())}, max {per.max()}; "
          f"total {len(df)}")


if __name__ == "__main__":
    main()
