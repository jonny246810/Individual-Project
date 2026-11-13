"""Step 2: assign every LEMON video to train / val / test, grouped by YouTube channel.

All videos from one channel go to the same split, so the model is never tested on a channel (surgeon,
hospital, camera setup) it saw in training. Rules, in order:
  1. Train only: animal surgeries, videos from veterinary channels, videos with no known channel
     (deleted/private on YouTube, so we can't tell which channel they share), and the TOP_K largest
     channels (one 5% channel would otherwise make up a third of a 15% test set).
  2. The remaining channels are shuffled (seeded) and given to val, then test, until each holds
     ~15% of all videos. Everything left goes to train.
Each video also gets a seeded random `rank` within its split; select_videos.py walks videos in rank
order, so a bigger selection always contains the smaller one.

Output: data/meta/splits.csv. Prints the balance of splits by procedure and robotic flag.

Usage: python scripts/make_splits.py
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

LABELS_PATH = Path("data/raw/labels.json")
CHANNELS_PATH = Path("data/meta/channels.csv")
OUT = Path("data/meta/splits.csv")

# Found by keyword search of titles (dog/perro/canine/pig...) then checked by hand, plus BCKK9hBJmfM,
# which has no animal in its title but comes from a veterinary channel. False positives excluded:
# "RATS" (robot-assisted thoracic surgery) and "horse shoe kidney" (human anatomy).
ANIMAL_VIDEOS = {
    "0EdLIaFKZvY": "dog splenectomy",
    "ELJXfxfWGo8": "dog splenectomy",
    "wut0TNicQDs": "dog adrenalectomy",
    "J3IfBHNhPHE": "dog cholecystectomy",
    "yMju0Sew8P4": "dog nephrectomy",
    "j9QtBYcflFE": "dog nephrectomy",
    "oLzsYmAmkeo": "dog nephrectomy",
    "SkDKb0tr35c": "pig kidney training",
    "BCKK9hBJmfM": "veterinary channel splenectomy",
}
VET_CHANNELS = {"Veterinary Endoscopy and Endosurgery", "VetmiSurg", "Renato Flaquer VET",
                "あや動物病院あや動物病院"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--top-k", type=int, default=10, help="largest channels forced into train")
    args = ap.parse_args()

    labels = {x["youtubeId"]: x for x in json.loads(LABELS_PATH.read_text())}
    ch = pd.DataFrame(csv.DictReader(CHANNELS_PATH.open()))
    df = pd.DataFrame({
        "youtube_id": ch.youtube_id,
        "channel": ch.channel,
        "channel_url": ch.channel_url,
        "procedure": [labels[v]["procedureName"][0] for v in ch.youtube_id],
        "all_procedures": ["|".join(labels[v]["procedureName"]) for v in ch.youtube_id],
        "robotic": [labels[v]["robotic"] for v in ch.youtube_id],
    })
    # Group = channel; a video with no known channel is its own group.
    df["group"] = np.where(df.channel_url != "", df.channel_url, "unknown:" + df.youtube_id)
    df["animal"] = df.youtube_id.isin(ANIMAL_VIDEOS) | df.channel.isin(VET_CHANNELS)

    size = df.group.value_counts()
    top = set(size[~size.index.str.startswith("unknown:")].index[:args.top_k])
    forced = df.animal | (df.channel_url == "") | df.group.isin(top)
    forced_groups = set(df.group[forced])
    reason = np.select([df.animal, df.channel_url == "", df.group.isin(top)],
                       ["animal", "no channel", f"top-{args.top_k} channel"], "")
    df["forced_train_reason"] = reason

    rng = np.random.default_rng(args.seed)
    free = [g for g in size.index if g not in forced_groups]
    free = list(np.array(free)[rng.permutation(len(free))])
    target = {"val": args.val * len(df), "test": args.test * len(df)}
    filled = {"val": 0, "test": 0}
    split_of = {g: "train" for g in forced_groups}
    for g in free:
        s = next((s for s in ("val", "test") if filled[s] < target[s]), "train")
        split_of[g] = s
        if s != "train":
            filled[s] += size[g]
    df["split"] = df.group.map(split_of)

    # Seeded random order within each split, used by select_videos.py.
    df["rank"] = 0
    for s in ("train", "val", "test"):
        m = df.split == s
        df.loc[m, "rank"] = rng.permutation(m.sum())

    cols = ["youtube_id", "split", "rank", "channel", "channel_url", "procedure", "all_procedures",
            "robotic", "animal", "forced_train_reason"]
    df.sort_values(["split", "rank"])[cols].to_csv(OUT, index=False)
    print("saved", OUT)

    print("\nVideos and channels per split:")
    print(df.groupby("split").agg(videos=("youtube_id", "size"), channels=("group", "nunique"),
                                  robotic=("robotic", "mean"), animal=("animal", "sum"))
          .assign(share=lambda t: t.videos / len(df)).round(3))
    print("\nForced into train:", dict(Counter(r for r in reason if r)))
    print("\nProcedure share per split (first label):")
    print((pd.crosstab(df.procedure, df.split, normalize="columns") * 100).round(1))


if __name__ == "__main__":
    main()
