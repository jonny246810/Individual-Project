"""Step 1: fetch LEMON's per-video metadata and each video's YouTube channel.

- labels.json (procedure, robotic flag, title per video) is downloaded from the LEMON GitHub repo at a
  pinned commit and checked against its SHA-256. It is not committed here (that repo has no licence),
  so it goes to data/raw/ (git-ignored).
- The channel of each video comes from YouTube's public oEmbed endpoint. Videos deleted
  or made private since LEMON was built return 404/403 and get an empty channel.
  YouTube changes over time, so the committed data/meta/channels.csv is the record used for the split.
  Re-running only looks up videos missing from it (use --refresh to redo everything).

Usage: python scripts/fetch_lemon_metadata.py
"""
import argparse
import csv
import hashlib
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LABELS_URL = ("https://raw.githubusercontent.com/visurg-ai/LEMON/"
              "24db5b167ee6c1c296ab05641f56ec34c1ee64cc/labels.json")
LABELS_SHA256 = "8365f1bae6452bf91951a900961460373c436c1e6aeb2530d5c6c5ef18a909ad"
LABELS_PATH = Path("data/raw/labels.json")
CHANNELS_PATH = Path("data/meta/channels.csv")
OEMBED = "https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={}&format=json"


def fetch_labels():
    if not LABELS_PATH.exists():
        LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        LABELS_PATH.write_bytes(urllib.request.urlopen(LABELS_URL, timeout=60).read())
    digest = hashlib.sha256(LABELS_PATH.read_bytes()).hexdigest()
    if digest != LABELS_SHA256:
        raise SystemExit(f"{LABELS_PATH} has SHA-256 {digest}, expected {LABELS_SHA256}")
    return json.loads(LABELS_PATH.read_text())


def lookup(video_id):
    req = urllib.request.Request(OEMBED.format(video_id), headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(4):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=20))
            return video_id, d.get("author_name", ""), d.get("author_url", ""), "ok"
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                return video_id, "", "", f"http{e.code}"
            time.sleep(2 ** attempt)
        except OSError:
            time.sleep(2 ** attempt)
    return video_id, "", "", "error"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="look up every video again")
    args = ap.parse_args()

    ids = [x["youtubeId"] for x in fetch_labels()]
    print(f"labels.json OK: {len(ids)} videos")

    known = {}
    if CHANNELS_PATH.exists() and not args.refresh:
        known = {r["youtube_id"]: r for r in csv.DictReader(CHANNELS_PATH.open())
                 if r["status"] != "error"}
    todo = [v for v in ids if v not in known]
    print(f"looking up {len(todo)} channels ({len(known)} already known)")
    with ThreadPoolExecutor(8) as ex:
        for v, name, url, status in ex.map(lookup, todo):
            known[v] = {"youtube_id": v, "channel": name, "channel_url": url, "status": status}

    CHANNELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHANNELS_PATH.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["youtube_id", "channel", "channel_url", "status"])
        w.writeheader()
        w.writerows(known[v] for v in ids)
    print("saved", CHANNELS_PATH)


if __name__ == "__main__":
    main()
