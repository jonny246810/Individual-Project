"""Step 3: read duration, fps, frame count and resolution of every LEMON video, without downloading.

ffprobe reads only the video header from Hugging Face over HTTP (a few KB per video)
Resumable: re-running only probes videos missing from the output.

Output: data/meta/video_info.csv (committed).

Usage: python scripts/probe_videos.py
"""
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import get_token, hf_hub_url

from common import LABELS_PATH, LEMON_REPO, LEMON_REVISION, VIDEO_INFO_PATH

OUT = VIDEO_INFO_PATH
FIELDS = ["youtube_id", "duration_s", "fps", "n_frames", "width", "height", "codec", "status"]


def probe(video_id, token):
    url = hf_hub_url(LEMON_REPO, f"{video_id}.mp4", repo_type="dataset", revision=LEMON_REVISION)
    cmd = ["ffprobe", "-v", "error", "-headers", f"Authorization: Bearer {token}\r\n",
           "-select_streams", "v:0",
           "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration",
           "-of", "json", url]
    for _ in range(3):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            break
    else:
        return {"youtube_id": video_id, "status": "error: " + r.stderr.strip()[:80]}
    d = json.loads(r.stdout)
    s, f = d["streams"][0], d["format"]
    num, den = map(int, s["avg_frame_rate"].split("/"))
    fps = num / den if den else 0.0
    dur = float(f["duration"])
    return {"youtube_id": video_id, "duration_s": round(dur, 3), "fps": round(fps, 3),
            "n_frames": int(s.get("nb_frames") or round(dur * fps)),
            "width": s["width"], "height": s["height"], "codec": s["codec_name"], "status": "ok"}


def main():
    ids = [x["youtubeId"] for x in json.loads(LABELS_PATH.read_text())]
    done = {}
    if OUT.exists():
        done = {r["youtube_id"]: r for r in csv.DictReader(OUT.open()) if r["status"] == "ok"}
    todo = [v for v in ids if v not in done]
    print(f"probing {len(todo)} videos ({len(done)} already done)")
    token = get_token()
    with ThreadPoolExecutor(8) as ex:
        for i, row in enumerate(ex.map(lambda v: probe(v, token), todo)):
            done[row["youtube_id"]] = row
            if i % 250 == 0:
                print(i, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(done[v] for v in ids)
    bad = [v for v in ids if done[v]["status"] != "ok"]
    print(f"saved {OUT}; {len(bad)} failed" + (f": {bad[:5]}" if bad else ""))


if __name__ == "__main__":
    main()
