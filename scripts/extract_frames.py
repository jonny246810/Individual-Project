"""Step 6: extract the listed frames from LEMON, then (optionally) upload them.

Two ways to get frames; both save <out>/<split>/<youtube_id>/<frame:06d>.jpg at the video's own
resolution, JPEG quality --quality (ffmpeg -q:v, 2 = high):

  default   Download each video into a temporary folder, decode it once keeping only the listed
            frame numbers (exact, decode order), delete the video. Best for many frames per video.
  --remote  Never download the video. ffmpeg reads it from Hugging Face over HTTP and seeks to each
            frame, fetching only the few MB around it. Best for a few frames per video.
            Seeks by time: frame n of a constant-fps video starts at n/fps, so we seek to
            (n - 0.5)/fps and keep the first frame at or after it. Videos with a non-standard average
            fps (likely variable frame rate) are decoded in full instead. --check compares against
            full decoding on a few videos.

Finished videos are logged in <out>/done.txt, so a dropped session resumes where it stopped.
ffmpeg's version and settings go in <out>/provenance.json.

--upload REPO uploads <out> into folder <frame list name>/ of a private Hugging Face dataset repo
(needs a token with write access to that repo).

Usage:
  python scripts/extract_frames.py data/meta/frames_<...>.csv --remote
  python scripts/extract_frames.py data/meta/frames_<...>.csv --remote --check 5   # verify first
  python scripts/extract_frames.py data/meta/frames_<...>.csv --remote --upload <user>/<repo>
"""
import argparse
import json
import platform
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, get_token, hf_hub_download, hf_hub_url
from PIL import Image

INFO = Path("data/meta/video_info.csv")
# video_info.csv rounds fps to 3 decimals; NTSC rates are really 24000/1001, 30000/1001, 60000/1001.
# The rounding error grows with frame number (~half a frame by 4 h), so snap them to exact values.
NTSC = [24000 / 1001, 30000 / 1001, 60000 / 1001]


def exact_fps(fps):
    return next((x for x in NTSC if abs(fps - x) < 0.005), fps)


def is_standard(fps):
    """Odd average rates (e.g. 22.72, 24.89) suggest variable frame rate, where seeking by time can
    miss the frame; --remote decodes those videos in full instead, which counts frames exactly."""
    return any(abs(fps - x) < 0.005 for x in [24, 25, 30, 50, 60, *NTSC])


def ffmpeg_version():
    return subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]


def extract_local(video_path, frames, out_dir, quality):
    """Decode once, keep only the given frame numbers (exact, decode order), save as JPEGs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)
    frames = sorted(frames)
    expr = "+".join(f"eq(n\\,{f})" for f in frames)
    script = tmp_dir / "filter.txt"  # long filter goes in a file, not the command line
    script.write_text(f"select='{expr}'")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video_path),
                    "-filter_script:v", str(script), "-fps_mode", "passthrough",
                    "-q:v", str(quality), str(tmp_dir / "%06d.jpg")], check=True)
    outs = sorted(tmp_dir.glob("*.jpg"))
    if len(outs) != len(frames):
        raise RuntimeError(f"{video_path.name}: expected {len(frames)} frames, got {len(outs)}")
    for f, p in zip(frames, outs):  # ffmpeg numbers outputs 1..k in order; rename to frame numbers
        p.rename(out_dir / f"{f:06d}.jpg")
    shutil.rmtree(tmp_dir)


def extract_remote(video_id, frames, fps, out_dir, quality, token):
    """Seek to each frame in the remote file; only the data around it is fetched."""
    out_dir.mkdir(parents=True, exist_ok=True)
    url = hf_hub_url("visurg/LEMON", f"{video_id}.mp4", repo_type="dataset")
    for f in frames:
        ss = max(0.0, (f - 0.5) / fps)
        subprocess.run(["ffmpeg", "-v", "error", "-y",
                        "-headers", f"Authorization: Bearer {token}\r\n",
                        "-ss", f"{ss:.6f}", "-i", url, "-frames:v", "1",
                        "-q:v", str(quality), str(out_dir / f"{f:06d}.jpg")],
                       check=True, timeout=300)


def download_and_extract(video_id, frames, out_dir, quality):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(hf_hub_download("visurg/LEMON", f"{video_id}.mp4", repo_type="dataset",
                                    local_dir=tmp))
        extract_local(path, frames, out_dir, quality)


def check(df, fps, n, quality, token):
    """Extract n videos both ways and compare pixels (JPEG noise only if they are the same frame)."""
    worst = 0.0
    for vid in df.youtube_id.unique()[:n]:
        frames = df[df.youtube_id == vid].frame.tolist()
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp, "remote"), Path(tmp, "full")
            extract_remote(vid, frames, fps[vid], a, quality, token)
            download_and_extract(vid, frames, b, quality)
            for f in frames:
                x = np.asarray(Image.open(a / f"{f:06d}.jpg"), np.float32)
                y = np.asarray(Image.open(b / f"{f:06d}.jpg"), np.float32)
                diff = float(np.abs(x - y).mean())
                worst = max(worst, diff)
                print(f"  {vid} frame {f}: mean abs pixel diff {diff:.3f}")
    print(f"worst mean abs diff {worst:.3f} (≈0 = same frame; neighbouring frames differ by much more)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame_list")
    ap.add_argument("--out", default=None,
                    help="default data/frames/<frame list name>, so different lists never mix")
    ap.add_argument("--remote", action="store_true", help="seek in the remote file, no download")
    ap.add_argument("--workers", type=int, default=8, help="videos in parallel (--remote only)")
    ap.add_argument("--quality", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="only process this many videos")
    ap.add_argument("--check", type=int, default=0, help="compare remote vs full decode on N videos")
    ap.add_argument("--upload", default=None, help="HF dataset repo id to upload --out to")
    args = ap.parse_args()

    df = pd.read_csv(args.frame_list)
    fps = {k: exact_fps(v) for k, v in pd.read_csv(INFO).set_index("youtube_id").fps.items()}
    token = get_token()
    if args.check:
        check(df, fps, args.check, args.quality, token)
        return

    out = Path(args.out or f"data/frames/{Path(args.frame_list).stem}")
    out.mkdir(parents=True, exist_ok=True)
    prov = {"frame_list": args.frame_list, "method": "remote seek" if args.remote else "full decode",
            "quality": args.quality, "ffmpeg": ffmpeg_version(), "platform": platform.platform()}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    shutil.copy(args.frame_list, out / "frames.csv")

    done_log = out / "done.txt"
    done = set(done_log.read_text().split()) if done_log.exists() else set()
    videos = [v for v in df.youtube_id.unique() if v not in done][:args.limit]
    print(f"{len(videos)} videos to extract ({len(done)} already done)")
    lock = threading.Lock()
    counter = [0]

    def run(vid):
        rows = df[df.youtube_id == vid]
        dest = out / rows.split.iloc[0] / vid
        t0 = time.time()
        if args.remote and is_standard(fps[vid]):
            extract_remote(vid, rows.frame.tolist(), fps[vid], dest, args.quality, token)
        else:
            download_and_extract(vid, rows.frame.tolist(), dest, args.quality)
        with lock:
            with done_log.open("a") as fh:
                fh.write(vid + "\n")
            counter[0] += 1
            print(f"[{counter[0]}/{len(videos)}] {vid}: {len(rows)} frames, {time.time() - t0:.0f}s",
                  flush=True)

    with ThreadPoolExecutor(args.workers if args.remote else 1) as ex:
        list(ex.map(run, videos))

    if args.upload:
        HfApi().upload_folder(repo_id=args.upload, repo_type="dataset", folder_path=out,
                              path_in_repo=out.name, ignore_patterns=["done.txt"],
                              commit_message=f"Add frames for {out.name}")
        print(f"uploaded to {args.upload}/{out.name}")


if __name__ == "__main__":
    main()
