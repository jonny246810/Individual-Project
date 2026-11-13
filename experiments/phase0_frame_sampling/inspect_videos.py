"""Phase 0: inspect a folder of LEMON videos.

For each .mp4: print codec, resolution, fps, duration, frame count and bitrate (ffprobe),
and save a contact sheet of evenly spaced frames (ffmpeg) for checking by eye.

Usage: python experiments/phase0_frame_sampling/inspect_videos.py data/raw/lemon_sample data/processed/lemon_sample
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

N_COLS, N_ROWS = 6, 4  # 24 frames per contact sheet


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames,pix_fmt"
                          ":format=duration,bit_rate,size",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    s, f = d["streams"][0], d["format"]
    num, den = map(int, s["avg_frame_rate"].split("/"))
    fps = num / den if den else 0.0
    dur = float(f["duration"])
    return {
        "video": path.stem,
        "codec": s["codec_name"],
        "pix_fmt": s.get("pix_fmt"),
        "width": s["width"],
        "height": s["height"],
        "fps": round(fps, 3),
        "duration_min": round(dur / 60, 2),
        "frames": int(s.get("nb_frames") or round(dur * fps)),
        "frames_at_1fps": int(dur),
        "size_mb": round(int(f["size"]) / 1e6, 1),
        "bitrate_kbps": round(int(f["bit_rate"]) / 1e3),
    }


def contact_sheet(path, duration_s, out_png):
    n = N_COLS * N_ROWS
    # one frame every duration/n seconds, scaled to 320 wide, tiled into a grid
    vf = f"fps={n}/{duration_s:.3f},scale=320:-2,tile={N_COLS}x{N_ROWS}:padding=4:color=white"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vf", vf,
                    "-frames:v", "1", str(out_png)], check=True)


def main(in_dir, out_dir):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(in_dir.glob("*.mp4")):
        r = probe(p)
        rows.append(r)
        contact_sheet(p, r["duration_min"] * 60, out_dir / f"{p.stem}_sheet.png")
        print(r)
    with open(out_dir / "video_stats.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(*sys.argv[1:3])
