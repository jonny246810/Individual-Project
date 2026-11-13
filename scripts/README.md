# Frame sampling procedure

How the frames used in this project are chosen from LEMON, and how to rebuild exactly the same set.
The method is **stratified sampling in time**, with videos grouped by YouTube channel.
See `experiments/phase0_frame_sampling/` for why embedding-based selection (clustering) is not used.

## The frame set
**1,428 frames** from 714 videos (499 channels), 2 frames per video:

| Split | Videos | Frames | Labels |
|---|---|---|---|
| train | 500 | 1,000 | automatic |
| val | 107 | 214 | automatic |
| test | 107 | 214 | manual |

The exact list (video, split, frame number) is `data/meta/frames_tr500_va107_te107_cap5_min30_int15_cap2.csv`.

**Frames used in this project.** The 1,428 extracted frames (plus `frames.csv` and `provenance.json`)
are also kept in a private Hugging Face dataset repo.

## Method

**1. Video metadata.** LEMON's `labels.json` (procedure types, robotic flag) is read at a pinned
commit and checked by SHA-256. Each video's YouTube channel is looked up once via YouTube's public
oEmbed endpoint and saved (`channels.csv`); 132 videos are deleted or private on YouTube and have no
channel. Duration, fps and frame count of every video are read from the file headers (`video_info.csv`).

**2. Split by channel (all 4,194 videos, 70/15/15 of videos).** All videos from one channel go to the
same split, so the test set never shares a surgeon, hospital or camera setup with training.
- Always train: animal surgeries and veterinary channels (9 videos), videos with no known channel,
  and the 10 largest channels (the largest holds 5% of LEMON, which would otherwise dominate a split).
- All other channels are shuffled (seed 0) and assigned to val, then test, until each holds 15% of
  videos; the rest go to train.
- Each video gets a seeded random rank within its split, which fixes the order videos are selected in.

**3. Select videos.** In rank order, take 500 train, 107 val and 107 test videos, skipping videos
shorter than 30 s and any video whose channel already has 5 in that split. Because the order is fixed,
selecting more videos later only adds to this list: val and test stay unchanged.

**4. Sample frames in time.** A video of D seconds gets k = min(2, ⌊D / 15⌋) frames: it is cut into k
equal windows and one frame is picked uniformly at random inside each. Here that means one random frame
from each half of every video. Each video has its own random generator, seeded from (0, video id), so its
frames don't depend on which other videos are selected. Frames are stored as exact frame numbers.

**5. Extract.** Videos are read at a fixed LEMON dataset commit (`bcfb797`, set in the scripts), so
later changes to the dataset can't change the frames. Each listed frame is decoded from the original video and saved as
`<split>/<youtube_id>/<frame:06d>.jpg` at native resolution (JPEG, ffmpeg `-q:v 2`). With `--remote`,
ffmpeg seeks inside the file on Hugging Face and fetches only the data around each frame, so no video is
downloaded; this gives pixel-identical frames to decoding the full video (checked with `--check`).

## Reproduce

Steps 1–5 are already done: their outputs are committed in `data/meta/`. To get the frames:
```bash
python scripts/fetch_lemon_metadata.py   # downloads labels.json (needed by the other scripts)
python scripts/extract_frames.py data/meta/frames_tr500_va107_te107_cap5_min30_int15_cap2.csv --remote
```
Frames are written to `data/frames/<frame list name>/`.

To rebuild the lists from scratch instead (gives identical files):
```bash
python scripts/make_splits.py
python scripts/probe_videos.py
python scripts/select_videos.py --train 500 --val 107 --test 107 --cap 5 --min-duration 30
python scripts/sample_frames.py data/meta/selected_videos_tr500_va107_te107_cap5_min30.csv --interval 15 --cap 2
```
`channels.csv` is kept as committed rather than re-fetched, because videos keep disappearing from
YouTube and a fresh lookup would change the split.

| Script | Output (in `data/meta/`) |
|---|---|
| `fetch_lemon_metadata.py` | `channels.csv` (+ `data/raw/labels.json`, not committed) |
| `make_splits.py` | `splits.csv` |
| `probe_videos.py` | `video_info.csv` |
| `select_videos.py` | `selected_videos_<...>.csv` |
| `sample_frames.py` | `frames_<...>.csv` |
| `extract_frames.py` | frames in `data/frames/` (not committed) |
