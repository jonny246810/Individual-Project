"""Constants shared by the frame sampling scripts, so they are defined in one place."""
from pathlib import Path

# visurg/LEMON is read at this fixed dataset commit, so later changes to the dataset can't change frames.
LEMON_REPO = "visurg/LEMON"
LEMON_REVISION = "bcfb79705a8e0dc20d3c4f79a2c1037d3ed94e2e"

LABELS_PATH = Path("data/raw/labels.json")  # from fetch_lemon_metadata.py, not committed
META_DIR = Path("data/meta")
CHANNELS_PATH = META_DIR / "channels.csv"
SPLITS_PATH = META_DIR / "splits.csv"
VIDEO_INFO_PATH = META_DIR / "video_info.csv"
