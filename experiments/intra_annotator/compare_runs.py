"""Agreement between two CVAT labelling runs of the same frames (intra-annotator test, decision log D11/D12).

Gives two separate things:
  1. Whole-frame agreement (IoU, NSD) on the binary tool mask: includes tools found in one run only.
     -> the "IoU ceiling" that hand labels allow.
  2. Object-level agreement: tool regions are matched across runs; for matched objects only, boundary
     distances measure how consistently the *same* object was outlined -> basis for the NSD tolerance tau.
     Objects found in only one run are counted separately (detection disagreement).

Input: two CVAT "Segmentation mask 1.1" exports (the folder that holds labelmap.txt).
A frame with no shapes has no mask PNG in the export and counts as all background.

Usage:
    python experiments/intra_annotator/compare_runs.py \
        data/processed/cvat/intra/run1/run1_masks data/processed/cvat/intra/run2/run2_masks \
        --out notes/evidence/intra/run1_vs_run2
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

FRAMES_DIR = Path("data/processed/cvat/practice_train_30")
TOOL_LABEL = "instrument"
IGNORE_LABEL = "ignore"
TAUS = [1, 2, 3, 5, 8, 10]  # pixels, for NSD
EIGHT_CONNECTED = np.ones((3, 3), bool)


def read_labelmap(run_dir):
    """labelmap.txt lines look like 'instrument:144,155,48::'. Returns {label: (r, g, b)} with lower-case
    label names, so the practice label 'instrument' and the group protocol's 'Instrument' both match."""
    colours = {}
    for line in (run_dir / "labelmap.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, rgb = line.split(":")[:2]
        colours[name.lower()] = tuple(int(v) for v in rgb.split(","))
    return colours


def load_masks(run_dir, frame, shape):
    """Returns (tool, ignore) boolean masks for one frame."""
    path = run_dir / "SegmentationClass" / f"{frame}.png"
    if not path.exists():
        empty = np.zeros(shape, bool)
        return empty, empty.copy()
    rgb = np.asarray(Image.open(path).convert("RGB"))
    colours = read_labelmap(run_dir)

    def is_label(name):
        if name not in colours:
            return np.zeros(shape, bool)
        return np.all(rgb == colours[name], axis=-1)

    return is_label(TOOL_LABEL), is_label(IGNORE_LABEL)


def boundary(mask):
    """Pixels of the mask that touch background. The image edge is not counted as a boundary
    (a tool leaving the frame is cut by the camera, not outlined by me)."""
    return mask & ~ndimage.binary_erosion(mask, EIGHT_CONNECTED, border_value=1)


def boundary_distances(a, b):
    """Distance (px) from each boundary pixel of a to the nearest boundary pixel of b, and vice versa."""
    ba, bb = boundary(a), boundary(b)
    if not ba.any() or not bb.any():
        return np.full(ba.sum(), np.inf), np.full(bb.sum(), np.inf)
    to_b = ndimage.distance_transform_edt(~bb)
    to_a = ndimage.distance_transform_edt(~ba)
    return to_b[ba], to_a[bb]


def nsd(d_ab, d_ba, tau):
    n = len(d_ab) + len(d_ba)
    return np.nan if n == 0 else ((d_ab <= tau).sum() + (d_ba <= tau).sum()) / n


def iou(a, b):
    union = (a | b).sum()
    return np.nan if union == 0 else (a & b).sum() / union


def components(mask, min_area):
    """Connected tool regions; regions smaller than min_area are dropped (stray clicks)."""
    labels, n = ndimage.label(mask, EIGHT_CONNECTED)
    areas = ndimage.sum(mask, labels, range(1, n + 1))
    keep = [i + 1 for i, area in enumerate(areas) if area >= min_area]
    dropped = n - len(keep)
    return [labels == i for i in keep], dropped


def match_objects(comps_a, comps_b):
    """Group regions that overlap across runs (union-find), so a tool drawn as one region in one run
    and two regions in the other is still a single matched object. Returns a list of (a_ids, b_ids)."""
    parent = {("a", i): ("a", i) for i in range(len(comps_a))}
    parent.update({("b", j): ("b", j) for j in range(len(comps_b))})

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, ca in enumerate(comps_a):
        for j, cb in enumerate(comps_b):
            if (ca & cb).any():
                parent[find(("a", i))] = find(("b", j))

    groups = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    return [([i for s, i in g if s == "a"], [j for s, j in g if s == "b"]) for g in groups.values()]


def union_of(comps, ids, shape):
    out = np.zeros(shape, bool)
    for i in ids:
        out |= comps[i]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-area", type=int, default=50, help="drop tool regions smaller than this (px)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    name_a, name_b = args.run_a.parent.name, args.run_b.parent.name

    frames = sorted(p.stem for p in FRAMES_DIR.glob("*.jpg"))
    frame_rows, object_rows, matched_dists = [], [], []
    all_dab, all_dba = [], []
    inter_total = union_total = 0
    both_empty = dropped_total = 0

    for idx, frame in enumerate(frames):
        w, h = Image.open(FRAMES_DIR / f"{frame}.jpg").size
        tool_a, ign_a = load_masks(args.run_a, frame, (h, w))
        tool_b, ign_b = load_masks(args.run_b, frame, (h, w))
        ignore = ign_a | ign_b  # pixels marked ignore in either run are left out of every number
        tool_a, tool_b = tool_a & ~ignore, tool_b & ~ignore

        # 1. whole frame
        if not tool_a.any() and not tool_b.any():
            both_empty += 1
            frame_rows.append({"frame_index": idx, "frame": frame, "width": w, "height": h, "status": "both empty"})
            continue
        d_ab, d_ba = boundary_distances(tool_a, tool_b)
        all_dab.append(d_ab)
        all_dba.append(d_ba)
        inter_total += (tool_a & tool_b).sum()
        union_total += (tool_a | tool_b).sum()
        row = {"frame_index": idx, "frame": frame, "width": w, "height": h, "status": "compared",
               f"tool_px_{name_a}": int(tool_a.sum()), f"tool_px_{name_b}": int(tool_b.sum()),
               "iou": round(iou(tool_a, tool_b), 4)}
        row.update({f"nsd_tau{t}": round(nsd(d_ab, d_ba, t), 4) for t in TAUS})

        # 2. objects
        comps_a, drop_a = components(tool_a, args.min_area)
        comps_b, drop_b = components(tool_b, args.min_area)
        dropped_total += drop_a + drop_b
        n_matched = n_only_a = n_only_b = 0
        for a_ids, b_ids in match_objects(comps_a, comps_b):
            obj_a = union_of(comps_a, a_ids, (h, w))
            obj_b = union_of(comps_b, b_ids, (h, w))
            orow = {"frame_index": idx, "frame": frame, f"regions_{name_a}": len(a_ids),
                    f"regions_{name_b}": len(b_ids), f"area_{name_a}": int(obj_a.sum()),
                    f"area_{name_b}": int(obj_b.sum())}
            if a_ids and b_ids:
                n_matched += 1
                oab, oba = boundary_distances(obj_a, obj_b)
                dists = np.concatenate([oab, oba])
                matched_dists.append(dists)
                orow.update(status="matched", iou=round(iou(obj_a, obj_b), 4),
                            mean_boundary_dist=round(dists.mean(), 2),
                            p95_boundary_dist=round(np.percentile(dists, 95), 2))
            else:
                n_only_a += bool(a_ids)
                n_only_b += bool(b_ids)
                orow["status"] = f"only in {name_a if a_ids else name_b}"
            object_rows.append(orow)
        row.update(objects_matched=n_matched, **{f"objects_only_{name_a}": n_only_a,
                                                 f"objects_only_{name_b}": n_only_b})
        frame_rows.append(row)

    # ---- summary
    compared = [r for r in frame_rows if r["status"] == "compared"]
    ious = np.array([r["iou"] for r in compared])
    worst = min(compared, key=lambda r: r["iou"])
    d_ab, d_ba = np.concatenate(all_dab), np.concatenate(all_dba)
    matched = [r for r in object_rows if r["status"] == "matched"]
    unmatched = [r for r in object_rows if r["status"] != "matched"]
    md = np.concatenate(matched_dists)
    md_finite = md[np.isfinite(md)]
    pct = {p: np.percentile(md_finite, p) for p in (50, 90, 95, 99)}
    tau_suggest = int(np.ceil(pct[95]))

    lines = [
        f"# Intra-annotator agreement: {name_a} vs {name_b}",
        "",
        f"Frames: {len(frames)} total, {len(compared)} compared, {both_empty} empty in both runs (left out).",
        f"Tool regions under {args.min_area} px dropped as stray clicks: {dropped_total}.",
        "",
        "## 1. Whole-frame agreement (includes tools found in one run only): the IoU ceiling",
        f"- IoU, mean over frames: **{ious.mean():.3f}** (median {np.median(ious):.3f}, "
        f"min {ious.min():.3f} at frame {worst['frame_index']} `{worst['frame']}`)",
        f"- IoU, pooled over all pixels: **{inter_total / union_total:.3f}**",
        "- NSD (pooled over all boundary pixels): " + ", ".join(
            f"tau={t}px **{nsd(d_ab, d_ba, t):.3f}**" for t in TAUS),
        "",
        "## 2. Object-level agreement (same object outlined twice): the basis for tau",
        f"- Objects matched across runs: {len(matched)}; found in one run only: {len(unmatched)}",
    ]
    for r in unmatched:
        lines.append(f"  - frame {r['frame_index']} `{r['frame']}`: {r['status']} "
                     f"(area {max(r[f'area_{name_a}'], r[f'area_{name_b}'])} px)")
    lines += [
        f"- Matched-object IoU: mean **{np.mean([r['iou'] for r in matched]):.3f}**, "
        f"median {np.median([r['iou'] for r in matched]):.3f}, min {min(r['iou'] for r in matched):.3f}",
        f"- Boundary distance between the two outlines (matched objects, {len(md_finite):,} boundary px): "
        f"mean {md_finite.mean():.2f}, median {pct[50]:.2f}, p90 {pct[90]:.2f}, "
        f"**p95 {pct[95]:.2f}**, p99 {pct[99]:.2f} px",
        f"- Suggested tau (p95, rounded up): **{tau_suggest} px** at native frame resolution "
        f"(frame widths here: {sorted({r['width'] for r in frame_rows})}). "
        "Rescale if the model works at a different resolution.",
        "",
        "Files: per_frame.csv, per_object.csv, boundary_distances.png",
    ]
    summary = "\n".join(lines)
    (args.out / "summary.md").write_text(summary + "\n")
    print(summary)

    for name, rows in [("per_frame.csv", frame_rows), ("per_object.csv", object_rows)]:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(args.out / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(md_finite, bins=np.arange(0, min(md_finite.max(), 40) + 1, 1), color="#4a7ab5")
    ax.axvline(pct[95], color="#c0392b", linestyle="--", label=f"p95 = {pct[95]:.1f} px")
    ax.set_xlabel("distance from one outline to the other (px)")
    ax.set_ylabel("boundary pixels")
    ax.set_title(f"Same objects outlined twice: {name_a} vs {name_b}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "boundary_distances.png", dpi=150)


if __name__ == "__main__":
    main()
