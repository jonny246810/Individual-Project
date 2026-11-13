"""Embed video frames with LemonFM (ConvNeXt-Large, 1536-d) or an ImageNet ResNet-50 (2048-d, --model resnet50).

ffmpeg decodes each video at FPS frames/second, already resized to 224x224 RGB, and pipes raw
pixels straight in (no frames written to disk). Frames are ImageNet-normalised, as in LemonFM's
training code, and embedded in batches.

Output: one .npz per video in out_dir with
    emb (N, D) float16, t (N,) timestamp in seconds, video (str)

Usage:
    python experiments/phase0_frame_sampling/embed_frames.py data/raw/lemon_sample data/processed/lemon_sample/emb_lemonfm
    python experiments/phase0_frame_sampling/embed_frames.py ... --max-frames 64     # quick timing test
"""
import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

WEIGHTS = Path("models/lemonfm/lemonfm.pth")
SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_lemonfm(weights=WEIGHTS):
    """Same as the official model_loader.py, minus its .cuda() call and missing import."""
    net = torchvision.models.convnext_large()
    net.classifier[2] = torch.nn.Identity()
    sd = torch.load(weights, map_location="cpu", weights_only=False)["teacher"]
    sd = {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    # The classifier's LayerNorm is kept; only the final Linear became Identity.
    print(f"loaded LemonFM: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if missing:
        print("  missing:", missing[:5])
    return net.eval()


def build_resnet50():
    """ImageNet ResNet-50 baseline: global-average-pooled output, 2048-d."""
    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity()
    return net.eval()


MODELS = {"lemonfm": build_lemonfm, "resnet50": build_resnet50}


def frames(path, fps, max_frames=None):
    """Yield (timestamp, HxWx3 uint8) frames decoded by ffmpeg at a fixed rate."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-vf", f"fps={fps},scale={SIZE}:{SIZE}:flags=bicubic",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    if max_frames:
        cmd[-1:-1] = ["-frames:v", str(max_frames)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n_bytes = SIZE * SIZE * 3
    i = 0
    while True:
        buf = proc.stdout.read(n_bytes)
        if len(buf) < n_bytes:
            break
        yield i / fps, np.frombuffer(buf, np.uint8).reshape(SIZE, SIZE, 3)
        i += 1
    proc.wait()


@torch.no_grad()
def embed_video(net, dev, path, fps, batch, max_frames):
    embs, ts, batch_imgs = [], [], []
    t_decode = t_model = 0.0

    def flush():
        nonlocal t_model
        t0 = time.perf_counter()
        x = torch.from_numpy(np.stack(batch_imgs)).permute(0, 3, 1, 2).float().div(255)
        x = ((x - MEAN) / STD).to(dev, dtype=next(net.parameters()).dtype)
        embs.append(net(x).float().cpu().numpy().astype(np.float16))
        if dev == "mps":
            torch.mps.synchronize()
        t_model += time.perf_counter() - t0
        batch_imgs.clear()

    t0 = time.perf_counter()
    for t, img in frames(path, fps, max_frames):
        t_decode += time.perf_counter() - t0
        ts.append(t)
        batch_imgs.append(img)
        if len(batch_imgs) == batch:
            flush()
        t0 = time.perf_counter()
    if batch_imgs:
        flush()
    return np.concatenate(embs), np.array(ts, np.float32), t_decode, t_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--model", choices=MODELS, default="lemonfm")
    ap.add_argument("--half", action="store_true",
                    help="run the model in fp16 (ResNet-50 on MPS: 1.46x faster, cosine 0.99998 vs fp32)")
    args = ap.parse_args()

    dev = device()
    net = MODELS[args.model]().to(dev)
    if args.half:
        net = net.half()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={dev} model={args.model} half={args.half} fps={args.fps} batch={args.batch}")

    tot_n = tot_dec = tot_mod = 0.0
    for p in sorted(Path(args.in_dir).glob("*.mp4")):
        emb, ts, t_dec, t_mod = embed_video(net, dev, p, args.fps, args.batch, args.max_frames)
        np.savez(out / f"{p.stem}.npz", emb=emb, t=ts, video=p.stem)
        tot_n += len(ts); tot_dec += t_dec; tot_mod += t_mod
        print(f"{p.stem}: {len(ts)} frames, emb {emb.shape}, "
              f"decode {t_dec:.1f}s, model {t_mod:.1f}s ({len(ts) / t_mod:.1f} img/s)")
    print(f"TOTAL {int(tot_n)} frames: decode {tot_dec:.1f}s, model {tot_mod:.1f}s, "
          f"{tot_n / tot_mod:.1f} img/s on {dev}")


if __name__ == "__main__":
    main()
