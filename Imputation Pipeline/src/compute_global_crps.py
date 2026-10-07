"""
Compute CRPS for every test-set chunk (or a capped subset per location),
for a given diffusion checkpoint, and save results to a CSV with per-chunk
lat/lon/season/CRPS. A separate plotting script aggregates this into maps.

Season is derived from the time index assuming index 0 = 2011-09-10 and
~1 day per index (matches the data's known date range through 2012-11-15).
Winter = Jan-Mar, Summer = Aug-Oct, other months are labeled "other".

Usage:
    python compute_global_crps.py \
        --diffusion-ckpt local_checkpoints/ddpm_chunk10_base16_kT3_lr1.00e-04_best.pt \
        --cold-start-t 700 --n-ensemble 5 \
        --max-chunks-per-location 3 \
        --out crps_results_chunk10.csv
"""
import argparse
import csv
import os
import re
from datetime import date, timedelta
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from model import UNET
from diffusion_code import (
    DDPM, make_endpoints_condition, linear_interp, load_config, _crps_ensemble,
)
from data import load_individual_time_series, build_datasets_for_chunk_size

DIFF_RE = re.compile(r"chunk(?P<chunk_size>\d+)_base(?P<base>\d+)_kT(?P<kT>\d+)")

DATA_START_DATE = date(2011, 9, 10)


def parse_diffusion_hparams(ckpt_path):
    m = DIFF_RE.search(os.path.basename(ckpt_path))
    if not m:
        raise ValueError(f"Could not parse chunk_size/base/kT from checkpoint filename: {ckpt_path}")
    return int(m["chunk_size"]), int(m["base"]), int(m["kT"])


def index_to_season(start_idx, chunk_size, lat):
    # use the midpoint frame of the chunk to assign a season
    mid_idx = start_idx + chunk_size // 2
    d = DATA_START_DATE + timedelta(days=int(mid_idx))
    month = d.month

    is_northern = lat >= 0
    if is_northern:
        if month in (1, 2, 3):
            return "winter"
        elif month in (8, 9, 10):
            return "summer"
    else:
        # Southern Hemisphere: local seasons are flipped relative to calendar month
        if month in (1, 2, 3):
            return "summer"
        elif month in (8, 9, 10):
            return "winter"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffusion-ckpt", required=True)
    ap.add_argument("--t-dim", type=int, default=128)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--cold-start-t", type=int, default=700)
    ap.add_argument("--n-ensemble", type=int, default=5)
    ap.add_argument("--max-chunks-per-location", type=int, default=None,
                     help="Cap test chunks per file location for a faster first pass (default: use all)")
    ap.add_argument("--season", choices=["summer", "winter", "other"], default=None,
                     help="Only evaluate chunks whose midpoint falls in this season (default: all seasons)")
    ap.add_argument("--out", default="crps_results.csv")
    args = ap.parse_args()

    chunk_size, base, kT = parse_diffusion_hparams(args.diffusion_ckpt)
    print(f"Parsed from checkpoint: chunk_size={chunk_size}, base={base}, kT={kT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()
    global_mean = np.float32(cfg["data"]["global_mean"])
    global_std  = np.float32(cfg["data"]["global_std"])

    time_series_list = load_individual_time_series(cfg["data"]["folder"])
    _, _, test_dataset = build_datasets_for_chunk_size(
        time_series_list, chunk_size, global_mean, global_std,
        train_ratio=cfg["data"]["train_ratio"], val_ratio=cfg["data"]["val_ratio"],
    )

    # optionally filter to a single season before sampling (saves GPU time)
    indices = list(range(len(test_dataset)))
    if args.season is not None:
        indices = [
            i for i in indices
            if index_to_season(test_dataset.chunks[i][1], chunk_size, test_dataset.chunks[i][4]) == args.season
        ]
        print(f"Filtered to season={args.season}: {len(indices)} chunks")

    # optionally cap chunks per location for a fast first pass
    if args.max_chunks_per_location is not None:
        per_location_count = defaultdict(int)
        capped_indices = []
        for i in indices:
            file_name = test_dataset.chunks[i][3]
            if per_location_count[file_name] < args.max_chunks_per_location:
                capped_indices.append(i)
                per_location_count[file_name] += 1
        indices = capped_indices
    print(f"Evaluating {len(indices)} test chunks (out of {len(test_dataset)} total)")

    model = UNET(in_channels=3, out_channels=1, base=base, t_dim=args.t_dim, kT=kT).to(device)
    model.load_state_dict(torch.load(args.diffusion_ckpt, map_location=device))
    model.eval()
    ddpm = DDPM(timesteps=args.timesteps, device=str(device))

    rows = []
    with torch.no_grad():
        for idx in tqdm(indices, desc="Computing CRPS"):
            x0, meta = test_dataset[idx]
            x0 = x0.unsqueeze(0).to(device)
            mask, x_cond = make_endpoints_condition(x0)
            interp = linear_interp(x0)

            ensemble = []
            for s in range(args.n_ensemble):
                torch.manual_seed(s * 10 + idx)
                noise = torch.randn(x0.shape, device=device)
                t_tensor = torch.full((1,), args.cold_start_t, device=device, dtype=torch.long)
                x_init = ddpm.q_sample(interp, t_tensor, noise)
                pred = ddpm.sample(model, x0.shape, x_cond, mask, x_init=x_init, start_t=args.cold_start_t)
                ensemble.append(pred[0, 0].cpu().numpy())
            ensemble = np.stack(ensemble)  # (N, T, H, W)

            gt_frames = x0[0, 0].cpu().numpy()
            season = index_to_season(meta["start"], chunk_size, meta["lat"])

            # average CRPS over the imputed (non-endpoint) frames of this chunk
            frame_crps = []
            for t in range(1, chunk_size - 1):  # skip known endpoints
                _, cscalar = _crps_ensemble(gt_frames[t], ensemble[:, t])
                frame_crps.append(cscalar)
            chunk_crps = float(np.mean(frame_crps)) if frame_crps else 0.0

            rows.append({
                "file_name": meta["file_name"],
                "lat": meta["lat"],
                "lon": meta["lon"],
                "start": meta["start"],
                "end": meta["end"],
                "season": season,
                "chunk_size": chunk_size,
                "crps": chunk_crps,
            })

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "lat", "lon", "start", "end", "season", "chunk_size", "crps"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
