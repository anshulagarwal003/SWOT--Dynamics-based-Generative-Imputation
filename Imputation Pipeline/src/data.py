import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset


_mmap_cache = {}


def load_individual_time_series(folder_path):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
    print("Number of files in folder:", len(files))
    time_series = [(os.path.join(folder_path, f), f) for f in files]
    print(f"Indexed {len(time_series)} files")
    return time_series


def compute_per_series_splits(time_series_list, train_ratio=0.7, val_ratio=0.15):
    per_series_splits = []
    for file_path, file_name in time_series_list:
        T = np.load(file_path, mmap_mode='r').shape[0]
        n_train = int(T * train_ratio)
        n_val = int(T * val_ratio)
        per_series_splits.append((
            file_path,
            (0, n_train),
            (n_train, n_train + n_val),
            (n_train + n_val, T),
            file_name,
        ))
    return per_series_splits


def parse_lat_lon(file_name):
    # format: eta_lat_{lat}_lon_{lon}.npy, e.g. eta_lat_m0p009682_lon_p54p989578.npy
    name = file_name.replace(".npy", "")
    parts = name.split("_")
    lat_str = parts[2]
    lon_str = parts[4]

    def decode(s):
        if s.startswith("m"):
            sign = -1.0; s = s[1:]
        elif s.startswith("p"):
            sign = 1.0; s = s[1:]
        else:
            sign = 1.0
        s = s.replace("p", ".")
        return sign * float(s)

    return decode(lat_str), decode(lon_str)


def build_all_chunks(time_series_list, chunk_size):
    chunks = []
    for file_path, file_name in time_series_list:
        T = np.load(file_path, mmap_mode='r').shape[0]
        lat, lon = parse_lat_lon(file_name)
        for start in range(0, T - chunk_size + 1, chunk_size):
            end = start + chunk_size
            chunks.append((file_path, start, end, file_name, lat, lon))
    return chunks


def split_chunks(chunks, train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(chunks))
    shuffled = [chunks[i] for i in idx]

    n_train = int(len(shuffled) * train_ratio)
    n_val   = int(len(shuffled) * val_ratio)

    train_chunks = shuffled[:n_train]
    val_chunks   = shuffled[n_train:n_train + n_val]
    test_chunks  = shuffled[n_train + n_val:]
    return train_chunks, val_chunks, test_chunks


class SpatiotemporalDataset(Dataset):
    def __init__(self, chunks, chunk_size, global_mean, global_std, augment=False):
        self.chunks = chunks  # list of (file_path, start, end, file_name, lat, lon)
        self.chunk_size = chunk_size
        self.global_mean = global_mean
        self.global_std = global_std
        self.augment = augment

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        file_path, start, end, file_name, lat, lon = self.chunks[idx]
        if file_path not in _mmap_cache:
            _mmap_cache[file_path] = np.load(file_path, mmap_mode='r')

        data = _mmap_cache[file_path]
        chunk = data[start:start + self.chunk_size].astype(np.float32)
        chunk -= chunk.mean(axis=(1, 2), keepdims=True)
        chunk = (chunk - self.global_mean) / self.global_std

        if self.augment:
            if np.random.rand() > 0.5:
                chunk = chunk[:, ::-1, :].copy()
            if np.random.rand() > 0.5:
                chunk = chunk[:, :, ::-1].copy()

        meta = {"file_name": file_name, "start": start, "end": end, "lat": lat, "lon": lon}
        return torch.from_numpy(chunk).unsqueeze(0), meta


def build_datasets_for_chunk_size(time_series_list, chunk_size, global_mean, global_std,
                                   train_ratio=0.7, val_ratio=0.15, seed=42):
    print("Building chunks..."); sys.stdout.flush()
    all_chunks = build_all_chunks(time_series_list, chunk_size)
    train_chunks, val_chunks, test_chunks = split_chunks(all_chunks, train_ratio, val_ratio, seed)

    print("Building train dataset..."); sys.stdout.flush()
    train_dataset = SpatiotemporalDataset(train_chunks, chunk_size, global_mean, global_std, augment=False)
    print(f"Train chunks: {len(train_dataset)}"); sys.stdout.flush()
    val_dataset   = SpatiotemporalDataset(val_chunks,   chunk_size, global_mean, global_std, augment=False)
    print(f"Val chunks: {len(val_dataset)}"); sys.stdout.flush()
    test_dataset  = SpatiotemporalDataset(test_chunks,  chunk_size, global_mean, global_std, augment=False)
    print(f"Test chunks: {len(test_dataset)}"); sys.stdout.flush()

    return train_dataset, val_dataset, test_dataset


def denormalize(x, mean, std):
    return x * std + mean


def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-8)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
