import os
import copy
import yaml
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader

from model import UNET
from data import (
    load_individual_time_series,
    compute_per_series_splits,
    build_datasets_for_chunk_size,
    worker_init_fn,
)
from losses import make_endpoints_condition, loss_missing_only, spatial_grad_loss_missing, eval_unet
from tqdm import tqdm
from viz import log_imputation_comparison_image, log_histogram_plot, log_spectral_plot



def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def train(config=None):
    cfg = load_config()

    with wandb.init(config=config):
        config = wandb.config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        # Data
        data_folder = cfg["data"]["folder"]
        global_mean = np.float32(cfg["data"]["global_mean"])
        global_std  = np.float32(cfg["data"]["global_std"])

        time_series_list = load_individual_time_series(data_folder)

        train_dataset, val_dataset, test_dataset = build_datasets_for_chunk_size(
            time_series_list, config.chunk_size, global_mean, global_std,
            train_ratio=cfg["data"]["train_ratio"],
            val_ratio=cfg["data"]["val_ratio"],
        )

        # large chunk_size needs a smaller batch_size to fit in GPU memory
        batch_size = 64 if config.chunk_size == 21 else config.batch_size
        print(f"chunk_size={config.chunk_size} -> using batch_size={batch_size}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

        # Model
        model = UNET(
            in_channels=1,
            out_channels=1,
            base=config.base,
            kT=config.kT,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)

        best_val_loss = float("inf")
        best_epoch = -1
        best_state = None
        epochs_no_improve = 0
        early_stop_patience = 10

        checkpoint_dir = cfg["checkpoints"]["dir"]
        os.makedirs(checkpoint_dir, exist_ok=True)

        run_name = (
            f"chunk{config.chunk_size}_"
            f"base{config.base}_"
            f"kT{config.kT}_"
            f"bs{config.batch_size}_"
            f"lr{config.lr:.2e}"
        )
        best_ckpt_path  = os.path.join(checkpoint_dir, f"{run_name}_best_model.pt")
        final_ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_final_model.pt")

        for epoch in range(config.epochs):
            model.train()
            train_loss_sum = 0.0
            train_seen = 0

            print("Starting training loop...")
            train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")

            for x0, _ in train_progress:
                x0 = x0.to(device)
                B = x0.shape[0]

                mask, x_cond = make_endpoints_condition(x0)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model(x_cond)
                    pred = pred * (1.0 - mask) + x0 * mask

                    mse_missing  = loss_missing_only(pred, x0, mask, config.mse_weight)
                    grad_missing = spatial_grad_loss_missing(pred, x0, mask)
                    loss = mse_missing + config.grad_loss_weight * grad_missing

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_sum += loss.item() * B
                train_seen += B
                train_progress.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss = train_loss_sum / max(train_seen, 1)

            val_loss = eval_unet(
                model=model,
                loader=val_loader,
                device=device,
                grad_loss_weight=config.grad_loss_weight,
                mse_weight=config.mse_weight,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, best_ckpt_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "num_train_samples": len(train_dataset),
                "num_val_samples": len(val_dataset),
                "num_test_samples": len(test_dataset),
                "chunk_size": config.chunk_size,
                "base": config.base,
                "kT": config.kT,
                "grad_loss_weight": config.grad_loss_weight,
                "mse_weight": config.mse_weight,
                "batch_size": config.batch_size,
                "lr": optimizer.param_groups[0]["lr"],
            })

            print(
                f"Epoch {epoch + 1:03d}/{config.epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"best_val={best_val_loss:.6f} @ epoch {best_epoch} | "
                f"no_improve={epochs_no_improve}/{early_stop_patience}"
            )

            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        torch.save(model.state_dict(), final_ckpt_path)
        print(f"Best model saved locally at:  {best_ckpt_path}")
        print(f"Final model saved locally at: {final_ckpt_path}")

        n_viz_samples = 5
        for idx in range(min(n_viz_samples, len(val_dataset))):
            log_imputation_comparison_image(
                model=model, dataset=val_dataset, device=device,
                mean=global_mean, std=global_std, split_name="val_final", sample_idx=idx,
            )
        for idx in range(min(n_viz_samples, len(test_dataset))):
            log_imputation_comparison_image(
                model=model, dataset=test_dataset, device=device,
                mean=global_mean, std=global_std, split_name="test_final", sample_idx=idx,
            )
        log_spectral_plot(model=model, dataset=val_dataset, device=device,
                          mean=global_mean, std=global_std, split_name="val_final")
        log_spectral_plot(model=model, dataset=test_dataset, device=device,
                          mean=global_mean, std=global_std, split_name="test_final")

        log_histogram_plot(
            model=model, loader=val_loader, device=device,
            mean=global_mean, std=global_std, split_name="val_final", max_batches=20,
        )
        log_histogram_plot(
            model=model, loader=test_loader, device=device,
            mean=global_mean, std=global_std, split_name="test_final", max_batches=20,
        )
