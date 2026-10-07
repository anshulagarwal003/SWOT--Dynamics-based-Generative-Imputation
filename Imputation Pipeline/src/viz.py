import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb

from losses import make_endpoints_condition
from data import denormalize, r2_score_np
from spectra import compute_isotropic_psd


def collect_gt_pred_values(model, loader, device, mean, std, max_batches=None):
    model.eval()
    mean_t = torch.as_tensor(mean, device=device, dtype=torch.float32)
    std_t = torch.as_tensor(std, device=device, dtype=torch.float32)
    all_gt, all_pred, all_interp = [], [], []

    with torch.no_grad():
        for batch_idx, (x0, _) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x0 = x0.to(device)
            mask, x_cond = make_endpoints_condition(x0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(x_cond)

            pred = pred * (1.0 - mask) + x0 * mask
            B, C, T, H, W = x0.shape
            tau = torch.linspace(0, 1, T, device=device, dtype=x0.dtype).view(1, 1, T, 1, 1)
            interp = (1.0 - tau) * x0[:, :, 0:1] + tau * x0[:, :, -1:]
            interp = interp * (1.0 - mask) + x0 * mask

            x0_den = denormalize(x0, mean_t, std_t)
            pred_den = denormalize(pred, mean_t, std_t)
            interp_den = denormalize(interp, mean_t, std_t)

            missing_mask = (1.0 - mask).bool()

            stride = 50
            all_gt.append(x0_den[missing_mask][::stride].detach().cpu().numpy())
            all_pred.append(pred_den[missing_mask][::stride].detach().cpu().numpy())
            all_interp.append(interp_den[missing_mask][::stride].detach().cpu().numpy())

    if len(all_gt) == 0:
        return None, None, None
    return (np.concatenate(all_gt), np.concatenate(all_pred), np.concatenate(all_interp))


def log_histogram_plot(model, loader, device, mean, std, split_name="val", max_batches=None, bins=100):
    gt_vals, pred_vals, interp_vals = collect_gt_pred_values(
        model=model, loader=loader, device=device, mean=mean, std=std, max_batches=max_batches
    )
    key = f"{split_name}_histogram_plot"

    if gt_vals is None:
        fig = plt.figure(figsize=(10, 5))
        plt.axis("off")
        fig_path = f"{split_name}_histogram_plot.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        wandb.log({f"{split_name}_hist_available": 0, key: wandb.Image(fig_path)})
        plt.close(fig)
        return

    bias = np.mean(pred_vals - gt_vals)
    interp_bias = np.mean(interp_vals - gt_vals)
    gt_std = np.std(gt_vals)
    pred_std = np.std(pred_vals)
    interp_std = np.std(interp_vals)
    pred_r2 = r2_score_np(gt_vals, pred_vals)
    interp_r2 = r2_score_np(gt_vals, interp_vals)

    wandb.log({
        f"{split_name}_hist_available": 1,
        f"{split_name}_bias": float(bias),
        f"{split_name}_interp_bias": float(interp_bias),
        f"{split_name}_gt_mean": float(np.mean(gt_vals)),
        f"{split_name}_pred_mean": float(np.mean(pred_vals)),
        f"{split_name}_interp_mean": float(np.mean(interp_vals)),
        f"{split_name}_gt_std": float(gt_std),
        f"{split_name}_pred_std": float(pred_std),
        f"{split_name}_interp_std": float(interp_std),
        f"{split_name}_std_ratio_pred_over_gt": float(pred_std / (gt_std + 1e-8)),
        f"{split_name}_std_ratio_interp_over_gt": float(interp_std / (gt_std + 1e-8)),
        f"{split_name}_gt_hist": wandb.Histogram(gt_vals),
        f"{split_name}_pred_hist": wandb.Histogram(pred_vals),
        f"{split_name}_interp_hist": wandb.Histogram(interp_vals),
        f"{split_name}_r2_pred": float(pred_r2),
        f"{split_name}_r2_interp": float(interp_r2),
    })

    fig = plt.figure(figsize=(10, 5))
    plt.hist(gt_vals, bins=bins, alpha=0.5, density=True, histtype="step", label=f"{split_name} GT")
    plt.hist(pred_vals, bins=bins, alpha=0.5, density=True, histtype="step", label=f"{split_name} UNet Prediction")
    plt.hist(interp_vals, bins=bins, alpha=0.5, density=True, histtype="step", label=f"{split_name} Linear Interpolation")
    plt.xlabel("SSH anomaly value")
    plt.ylabel("Density")
    plt.title(f"{split_name}: Full-split Histogram (GT vs Interp vs UNet)")
    plt.legend()
    plt.grid(True)
    fig_path = f"{split_name}_histogram_plot.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    wandb.log({key: wandb.Image(fig_path)})
    plt.close(fig)


def log_imputation_comparison_image(model, dataset, device, mean, std, split_name="val_final", sample_idx=0):
    model.eval()
    mean_t = torch.as_tensor(mean, device=device, dtype=torch.float32)
    std_t = torch.as_tensor(std, device=device, dtype=torch.float32)

    if len(dataset) == 0:
        return

    sample_idx = min(sample_idx, len(dataset) - 1)
    n_samples = len(dataset)

    x0, meta = dataset[sample_idx]
    x0 = x0.unsqueeze(0).to(device)

    with torch.no_grad():
        mask, x_cond = make_endpoints_condition(x0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(x_cond)

        pred = pred * (1.0 - mask) + x0 * mask
        B, C, T, H, W = x0.shape
        tau = torch.linspace(0, 1, T, device=device, dtype=x0.dtype).view(1, 1, T, 1, 1)
        interp = (1.0 - tau) * x0[:, :, 0:1] + tau * x0[:, :, -1:]

        x0 = denormalize(x0, mean_t, std_t)
        x_cond = denormalize(x_cond, mean_t, std_t)
        pred = denormalize(pred, mean_t, std_t)
        interp = denormalize(interp, mean_t, std_t)

    gt = x0[0, 0].detach().cpu().numpy()
    inp = x_cond[0, 0].detach().cpu().numpy()
    pr = pred[0, 0].detach().cpu().numpy()
    interp_np = interp[0, 0].detach().cpu().numpy()
    diff_pred = gt - pr
    diff_interp = gt - interp_np
    diff_interp_pred = interp_np - pr
    T = gt.shape[0]
    vmin, vmax = gt.min(), gt.max()
    diff_abs = max(
        np.max(np.abs(diff_pred)),
        np.max(np.abs(diff_interp)),
        np.max(np.abs(diff_interp_pred)),
        1e-8,
    )

    rows = [
        ("GT", gt, vmin, vmax),
        ("Input endpoints", inp, vmin, vmax),
        ("UNet prediction", pr, vmin, vmax),
        ("Linear interpolation", interp_np, vmin, vmax),
        ("GT - UNet", diff_pred, -diff_abs, diff_abs),
        ("GT - Interp", diff_interp, -diff_abs, diff_abs),
        ("Interp - UNet", diff_interp_pred, -diff_abs, diff_abs),
    ]

    n_image_rows = len(rows)
    n_total_rows = n_image_rows + 1  # +1 for spectra row
    fig, axes = plt.subplots(n_total_rows, T, figsize=(2.2 * T, 2.0 * n_total_rows), constrained_layout=True)
    if n_total_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, (label, arr, row_vmin, row_vmax) in enumerate(rows):
        for t in range(T):
            ax = axes[r, t]
            im = ax.imshow(arr[t], cmap="RdBu_r", vmin=row_vmin, vmax=row_vmax)
            if r == 0:
                ax.set_title(f"t={t}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axes[r, 0].set_ylabel(label, fontsize=10, rotation=0, labelpad=45, va="center")

    # Spectra row: one PSD plot per time step
    for t in range(T):
        ax = axes[n_image_rows, t]
        k_gt, p_gt = compute_isotropic_psd(gt[t], dx=2.0, dy=2.0)
        k_pr, p_pr = compute_isotropic_psd(pr[t], dx=2.0, dy=2.0)
        k_in, p_in = compute_isotropic_psd(interp_np[t], dx=2.0, dy=2.0)
        if k_gt is not None and p_gt is not None:
            ax.loglog(k_gt, p_gt, color="black", lw=1.0, label="GT")
        if k_pr is not None and p_pr is not None:
            ax.loglog(k_pr, p_pr, color="blue", lw=1.0, label="UNet")
        if k_in is not None and p_in is not None:
            ax.loglog(k_in, p_in, color="red", lw=1.0, linestyle="--", label="Interp")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, which="both", ls="--", lw=0.4)
        if t == 0:
            ax.legend(fontsize=5, loc="lower left")
    axes[n_image_rows, 0].set_ylabel("PSD", fontsize=10, rotation=0, labelpad=45, va="center")

    plt.suptitle(
        f"{split_name}: single-sample visualization (idx={sample_idx}/{n_samples - 1}) | "
        f"lat={meta['lat']:.2f}, lon={meta['lon']:.2f}, t=[{meta['start']}:{meta['end']}]",
        fontsize=12,
    )
    key = f"{split_name}_imputation_comparison_idx{sample_idx}"
    wandb.log({key: wandb.Image(fig)})
    plt.close(fig)

def log_spectral_plot(model, dataset, device, mean, std, split_name="val", sample_idx=0, dx=2.0, dy=2.0):
    model.eval()
    mean_t = torch.as_tensor(mean, device=device, dtype=torch.float32)
    std_t = torch.as_tensor(std, device=device, dtype=torch.float32)

    if len(dataset) == 0:
        return

    sample_idx = min(sample_idx, len(dataset) - 1)
    x0, _ = dataset[sample_idx]
    x0 = x0.unsqueeze(0).to(device)

    with torch.no_grad():
        mask, x_cond = make_endpoints_condition(x0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(x_cond)
        pred = pred * (1.0 - mask) + x0 * mask
        B, C, T, H, W = x0.shape
        tau = torch.linspace(0, 1, T, device=device, dtype=x0.dtype).view(1, 1, T, 1, 1)
        interp = (1.0 - tau) * x0[:, :, 0:1] + tau * x0[:, :, -1:]

        x0 = denormalize(x0, mean_t, std_t)
        pred = denormalize(pred, mean_t, std_t)
        interp = denormalize(interp, mean_t, std_t)

    gt = x0[0, 0].detach().cpu().numpy()          # (T, H, W)
    pr = pred[0, 0].detach().cpu().numpy()
    interp_np = interp[0, 0].detach().cpu().numpy()

    gt_psds, pred_psds, interp_psds, k_ref = [], [], [], None
    for t in range(gt.shape[0]):
        k, p_gt = compute_isotropic_psd(gt[t], dx, dy)
        _, p_pr = compute_isotropic_psd(pr[t], dx, dy)
        _, p_in = compute_isotropic_psd(interp_np[t], dx, dy)
        if k_ref is None:
            k_ref = k
        gt_psds.append(p_gt)
        pred_psds.append(p_pr)
        interp_psds.append(p_in)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(k_ref, np.mean(gt_psds, axis=0), label='GT', color='black')
    ax.loglog(k_ref, np.mean(pred_psds, axis=0), label='UNet Prediction', color='blue')
    ax.loglog(k_ref, np.mean(interp_psds, axis=0), label='Linear Interpolation', color='red', linestyle='--')
    ax.set_xlabel('Wavenumber (cpkm)')
    ax.set_ylabel('PSD (m² cpkm⁻¹)')
    ax.set_title(f'{split_name}: Isotropic PSD (sample idx={sample_idx})')
    ax.legend()
    ax.grid(True, which='both', ls='--')
    plt.tight_layout()

    wandb.log({f"{split_name}_spectral_plot_idx{sample_idx}": wandb.Image(fig)})
    plt.close(fig)
