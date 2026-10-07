import torch
from tqdm import tqdm


def make_endpoints_condition(x0):
    B, C, T, H, W = x0.shape
    mask = torch.zeros((B, 1, T, H, W), device=x0.device, dtype=x0.dtype)
    mask[:, :, 0] = 1.0
    mask[:, :, -1] = 1.0
    x_cond = x0 * mask
    return mask, x_cond


def loss_missing_only(pred, x0, mask, mse_weight=0.5):
    w = 1.0 - mask
    l2 = ((pred - x0) ** 2 * w).sum() / (w.sum() + 1e-8)
    l1 = (torch.abs(pred - x0) * w).sum() / (w.sum() + 1e-8)
    return mse_weight * l2 + (1.0 - mse_weight) * l1


def spatial_grad_loss_missing(pred, target, mask):
    w = 1.0 - mask
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    wx = w[..., :, 1:] * w[..., :, :-1]

    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    wy = w[..., 1:, :] * w[..., :-1, :]

    loss_x = ((dx_p - dx_t) ** 2 * wx).sum() / (wx.sum() + 1e-8)
    loss_y = ((dy_p - dy_t) ** 2 * wy).sum() / (wy.sum() + 1e-8)
    return loss_x + loss_y


def eval_unet(model, loader, device, grad_loss_weight=0.0, mse_weight=0.5):
    model.eval()
    loss_sum = 0.0
    n_seen = 0

    with torch.no_grad():
        val_progress = tqdm(loader, desc="Validating", leave=False)
        for x0, _ in val_progress:
            x0 = x0.to(device)
            B = x0.shape[0]

            mask, x_cond = make_endpoints_condition(x0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(x_cond)
                pred = pred * (1.0 - mask) + x0 * mask

                mse = loss_missing_only(pred, x0, mask, mse_weight)
                grad = spatial_grad_loss_missing(pred, x0, mask)
                loss = mse + grad_loss_weight * grad

            loss_sum += loss.item() * B
            n_seen += B

    return loss_sum / max(n_seen, 1)
