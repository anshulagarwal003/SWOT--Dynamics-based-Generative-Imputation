# Checkpoints

`best_models/` contains the selected best checkpoint for each model and
reconstruction gap length (5, 10, and 21 days), chosen from the
hyperparameter sweep described in the paper (see Figures 32–35):

```
best_models/
├── Unet/
│   ├── chunk5_base8_kT5_bs128_lr1.00e-03_best_model.pt
│   ├── chunk10_base8_kT5_bs128_lr5.00e-04_best_model.pt
│   └── chunk21_base16_kT3_bs128_lr5.00e-04_best_model.pt
└── Diffusion/
    ├── ddpm_chunk5_base16_kT5_lr3.00e-04_best.pt
    ├── ddpm_chunk10_base16_kT3_lr1.00e-04_best.pt
    └── ddpm_chunk21_base16_kT3_lr1.00e-04_best.pt
```

Filenames encode the key hyperparameters: `chunk{N}` (temporal gap length in
days), `base{N}` (U-Net base channel width), `kT{N}` (temporal convolution
kernel size), `bs{N}` (batch size, U-Net only), `lr{...}` (learning rate).

New checkpoints produced by `src/train.py` or `src/run_sweep.py` are saved
to `./local/` by default (see `checkpoints.dir` in `src/config.yaml`) and
are not tracked by this repository.
