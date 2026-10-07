import argparse
import pprint
import wandb
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet", "diffusion"], default="unet",
                        help="Which model to train: unet or diffusion")
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim",
                        help="Sampler for diffusion inference: ddim (fast, 50 steps) or ddpm (slow, 1000 steps)")
    parser.add_argument("--cold-start", action="store_true",
                        help="Cold diffusion: start inference from noisy interpolation instead of pure noise")
    parser.add_argument("--cold-start-t", type=int, default=700,
                        help="Noise level for cold diffusion start (default: 200 out of 1000)")
    parser.add_argument("--n-ensemble", type=int, default=3,
                        help="Number of ensemble samples in visualization (default: 3)")
    parser.add_argument("--sweep-id", type=str, default=None,
                    help="Join an existing sweep instead of creating a new one")
    args = parser.parse_args()
    cfg = load_config()
    wandb.login()

    if args.model == "unet":
        from train import train
        sweep_cfg = cfg["sweep"]
    else:
        import diffusion_code
        diffusion_code.SAMPLER = args.sampler
        diffusion_code.COLD_START = args.cold_start
        diffusion_code.COLD_START_T = args.cold_start_t
        diffusion_code.N_ENSEMBLE = args.n_ensemble
        from diffusion_code import train
        sweep_cfg = cfg["sweep_diffusion"]

    sweep_config = {
        "method": sweep_cfg["method"],
        "metric": sweep_cfg["metric"],
        "parameters": sweep_cfg["parameters"],
    }
    pprint.pprint(sweep_config)


    if args.sweep_id:
        sweep_id=args.sweep_id
        print(f"Joining existing sweep:{sweep_id}")
    else:
        sweep_id = wandb.sweep(sweep_config, project=cfg["wandb"]["project"])
        print(f"Sweep ID: {sweep_id}")


    wandb.agent(sweep_id, function=train, count=sweep_cfg["count"], project=cfg["wandb"]["project"])


if __name__ == "__main__":
    main()
