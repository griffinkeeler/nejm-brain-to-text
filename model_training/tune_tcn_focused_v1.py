"""Focused Optuna TPE search for the TCN using validation PER."""

import argparse
import gc
import math
from pathlib import Path

from omegaconf import OmegaConf


def trial_config(base, trial, output_dir):
    """Build a fresh training configuration for one trial."""
    args = OmegaConf.create(
        OmegaConf.to_container(base, resolve=True)
    )
    args.model.model_type = "tcn"

    # Search architecture near the strongest logged CUDA configuration.
    architecture = {
        "hidden_dim": trial.suggest_categorical(
            "hidden_dim", [256, 288, 384]
        ),
        "num_blocks": trial.suggest_int(
            "num_blocks", 6, 8
        ),
        "kernel_size": trial.suggest_categorical(
            "kernel_size", [3, 5, 7]
        ),
        "dropout": trial.suggest_float(
            "dropout", 0.05, 0.25
        ),
        "expansion_factor": trial.suggest_categorical(
            "expansion_factor", [2, 4]
        ),
    }

    for key, value in architecture.items():
        OmegaConf.update(
            args, f"model.tcn.{key}", value, force_add=True
        )

    # Keep the baseline architecture and patch timing fixed.
    fixed_settings = {
        "model.tcn.frame_embed_dim": None,
        "model.tcn.patch_size": 5,
        "model.tcn.patch_stride": 4,
        "model.tcn.use_glu": False,
        "model.tcn.residual_fusion": False,
        "model.tcn.use_gilr": False,
        "model.tcn.trainable": True,
        "dataset.batch_size": 64,
    }

    for key, value in fixed_settings.items():
        OmegaConf.update(args, key, value, force_add=True)

    args.model.input_network.input_layer_dropout = (
        trial.suggest_float("input_dropout", 0.1, 0.4)
    )

    # Use the same schedule for the main model and day-specific layers.
    peak_lr = trial.suggest_float(
        "peak_lr", 0.02, 0.08, log=True
    )
    decay_steps = trial.suggest_categorical(
        "decay_steps", [4000, 6000, 8000]
    )

    args.lr_scheduler_type = "cosine"

    for suffix in ("", "_day"):
        args[f"lr_max{suffix}"] = peak_lr
        args[f"lr_min{suffix}"] = 0.0001
        args[f"lr_decay_steps{suffix}"] = decay_steps
        args[f"lr_warmup_steps{suffix}"] = 78

    args.weight_decay = trial.suggest_float(
        "weight_decay", 1e-5, 1e-2, log=True
    )

    args.output_dir = str(
        Path(output_dir) / f"trial_{trial.number:05d}"
    )
    args.checkpoint_dir = str(
        Path(args.output_dir) / "checkpoint"
    )

    # Every trial starts from scratch with the same training budget.
    args.mode = "train"
    args.init_from_checkpoint = False
    args.early_stopping = False
    args.save_best_checkpoint = True
    args.save_all_val_steps = False
    args.save_final_model = False

    return args


def run_trial(trial, base, output_dir):
    import optuna
    import torch

    from rnn_trainer import BrainToTextDecoder_Trainer

    args = trial_config(base, trial, output_dir)
    trial.set_user_attr("output_dir", args.output_dir)
    trainer = None

    def report(step, metrics):
        per = float(metrics["avg_PER"])

        if not math.isfinite(per):
            raise ValueError("Validation PER is not finite")

        trial.report(per, step)

        if trial.should_prune():
            raise optuna.TrialPruned(
                f"Pruned after {step} training batches"
            )

    try:
       
        trainer = BrainToTextDecoder_Trainer(args)
        OmegaConf.save(
          args, Path(args.output_dir) / "args.yaml"
        )
        trainer.train(validation_callback=report)

        best = float(trainer.best_val_PER)

        if not math.isfinite(best):
            raise ValueError(
                "Trial completed without a finite validation PER"
            )

        return best

    finally:
        if trainer is not None:
            for handler in trainer.logger.handlers[:]:
                handler.close()
                trainer.logger.removeHandler(handler)

        del trainer
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--config",
        default="rnn_args.yaml",
    )
    parser.add_argument(
        "--output_dir",
        default="trained_models/tcn_search_cuda_focused_v1",
    )
    parser.add_argument(
        "--study_name",
        default="tcn_focused_v1",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=30,
        help="Number of additional trials for this invocation.",
    )
    parser.add_argument(
        "--num_training_batches",
        type=int,
        default=8000,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--no_pruning",
        action="store_true",
    )

    cli = parser.parse_args()

    if cli.n_trials <= 0:
        parser.error("--n_trials must be positive")

    if cli.num_training_batches <= 0:
        parser.error("--num_training_batches must be positive")

    try:
        import optuna
    except ImportError:
        parser.error(
            "Install Optuna with: python -m pip install optuna"
        )

    base = OmegaConf.load(cli.config)

    if cli.device:
        base.device = cli.device

    base.num_training_batches = cli.num_training_batches

    if base.batches_per_val_step <= 0:
        parser.error("Validation interval must be positive")

    if base.seed == -1:
        parser.error(
            "Set a fixed training seed for comparable trials"
        )

    output = Path(cli.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Check the base configuration before resuming a study.
    snapshot = output / "search_base.yaml"

    if snapshot.exists():
        previous = OmegaConf.to_container(
            OmegaConf.load(snapshot), resolve=True
        )
        current = OmegaConf.to_container(
            base, resolve=True
        )

        if previous != current:
            parser.error(
                "Search config changed; use a new --output_dir"
            )
    else:
        OmegaConf.save(base, snapshot)

    # Also prevent accidental reuse of an older search space.
    search_settings = {
        "search_version": "focused_v1",
        "sampler_seed": cli.sampler_seed,
        "pruning_enabled": not cli.no_pruning,
        "pruning_warmup_steps": 2000,
    }
    settings_path = output / "search_settings.yaml"
    database_path = output / "study.sqlite3"

    if settings_path.exists():
        previous = OmegaConf.to_container(
            OmegaConf.load(settings_path), resolve=True
        )

        if previous != search_settings:
            parser.error(
                "Search settings changed; use a new --output_dir"
            )
    elif database_path.exists():
        parser.error(
            "Existing study has no matching search settings; "
            "use a new --output_dir"
        )
    else:
        OmegaConf.save(
            OmegaConf.create(search_settings),
            settings_path,
        )

    if cli.no_pruning:
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2000,
        )

    study = optuna.create_study(
        study_name=cli.study_name,
        direction="minimize",
        load_if_exists=True,
        storage=f"sqlite:///{database_path}",
        sampler=optuna.samplers.TPESampler(
            seed=cli.sampler_seed
        ),
        pruner=pruner,
    )

    def save_best(study, trial):
        completed = study.get_trials(
            states=(optuna.trial.TrialState.COMPLETE,)
        )

        if not completed:
            return

        best = study.best_trial
        config = OmegaConf.load(
            Path(best.user_attrs["output_dir"]) / "args.yaml"
        )

        # Save a fresh-run configuration for retraining.
        # The winning checkpoint remains in its trial directory.
        config.output_dir = str(output / "best_retrain")
        config.checkpoint_dir = str(
            output / "best_retrain" / "checkpoint"
        )
        config.init_from_checkpoint = False

        OmegaConf.save(config, output / "best_args.yaml")

    study.optimize(
        lambda trial: run_trial(trial, base, output),
        n_trials=cli.n_trials,
        n_jobs=1,
        callbacks=[save_best],
        gc_after_trial=True,
    )

    save_best(study, None)

    completed = study.get_trials(
        states=(optuna.trial.TrialState.COMPLETE,)
    )

    if completed:
        print(
            f"Best validation PER: {study.best_value:.5f} "
            f"({study.best_value:.2%})"
        )
        print(
            "Winning run: "
            f"{study.best_trial.user_attrs['output_dir']}"
        )
        print(
            f"Retraining config: {output / 'best_args.yaml'}"
        )
        print(f"Best parameters: {study.best_trial.params}")
    else:
        print("No completed trials yet.")


if __name__ == "__main__":
    main()