"""Sequential Optuna TPE search using the existing TCN trainer and validation set."""

import argparse
import gc
import math
from pathlib import Path

from omegaconf import OmegaConf


def trial_config(base, trial, output_dir):
    """Copy the training config, preserving data splits and patch timing."""
    args = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    args.model.model_type = 'tcn'
    for key, value in {
        'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 384]),
        'num_blocks': trial.suggest_int('num_blocks', 4, 8),
        'kernel_size': trial.suggest_int('kernel_size', 3, 7),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'expansion_factor': trial.suggest_categorical('expansion_factor', [1, 2, 4]),
    }.items():
        OmegaConf.update(args, f'model.tcn.{key}', value, force_add=True)
    args.model.input_network.input_layer_dropout = trial.suggest_float(
        'input_dropout', 0.0, 0.5)
    # Keep each group's configured minimum/maximum LR ratio valid.
    for suffix in ('', '_day'):
        ratio = float(base[f'lr_min{suffix}']) / float(base[f'lr_max{suffix}'])
        args[f'lr_max{suffix}'] = trial.suggest_float(
            f'lr_max{suffix}', 1e-4, 1e-2, log=True)
        args[f'lr_min{suffix}'] = args[f'lr_max{suffix}'] * ratio
    args.weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    args.output_dir = str(Path(output_dir) / f'trial_{trial.number:05d}')
    args.checkpoint_dir = str(Path(args.output_dir) / 'checkpoint')
    args.mode = 'train'
    args.init_from_checkpoint = False
    args.save_best_checkpoint = True
    args.save_all_val_steps = False
    args.save_final_model = False
    return args


def run_trial(trial, base, output_dir):
    import optuna
    import torch
    from rnn_trainer import BrainToTextDecoder_Trainer

    args = trial_config(base, trial, output_dir)
    trial.set_user_attr('output_dir', args.output_dir)
    trainer = None

    def report(step, metrics):
        per = float(metrics['avg_PER'])
        if not math.isfinite(per):
            raise ValueError('Validation PER is not finite')
        trial.report(per, step)
        if trial.should_prune():
            raise optuna.TrialPruned(f'Pruned after {step} training batches')

    try:
        trainer = BrainToTextDecoder_Trainer(args)
        OmegaConf.save(args, Path(args.output_dir) / 'args.yaml')
        trainer.train(validation_callback=report)
        best = float(trainer.best_val_PER)
        if not math.isfinite(best):
            raise ValueError('Trial completed without a finite validation PER')
        return best
    finally:
        # Release datasets, models and optimizer state between sequential trials.
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
    parser.add_argument('--config', default='rnn_args.yaml')
    parser.add_argument('--output_dir', default='trained_models/tcn_search')
    parser.add_argument('--study_name', default='tcn')
    parser.add_argument('--n_trials', type=int, default=30,
                        help='Number of additional trials for this invocation.')
    parser.add_argument('--num_training_batches', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--sampler_seed', type=int, default=42)
    parser.add_argument('--no_pruning', action='store_true')
    cli = parser.parse_args()
    if cli.n_trials <= 0:
        parser.error('--n_trials must be positive')
    try:
        import optuna
    except ImportError:
        parser.error('Install Optuna with: python -m pip install optuna')
    base = OmegaConf.load(cli.config)
    if cli.device:
        base.device = cli.device
    if cli.num_training_batches is not None:
        base.num_training_batches = cli.num_training_batches
    if base.num_training_batches <= 0 or base.batches_per_val_step <= 0:
        parser.error('Training batches and validation interval must be positive')
    if base.seed == -1:
        parser.error('Set a fixed training seed for comparable trials')
    output = Path(cli.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Refuse to mix different datasets, budgets or configurations on resume.
    snapshot = output / 'search_base.yaml'
    if snapshot.exists():
        previous = OmegaConf.to_container(OmegaConf.load(snapshot), resolve=True)
        if previous != OmegaConf.to_container(base, resolve=True):
            parser.error('Search config changed; use a new --output_dir')
    else:
        OmegaConf.save(base, snapshot)
    study = optuna.create_study(
        study_name=cli.study_name, direction='minimize', load_if_exists=True,
        storage=f'sqlite:///{output / "study.sqlite3"}',
        sampler=optuna.samplers.TPESampler(seed=cli.sampler_seed),
        pruner=(optuna.pruners.NopPruner() if cli.no_pruning else
                optuna.pruners.MedianPruner(n_startup_trials=5,
                                          n_warmup_steps=base.batches_per_val_step * 3)),
    )

    def save_best(study, trial):
        completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
        if completed:
            best = study.best_trial
            # Fresh-run config for retraining; the winning checkpoint stays in its trial directory.
            config = OmegaConf.load(Path(best.user_attrs['output_dir']) / 'args.yaml')
            config.output_dir = str(output / 'best_retrain')
            config.checkpoint_dir = str(output / 'best_retrain' / 'checkpoint')
            OmegaConf.save(config, output / 'best_args.yaml')

    study.optimize(lambda trial: run_trial(trial, base, output),
                   n_trials=cli.n_trials, n_jobs=1, callbacks=[save_best])
    save_best(study, None)
    if study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)):
        print(f'Best validation PER: {study.best_value:.5f}')
        print(f'Winning run: {study.best_trial.user_attrs["output_dir"]}')
        print(f'Retraining config: {output / "best_args.yaml"}')


if __name__ == '__main__':
    main()
