import argparse
import os
from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer


def main():
    parser = argparse.ArgumentParser(description='Train a GRU or TCN phoneme decoder.')
    parser.add_argument('--config', default='rnn_args.yaml')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--output_dir', help='New run directory; must not already exist.')
    parser.add_argument('--model_type', choices=['gru', 'tcn'])
    cli = parser.parse_args()
    args = OmegaConf.load(cli.config)
    if cli.device is not None:
        args.device = cli.device
    if cli.model_type is not None:
        args.model.model_type = cli.model_type
    if cli.output_dir is not None:
        args.output_dir = cli.output_dir
        args.checkpoint_dir = os.path.join(cli.output_dir, 'checkpoint')
    trainer = BrainToTextDecoder_Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
