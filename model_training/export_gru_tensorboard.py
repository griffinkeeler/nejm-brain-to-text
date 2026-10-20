"""Reconstruct TensorBoard scalars from saved GRU logs without training."""
import argparse
from datetime import datetime
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from torch.utils.tensorboard import SummaryWriter

STAMP = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}): (.*)$')
TRAIN = re.compile(r'Train batch (\d+): loss: ([\d.eE+-]+) grad norm: ([\w.eE+-]+) time: ([\d.eE+-]+)')
VAL = re.compile(r'Val batch (\d+): PER \(avg\): ([\d.eE+-]+) CTC Loss \(avg\): ([\d.eE+-]+) time: ([\d.eE+-]+)')
ELAPSED = re.compile(r'Elapsed training time including validation: ([\d.eE+-]+) seconds')
SESSION = re.compile(r'(\S+) val PER: ([\d.eE+-]+)')


def export_run(run, destination, timezone):
    log = run / 'training_log'
    if not log.exists():
        return 0, 0
    if destination.exists():
        raise FileExistsError(f'Export already exists: {destination}; choose a new output directory')
    train_count = val_count = 0
    elapsed = None
    step = None
    best = float('inf')
    with SummaryWriter(str(destination)) as writer:
        for line in log.read_text().splitlines():
            match = STAMP.match(line)
            if not match:
                continue
            stamp, message = match.groups()
            walltime = datetime.strptime(stamp, '%Y-%m-%d %H:%M:%S,%f').replace(
                tzinfo=ZoneInfo(timezone)).timestamp()
            elapsed_match = ELAPSED.search(message)
            if elapsed_match:
                elapsed = float(elapsed_match[1])
            train = TRAIN.search(message)
            if train:
                batch, loss, gradient, duration = train.groups()
                train_step = int(batch) + 1
                for tag, value in [('train/ctc_loss', loss),
                                   ('train/gradient_norm_before_clip', gradient),
                                   ('timing/train_step_seconds', duration)]:
                    writer.add_scalar(tag, float(value), train_step, walltime=walltime)
                train_count += 1
            val = VAL.search(message)
            if val:
                batch, per, loss, duration = val.groups()
                step = int(batch) + 1
                best = min(best, float(per))
                for tag, value in [('validation/per', per), ('validation/ctc_loss', loss),
                                   ('validation/best_per', best),
                                   ('timing/validation_seconds', duration)]:
                    writer.add_scalar(tag, float(value), step, walltime=walltime)
                if elapsed is not None:
                    writer.add_scalar('timing/elapsed_seconds', elapsed, step, walltime=walltime)
                    writer.add_scalar('convergence/per_by_elapsed_second', float(per),
                                      int(elapsed), walltime=walltime)
                elapsed = None
                val_count += 1
            session = SESSION.search(message)
            if session and step is not None:
                writer.add_scalar(f'validation_by_session/{session[1]}/per',
                                  float(session[2]), step, walltime=walltime)
    return train_count, val_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).parent / 'trained_models')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timezone', default='America/Chicago', help='Timezone of saved log timestamps')
    args = parser.parse_args()
    output = args.output or args.root / 'tensorboard_gru_comparison'
    runs = sorted(p for p in args.root.rglob('gru*') if p.is_dir() and (p / 'training_log').exists())
    destinations = [(run, output / run.relative_to(args.root)) for run in runs]
    for _, destination in destinations:
        if destination.exists():
            parser.error(f'{destination} already exists; use a fresh --output directory')
    for run, destination in destinations:
        train, val = export_run(run, destination, args.timezone)
        print(f'{run.relative_to(args.root)}: {train} training points, {val} validation points')
    print(f'TensorBoard log directory: {output.resolve()}')


if __name__ == '__main__':
    main()
