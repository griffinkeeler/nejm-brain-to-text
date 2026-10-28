"""Merge an ordered checkpoint continuation chain into a new TensorBoard run.

Requires tensorboard (no PyTorch needed). Originals are never modified.
Example:
    python model_training/merge_tensorboard_runs.py --output /tmp/merged RUN RESUME RESUME2
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re

from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


VAL = re.compile(r'Val batch (\d+):')
LOAD = 'Loaded model from checkpoint: '
SAVE = 'Saved model to checkpoint: '


def checkpoint_boundary(previous, resumed):
    """Locate the exact checkpoint save preceding the next run's load."""
    loads = [line for line in (resumed / 'training_log').read_text().splitlines() if LOAD in line]
    if len(loads) != 1:
        raise ValueError(f'{resumed}: expected exactly one checkpoint load')
    load = loads[0]
    path = load.split(LOAD, 1)[1].strip()
    load_time = datetime.strptime(load[:23], '%Y-%m-%d %H:%M:%S,%f')
    candidates = []
    step = None
    for line in (previous / 'training_log').read_text().splitlines():
        match = VAL.search(line)
        if match:
            step = int(match[1]) + 1  # Trainer logs batch i at TensorBoard step i+1.
        if SAVE in line and line.split(SAVE, 1)[1].strip() == path:
            save_time = datetime.strptime(line[:23], '%Y-%m-%d %H:%M:%S,%f')
            if save_time <= load_time and step is not None:
                candidates.append(step)
    if not candidates:
        raise ValueError(f'Cannot match {path} to a save in {previous}')
    return candidates[-1]


def read_scalars(run):
    points = {}
    files = sorted((run / 'tensorboard').glob('events.out.tfevents.*'))
    if not files:
        raise ValueError(f'No TensorBoard event files in {run}')
    for path in files:
        for event in LegacyEventFileLoader(str(path)).Load():
            for value in event.summary.value:
                if value.WhichOneof('value') == 'simple_value':
                    points[value.tag, event.step] = (event.wall_time, value.simple_value)
    if not points:
        raise ValueError(f'No scalar metrics in {run}')
    return points


def merge(runs, output):
    if output.exists():
        raise FileExistsError(f'{output} already exists; choose a fresh output directory')
    boundaries = [checkpoint_boundary(a, b) for a, b in zip(runs, runs[1:])]
    sources = [read_scalars(run) for run in runs]
    merged = []
    manifest = []
    step_offset = 0
    elapsed_offset = 0.0
    for index, (run, points) in enumerate(zip(runs, sources)):
        limit = boundaries[index] if index < len(boundaries) else max(
            step for tag, step in points if tag != 'convergence/per_by_elapsed_second')
        elapsed = points.get(('timing/elapsed_seconds', limit))
        if index < len(boundaries) and elapsed is None:
            raise ValueError(f'{run}: missing elapsed time at checkpoint step {limit}')
        # This metric uses elapsed seconds as its x axis, not training steps.
        elapsed_limit = elapsed[1] if elapsed else max(
            (value for (tag, _), (_, value) in points.items()
             if tag == 'timing/elapsed_seconds'), default=0.0)
        count = 0
        for (tag, step), (wall_time, value) in points.items():
            if tag == 'timing/total_run_seconds':
                continue  # Per-invocation totals do not describe the retained trajectory.
            if tag == 'convergence/per_by_elapsed_second':
                if step > int(elapsed_limit):
                    continue
                new_step = int(elapsed_offset) + step
            else:
                if step > limit:
                    continue
                new_step = step_offset + step
                if tag == 'timing/elapsed_seconds':
                    value += elapsed_offset
            merged.append((wall_time, new_step, tag, value))
            count += 1
        manifest.append(dict(source=str(run.resolve()), local_end_step=limit,
                             step_offset=step_offset, elapsed_offset_seconds=elapsed_offset,
                             scalar_points=count))
        step_offset += limit
        elapsed_offset += elapsed_limit
    # Validate before creating output so failures leave no partial merged run.
    seen = set()
    for _, step, tag, _ in merged:
        if (tag, step) in seen:
            raise ValueError(f'Duplicate merged point: {tag} at {step}')
        seen.add((tag, step))
    writer = EventFileWriter(str(output))
    try:
        for wall_time, step, tag, value in sorted(merged):
            writer.add_event(Event(wall_time=wall_time, step=step,
                                   summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)])))
        writer.flush()
    finally:
        writer.close()
    (output / 'merge_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path, help='Run directories in continuation order')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = merge(args.runs, args.output)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    for item in manifest:
        print(f"{item['source']}: steps 1–{item['local_end_step']} + {item['step_offset']}")
    print(f'TensorBoard log directory: {args.output.resolve()}')


if __name__ == '__main__':
    main()
