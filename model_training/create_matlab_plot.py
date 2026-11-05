import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

# Resolve relative to this script, regardless of the launch directory.
study_dir = Path(__file__).resolve().parent / "trained_models"

parser = argparse.ArgumentParser(
    description="Compare TensorBoard validation PER for selected training runs."
)
parser.add_argument(
    "run_names", nargs="+",
    help="Run folders or TensorBoard event folders relative to trained_models.",
)
args = parser.parse_args()
run_names = args.run_names
run_dirs = [study_dir / name for name in run_names]

for run_dir in run_dirs:
    if not run_dir.is_dir():
        parser.error(f"Run directory does not exist: {run_dir}")

fig, ax = plt.subplots(figsize=(11, 6))
plotted = 0

labels = {
    "001_gru_baseline/tensorboard_merged": "GRU baseline",
    "052_tcn_gpu_batch64_expansion4": "TCN expansion 4",
}

for trial_dir in run_dirs:
    # Accept event directories directly, including tensorboard_merged.
    event_dir = (
        trial_dir
        if list(trial_dir.glob("events.out.tfevents.*"))
        else trial_dir / "tensorboard"
    )
    run_label = str(trial_dir.relative_to(study_dir))

    if not list(event_dir.glob("events.out.tfevents.*")):
        print(f"{run_label}: no TensorBoard event files found in {event_dir}")
        continue

    # Load all scalar points, without downsampling.
    events = EventAccumulator(
        str(event_dir),
        size_guidance={"scalars": 0},
    )
    events.Reload()

    tag = "validation/per"
    if tag not in events.Tags()["scalars"]:
        print(f"{run_label}: available tags:",
              events.Tags()["scalars"])
        continue

    points = events.Scalars(tag)
    if not points:
        continue
    steps = [point.step for point in points]
    per = [100 * point.value for point in points]

    ax.plot(steps, per, label=labels.get(run_label, run_label))
    print(f"Loaded {len(points)} validation points from {run_label}")
    plotted += 1

ax.set(
    xlabel="Training step",
    ylabel="Validation phoneme error rate (%)",
    title="Training runs: validation PER",
)
ax.grid(alpha=0.25)

# Hide the very large initial PER to make convergence easier to see.
ax.set_ylim(0, 50)

if plotted:
    ax.legend(title="Run", bbox_to_anchor=(1.02, 1),
              loc="upper left")
else:
    plt.close(fig)
    raise RuntimeError(f"No validation curves found in {study_dir}")

fig.tight_layout()

# Save a publication-friendly vector file and a high-resolution image.
fig.savefig(study_dir / "validation_per.pdf", bbox_inches="tight")
fig.savefig(
    study_dir / "validation_per.png",
    dpi=300,
    bbox_inches="tight",
)
print(f"Saved validation_per.pdf and validation_per.png in {study_dir}")
plt.show()
