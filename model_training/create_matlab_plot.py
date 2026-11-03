from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

# Resolve relative to this script, regardless of the launch directory.
study_dir = (
    Path(__file__).resolve().parent
    / "trained_models"
    / "052_tcn_gpu_batch64_expansion4"
)

if not study_dir.is_dir():
    raise FileNotFoundError(f"Run directory does not exist: {study_dir}")

# Accept either one training run or an Optuna study containing trials.
run_dirs = (
    [study_dir]
    if (study_dir / "tensorboard").is_dir()
    else sorted(study_dir.glob("trial_*"))
)

fig, ax = plt.subplots(figsize=(11, 6))
plotted = 0

for trial_dir in run_dirs:
    event_dir = trial_dir / "tensorboard"

    if not list(event_dir.glob("events.out.tfevents.*")):
        continue

    # Load all scalar points, without downsampling.
    events = EventAccumulator(
        str(event_dir),
        size_guidance={"scalars": 0},
    )
    events.Reload()

    tag = "validation/per"
    if tag not in events.Tags()["scalars"]:
        print(f"{trial_dir.name}: available tags:",
              events.Tags()["scalars"])
        continue

    points = events.Scalars(tag)
    if not points:
        continue
    steps = [point.step for point in points]
    per = [100 * point.value for point in points]

    ax.plot(steps, per, label=trial_dir.name)
    print(f"Loaded {len(points)} validation points from {trial_dir.name}")
    plotted += 1

ax.set(
    xlabel="Training step",
    ylabel="Validation phoneme error rate (%)",
    title=f"Validation PER: {study_dir.name}",
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
