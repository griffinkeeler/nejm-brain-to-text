# Model Training & Evaluation
This directory contains code and resources for training the brain-to-text RNN model. This model is largely based on the architecture described in the paper "*An Accurate and Rapidly Calibrating Speech Neuroprosthesis*" by Card et al. (2024), but also contains modifications to improve performance, efficiency, and usability.

A pretrained baseline RNN model is included in the [Dryad Dataset](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85), as is the neural data required to train that model. The code for training the same model is included here.

All model training and evaluation code was tested on a computer running Ubuntu 22.04 with two RTX 4090's and 512 GB of RAM.

## Setup
1. Install the required `b2txt25` conda environment by following the instructions in the root `README.md` file. This will set up the necessary dependencies for running the model training and evaluation code.

2. Download the dataset from Dryad: [Dryad Dataset](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85). Place the downloaded data in the `data` directory. See the main [README.md](../README.md) file for more details on the included datasets and the proper `data` directory structure.

## Training
### Baseline RNN Model
We have included a custom PyTorch implementation of the RNN model used in the paper (the paper used a TensorFlow implementation). This implementation aims to replicate or improve upon the original model's performance while leveraging PyTorch's features, resulting in a more efficient training process with a slight increase in decoding accuracy. This model includes day-specific input layers (512x512 linear input layers with softsign activation), a 5-layer GRU with 768 hidden units per layer, and a linear output layer. The model is trained to predict phonemes from neural data using CTC loss and the AdamW optimizer. Data is augmented with noise and temporal jitter to improve robustness. All model hyperparameters are specified in the [`rnn_args.yaml`](rnn_args.yaml) file.

### Model training script
To train the baseline RNN model, use the `b2txt25` conda environment to run the `train_model.py` script from the `model_training` directory:
```bash
conda activate b2txt25
python train_model.py
```
The model will train for 120,000 mini-batches (~3.5 hours on an RTX 4090) and should achieve an aggregate phoneme error rate of 10.1% on the validation partition. We note that the number of training batches and specific model hyperparameters may not be optimal here, and this baseline model is only meant to serve as an example. See [`rnn_args.yaml`](rnn_args.yaml) for a list of all hyperparameters.

### Hierarchical TCN embeddings

The TCN configuration uses `model.tcn.frame_embed_dim: 128` to encode each
frame with LayerNorm, a shared linear projection, and GELU before temporal
patching. Set this to `256` for a wider frame encoder or `null` for the original
patch embedding. `hidden_dim` independently controls the patch and TCN width.
Patch size and stride, and therefore CTC output lengths, are unchanged.

Train each variant from scratch in a fresh output directory, keeping other
settings fixed when comparing validation PER. Baseline checkpoints require
`frame_embed_dim: null`; older saved configs without this setting automatically
use the baseline architecture. Evaluation reads the setting from the saved
checkpoint args.

### Residual patch fusion in the TCN

`model.tcn.residual_fusion: true` requires a non-null `frame_embed_dim`.
The raw branch projects concatenated day-transformed neural frames to
`hidden_dim`; the hierarchical branch encodes frames before patch projection.
Their concatenated embeddings pass through pointwise convolutions
`2H → 2H → GELU → H`, then the raw embedding is added to the fusion output
before the TCN. At `hidden_dim: 128`, fusion is `256 → 256 → 128`.
Both branches share the same day-transformed, dropped-out input and patch timing.
This adds parameters and computation but leaves CTC output lengths unchanged.
GLU gating can be used independently. Set `residual_fusion: false` to restore
the previous single-branch encoder. Older saved configs default to false;
train fusion variants from scratch in a fresh output directory. Evaluation
reconstructs fusion from saved args.

### GLU gating in the TCN

`model.tcn.use_glu: true` enables a gated pointwise expansion inside each
compact temporal block: `C → 2E → GLU → E → C`, where
`E = C * expansion_factor`. The depthwise causal convolution, its GELU,
normalizations, dropout, residual connection, and CTC output lengths are retained.
The wider input projection adds parameters and computation. Set `use_glu: false`
for the original GELU expansion; older saved configs without the setting also
use that architecture. Gated models require fresh training because the input
projection weights have a different shape. Evaluation reads the saved setting.

### Hierarchical GRU embeddings

Set `model.model_type: gru` to select the GRU. Its
`model.gru.frame_embed_dim: 128` setting applies the same shared frame encoder
as the TCN, then concatenates encoded frames and projects each patch to
`n_units` before the GRU. Existing patch size (14) and stride (4) are retained;
set patch size to 5 explicitly if you want five-frame patches. With patching
disabled (`patch_size: 0`), the projection operates on individual encoded frames.

Set `frame_embed_dim: null` for the baseline GRU. Older checkpoint configs
without the setting also use the baseline. Train hierarchical variants from
scratch in fresh output directories; their GRU input weights and embedding
layers differ from baseline checkpoints. Evaluation reconstructs the model
from saved args, including the frame encoder width.

## Evaluation
### Start redis server
To evaluate the model, first start a redis server on `localhost` in terminal with:
```bash
redis-server
```

### Start language model
Next, use the `b2txt25_lm` conda environment to start the ngram language model in a seperate terminal window. For example, the 1gram language model can be started using the command below. Note that the 1gram model has no gramatical structure built into it. Details on downloading pretrained 3gram and 5gram language models and running them can be found in the README.md in the `language_model` directory.
To run the 1gram language model from the root directory of this repository:
```bash
conda activate b2txt25_lm
python language_model/language-model-standalone.py --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil --do_opt --nbest 100 --acoustic_scale 0.325 --blank_penalty 90 --alpha 0.55 --redis_ip localhost --gpu_number 0
```
If the language model successfully starts and connects to Redis, you should see a message saying "Successfully connected to the redis server" in the Terminal.

### Evaluate
Finally, use the `b2txt25` conda environment to run the `evaluate_model.py` script to load the pretrained baseline RNN, use it for inference on the heldout val or test sets to get phoneme logits, pass them through the language model via redis to get word predictions, and then save the predicted sentences to a .csv file in the format required for competition submission. An example output file for the val split can be found at `rnn_baseline_submission_file_valsplit.csv`.
```bash
conda activate b2txt25
python evaluate_model.py --model_path ../data/t15_pretrained_rnn_baseline --data_dir ../data/hdf5_data_final --eval_type test --gpu_number 1
```
If the script runs successfully, it will save the predicted sentences to a text file named `baseline_rnn_{eval_type}_predicted_sentences_YYYYMMDD_HHMMSS.csv` in the pretrained model's directory (`/data/t15_pretrained_rnn_baseline`). The `eval_type` can be set to either `val` or `test`, depending on which dataset you want to evaluate.

### Shutdown redis
When you're done, you can shutdown the redis server from any terminal using `redis-cli shutdown`.

## Apple GPU training (MPS)

On an Apple Silicon Mac, `device: auto` selects CUDA if available, otherwise
Apple MPS if available, otherwise CPU. Set `device: mps` or pass `--device mps`
to require the Apple GPU and fail clearly if it is unavailable. `gpu_number`
selects a CUDA GPU only; with automatic selection, `-1` forces CPU.

From this directory, run each model separately with a fresh output directory:

```bash
conda activate b2txt25
python train_model.py --device mps --model_type tcn --output_dir trained_models/tcn_mps_run1
python train_model.py --device mps --model_type gru --output_dir trained_models/gru_mps_run1
```

The decoder runs on MPS in float32. AMP and `torch.compile` remain CUDA-only.
CTC loss is explicitly computed on CPU because the installed PyTorch 2.5
backend lacks MPS CTC support; gradients flow back to the GPU model. This
transfer and CPU computation are included in training timings. No global
`PYTORCH_ENABLE_MPS_FALLBACK` setting is required for this path.

Training-step timings synchronize the GPU, and each validation logs elapsed
training time including validation. Compare time to the same validation PER
on the same device, using the same data and multiple seeds. These are hybrid
MPS/CPU results, not measurements of pure GPU convolution or recurrence speed.
The default 120,000 batches is a full run, not a smoke test.

For neural-model inference, `evaluate_model.py --device mps` selects Apple GPU
execution. Its external Redis/language-model service still requires its own
setup; this option does not port that service to MPS.

The default TCN patch stride is 4. A stride of 5 gives only 40 output positions
for training trial `t15.2023.10.22/trial_0015`, whose target has 41 phonemes,
making its CTC alignment impossible. CTC also requires extra positions between
adjacent repeated labels. The loss helper checks these requirements and raises
a descriptive error instead of propagating an infinite loss. Use a fresh run
after changing stride, and keep it fixed when comparing timing results.

## Architecture configuration

`rnn_args.yaml` separates architecture settings into `model.gru` and
`model.tcn`. Shared neural input settings remain under `model.n_input_features`
and `model.input_network`; training schedules and dataset settings remain shared.
Select a group with `model.model_type` or the existing `--model_type` option.
For example, edit `model.gru.n_units` for GRU width or `model.tcn.hidden_dim`
for TCN width. Each group's `trainable` flag retains the existing behavior:
it controls the recurrent layers or TCN residual blocks, not the output head
or TCN patch embedding. Older checkpoints with flat architecture settings
remain supported; grouped values take precedence when both formats exist.

## TensorBoard

Install TensorBoard in the training environment with `python -m pip install tensorboard`. Logging is enabled by `tensorboard.enabled: true`
in `rnn_args.yaml`; set it to `false` to train without the dependency.
Each new run writes events to `<output_dir>/tensorboard`.

From `model_training`, start the dashboard in a second terminal:

```bash
conda activate b2txt25
tensorboard --logdir trained_models --port 6006
```

Open http://localhost:6006 and select the GRU/TCN runs to compare. Logged charts:

- `train/ctc_loss` and gradient norm before clipping, at the training log interval.
- `validation/per`, best PER, CTC loss, and PER by session, at validation steps.
- Learning rates actually used for the logged update, by optimizer group.
- Training-step, validation, elapsed, and total run times in seconds.
- `convergence/per_by_elapsed_second`: PER against elapsed seconds (choose the
  **Step** horizontal axis for this chart). Other charts use completed batches
  as their step axis. Elapsed time includes data loading, validation, and prior
  checkpoint/logging overhead, but excludes trainer initialization.

PER is a fraction: `0.2` means 20%. The config and selected device are recorded
in the Text dashboard. Event files are flushed during validation and closed
even if training raises an error or is interrupted. Logging starts with new
training runs; existing text logs are not automatically imported.
