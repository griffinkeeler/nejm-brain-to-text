# Whole-word CTC pilot

This optional mode trains the existing TCN to emit whole words, without a
pronunciation dictionary or external language model. Existing configs default to
phoneme targets. Creating or inspecting the config does not start training.

## Configuration

`word_ctc_args.yaml` defines a 1,000-batch pilot with batch size 8. Paths are
relative to `model_training`; change dataset/device/output paths for your machine.
The model architecture and session order match focused-search trial 00015.
The supplied config initializes from scratch. To transfer the phoneme backbone,
extract that run's `checkpoint/best_checkpoint.zip` and set
`backbone_checkpoint_path` to the extracted checkpoint file. Only `out.weight`
and `out.bias` are excluded; incompatible backbone dimensions fail explicitly.
Optimizer and scheduler states are fresh. Do not combine backbone initialization
with `init_from_checkpoint`.

When you choose to launch training later:

```sh
cd model_training
python train_model.py --config word_ctc_args.yaml
```

The vocabulary is built only from selected training trials. Text is lowercased;
letters, digits, and internal apostrophes are retained. Punctuation separates
words. IDs 0 and 1 are blank and unknown. `word_min_count` defaults to 1. Words
outside the vocabulary cannot be generated; validation WER uses original
normalized words, rather than replacing reference words with unknown tokens.
Unknown-word rate is reported separately. Vocabulary size controls output-layer
size and can substantially increase logits memory; batch size 8 is a starting
point, not a measured memory guarantee.

The checkpoint folder contains `word_vocab.json`, `args.yaml`, and the best
checkpoint selected by WER. Checkpoints embed their vocabulary and reject resume
with a different vocabulary or target type. To resume, set `init_from_checkpoint`
and `init_checkpoint_path`, provide the original `word_vocab_path`, disable
`backbone_checkpoint_path`, and choose a new output directory. Existing resume
behavior restarts the configured batch loop rather than restoring a batch cursor.

Word-mode TensorBoard tags use WER. Validation metrics include `avg_WER`,
`unknown_word_rate`, and up to ten reference/prediction examples. Some internal
fields (`avg_PER`, `val_PER`, `phone_seq_lens`) retain legacy names for compatibility;
in word mode these contain word error rates or word lengths.

## Evaluation

```sh
cd model_training
python evaluate_model.py --model_path trained_models/word_ctc_pilot \
  --data_dir ../data/hdf5_data_final --eval_type val --device cpu \
  --predictions_path word_ctc_validation.json
```

Word evaluation uses greedy CTC decoding: collapse consecutive token repeats,
then remove blanks. It bypasses Redis and the external language-model process.
The JSON contains sentence predictions and references, aggregate WER, and
unknown-word rate. Unlabeled test data produces predictions without WER.
Word inference uses the same smoothing convention as training validation.

## Checks without training

`tests/test_word_ctc.py` covers normalization, vocabulary persistence,
blank-separated repeated words, word targets versus unchanged phoneme targets,
backbone transfer, a forward pass and finite CTC loss, scoring unknown reference
words correctly, and trainer initialization/validation/checkpoint roundtrip on
synthetic data. These tests never call training or perform optimizer steps.

```sh
python -m pytest model_training/tests/test_word_ctc.py
```
