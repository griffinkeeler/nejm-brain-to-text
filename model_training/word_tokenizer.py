"""Deterministic whole-word targets and CTC decoding (no language model)."""
import json
import re
from collections import Counter
from pathlib import Path


def transcript_text(values):
    if isinstance(values, str):
        return values
    return ''.join(chr(int(v)) for v in values if int(v) != 0)


def words(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text.lower().replace('’', "'"))


class WordTokenizer:
    def __init__(self, tokens):
        if tokens[:2] != ['<blank>', '<unk>'] or len(tokens) != len(set(tokens)):
            raise ValueError('Vocabulary must have unique tokens, beginning with blank and unk')
        self.tokens = tokens
        self.ids = {word: i for i, word in enumerate(tokens)}

    @classmethod
    def build(cls, transcripts, min_count=1):
        if min_count < 1:
            raise ValueError('min_count must be positive')
        counts = Counter(w for text in transcripts for w in words(text))
        tokens = sorted(w for w, n in counts.items() if n >= min_count)
        if not tokens:
            raise ValueError('No training words available for vocabulary')
        return cls(['<blank>', '<unk>'] + tokens)

    def encode(self, text):
        return [self.ids.get(w, 1) for w in words(text)]

    def decode_ctc(self, ids):
        result, previous = [], None
        for value in ids:
            value = int(value)
            if value != previous and value != 0:
                result.append(self.tokens[value])
            previous = value
        return result

    def save(self, path):
        Path(path).write_text(json.dumps({'version': 1, 'normalization': 'lowercase_alphanumeric_apostrophe_v1', 'tokens': self.tokens}, indent=2))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data['version'] != 1 or data['normalization'] != 'lowercase_alphanumeric_apostrophe_v1':
            raise ValueError('Unsupported vocabulary format')
        return cls(data['tokens'])


def training_transcripts(trials):
    import h5py
    for day in trials.values():
        with h5py.File(day['session_path'], 'r') as source:
            for trial in day['trials']:
                yield transcript_text(source[f'trial_{trial:04d}']['transcription'][:])


def load_backbone(model, checkpoint):
    state = {k.removeprefix('module.').removeprefix('_orig_mod.'): v
             for k, v in checkpoint['model_state_dict'].items()}
    state = {k: v for k, v in state.items() if not k.startswith('out.')}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) != {'out.weight', 'out.bias'} or unexpected:
        raise ValueError(f'Incompatible backbone: missing={missing}, unexpected={unexpected}')
