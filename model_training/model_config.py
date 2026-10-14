"""Resolve grouped architecture settings while accepting legacy checkpoints."""


def resolve_model_config(model):
    """Return the selected architecture in the legacy decoder argument format.

    Grouped values take precedence over legacy flat values. The source config
    is not mutated, so saved args retain the user's grouped configuration.
    """
    resolved = dict(model)
    model_type = str(model.get('model_type', 'gru')).lower()
    if model_type not in ('gru', 'tcn'):
        raise ValueError(f'Invalid model type: {model_type}')
    resolved['model_type'] = model_type
    mappings = {
        'gru': {
            'n_units': 'n_units', 'n_layers': 'n_layers',
            'dropout': 'rnn_dropout', 'trainable': 'rnn_trainable',
            'patch_size': 'patch_size', 'patch_stride': 'patch_stride',
        },
        'tcn': {
            'hidden_dim': 'tcn_hidden_dim', 'num_blocks': 'tcn_num_blocks',
            'kernel_size': 'kernel_size', 'dropout': 'tcn_dropout',
            'expansion_factor': 'tcn_expansion_factor',
            'trainable': 'rnn_trainable', 'patch_size': 'tcn_patch_size',
            'patch_stride': 'tcn_patch_stride',
        },
    }
    group = model.get(model_type, {})
    for key, legacy_key in mappings[model_type].items():
        if key in group:
            resolved[legacy_key] = group[key]
    return resolved
