"""Shared device selection and portable CTC loss for decoder training."""
import torch


def select_device(requested='auto', gpu_number=0):
    requested = str(requested).lower()
    if requested not in ('auto', 'cpu', 'cuda', 'mps'):
        raise ValueError('device must be auto, cpu, cuda, or mps')
    if requested == 'cpu' or (requested == 'auto' and int(gpu_number) < 0):
        return torch.device('cpu')
    if requested == 'cuda' or (requested == 'auto' and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        index = int(gpu_number)
        if index < 0 or index >= torch.cuda.device_count():
            index = 0
        return torch.device(f'cuda:{index}')
    if requested == 'mps' or (requested == 'auto' and torch.backends.mps.is_available()):
        if not torch.backends.mps.is_available():
            raise RuntimeError('Apple MPS requested but unavailable in this Python process')
        return torch.device('mps')
    return torch.device('cpu')


def decoder_ctc_loss(criterion, logits, targets, input_lengths, target_lengths):
    # CTC needs a blank between adjacent repeated labels as well as one
    # output position per target label. Reject impossible alignments before
    # infinite loss can contaminate gradients; never silently drop trials.
    positions = torch.arange(targets.shape[1] - 1, device=targets.device)
    repeated = ((targets[:, 1:] == targets[:, :-1]) &
                (positions[None, :] < target_lengths[:, None] - 1)).sum(dim=1)
    required = target_lengths + repeated
    invalid = input_lengths < required
    if invalid.any().item():
        rows = invalid.nonzero().flatten().tolist()
        available = input_lengths[invalid].tolist()
        needed = required[invalid].tolist()
        raise ValueError(
            f'Impossible CTC alignment at batch rows {rows}: output lengths '
            f'{available}, required lengths {needed} (including repeated labels). '
            'Reduce the decoder patch stride or inspect the trial labels; '
            'do not disable the non-finite gradient check.')

    # PyTorch 2.5 lacks MPS CTC. This explicit CPU path retains the autograd
    # connection to the GPU decoder and avoids a global implicit fallback.
    log_probs = logits.float().log_softmax(2).permute(1, 0, 2)
    if logits.device.type == 'mps':
        return criterion(log_probs.cpu(), targets.cpu(), input_lengths.cpu(),
                         target_lengths.cpu())
    return criterion(log_probs, targets, input_lengths, target_lengths)


def synchronize_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()
