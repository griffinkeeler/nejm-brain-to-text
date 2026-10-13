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
