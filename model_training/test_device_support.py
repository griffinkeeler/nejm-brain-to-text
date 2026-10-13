"""Run with python -m unittest test_device_support from model_training."""
import unittest
from unittest.mock import patch
import torch
from device_utils import select_device, decoder_ctc_loss
from rnn_model import temporal_patches, GRUDecoder, TCNDecoder


class DeviceSupportTests(unittest.TestCase):
    def test_auto_selection_and_explicit_unavailable(self):
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=True):
            self.assertEqual(select_device().type, 'mps')
            self.assertEqual(select_device('auto', -1).type, 'cpu')
            self.assertEqual(select_device('cpu').type, 'cpu')
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1):
            self.assertEqual(str(select_device('auto', 1)), 'cuda:0')
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            self.assertEqual(select_device().type, 'cpu')
            with self.assertRaises(RuntimeError):
                select_device('mps')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'Apple GPU unavailable')
    def test_overlapping_patch_values_and_gradients(self):
        for size, stride in [(14, 4), (5, 5)]:
            cpu = torch.randn(2, 80, 8, requires_grad=True)
            gpu = cpu.detach().to('mps').requires_grad_()
            expected = temporal_patches(cpu, size, stride)
            actual = temporal_patches(gpu, size, stride)
            torch.testing.assert_close(actual.cpu(), expected)
            expected.square().sum().backward()
            actual.square().sum().backward()
            torch.testing.assert_close(gpu.grad.cpu(), cpu.grad)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'Apple GPU unavailable')
    def test_decoders_ctc_backward(self):
        for cls, kw in [(GRUDecoder, dict(n_units=16, n_layers=2, patch_size=14, patch_stride=4)),
                        (TCNDecoder, dict(hidden_dim=16, tcn_num_blocks=2))]:
            model = cls(neural_dim=8, n_days=1, n_classes=5, **kw).to('mps')
            logits = model(torch.randn(2, 80, 8, device='mps'), torch.tensor([0, 0], device='mps'))
            lengths = model.compute_output_lengths(torch.tensor([80, 70], device='mps'))
            targets = torch.tensor([[1, 2, 3], [2, 3, 1]], device='mps')
            loss = decoder_ctc_loss(torch.nn.CTCLoss(reduction='none'), logits,
                                    targets, lengths, torch.tensor([3, 3], device='mps')).mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
