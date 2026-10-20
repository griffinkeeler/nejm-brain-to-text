"""Run with python -m unittest tests.test_glu from model_training."""
import unittest

import torch

from model_config import resolve_model_config
from rnn_model import CompactTemporalBlock, TCNDecoder


class GLUTests(unittest.TestCase):
    def test_block_causality_and_both_gate_gradients(self):
        devices = ['cpu']
        if torch.backends.mps.is_available():
            devices.append('mps')
        for device in devices:
            with self.subTest(device=device):
                block = CompactTemporalBlock(8, dilation=2, use_glu=True).to(device)
                x = torch.randn(2, 8, 20, device=device, requires_grad=True)
                y = block(x)
                self.assertEqual(y.shape, x.shape)
                altered = x.detach().clone()
                altered[:, :, 10:] += 10
                torch.testing.assert_close(block(altered)[:, :, :10], y[:, :, :10])
                y[:, 0].square().mean().backward()
                grad = block.pointwise_in.weight.grad
                for half in grad.chunk(2, dim=0):
                    self.assertTrue(torch.isfinite(half).all())
                    self.assertGreater(half.abs().sum().item(), 0)

    def test_decoder_ctc_backward(self):
        for frame_width in (None, 4):
            with self.subTest(frame_width=frame_width):
                model = TCNDecoder(8, 1, 5, hidden_dim=16, tcn_num_blocks=2,
                                   patch_stride=4, use_glu=True,
                                   frame_embed_dim=frame_width)
                logits = model(torch.randn(2, 25, 8), torch.tensor([0, 0]))
                self.assertEqual(logits.shape, (2, 6, 5))
                lengths = model.compute_output_lengths(torch.tensor([25, 21]))
                self.assertEqual(lengths.tolist(), [6, 5])
                loss = torch.nn.CTCLoss()(
                    logits.log_softmax(-1).transpose(0, 1),
                    torch.tensor([[1, 2], [2, 3]]), lengths, torch.tensor([2, 2]))
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))

    def test_baseline_weights_and_config(self):
        baseline = CompactTemporalBlock(8).eval()
        restored = CompactTemporalBlock(8, use_glu=False).eval()
        self.assertEqual(baseline.pointwise_in.weight.shape, (16, 8, 1))
        restored.load_state_dict(baseline.state_dict(), strict=True)
        x = torch.randn(2, 8, 20)
        torch.testing.assert_close(baseline(x), restored(x))
        self.assertEqual(CompactTemporalBlock(8, use_glu=True).pointwise_in.weight.shape,
                         (32, 8, 1))
        config = {'model_type': 'tcn', 'tcn_use_glu': True,
                  'tcn': {'use_glu': False}}
        self.assertFalse(resolve_model_config(config)['tcn_use_glu'])
        self.assertTrue(config['tcn_use_glu'])
        self.assertNotIn('tcn_use_glu', resolve_model_config({'model_type': 'tcn'}))


if __name__ == '__main__':
    unittest.main()
