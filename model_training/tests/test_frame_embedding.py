"""Run with python -m unittest tests.test_frame_embedding from model_training."""
import unittest

import torch

from model_config import resolve_model_config
from rnn_model import GRUDecoder, TCNDecoder


class FrameEmbeddingTests(unittest.TestCase):
    def make_model(self, **kwargs):
        return TCNDecoder(neural_dim=8, n_days=2, n_classes=5,
                          hidden_dim=16, tcn_num_blocks=2,
                          patch_size=5, patch_stride=4, **kwargs)

    def test_shapes_lengths_and_gradient_flow(self):
        devices = ['cpu']
        if torch.backends.mps.is_available():
            devices.append('mps')
        for device in devices:
            for width in (4, 16):
                with self.subTest(device=device, width=width):
                    model = self.make_model(frame_embed_dim=width).to(device)
                    x = torch.randn(2, 25, 8, device=device, requires_grad=True)
                    logits = model(x, torch.tensor([0, 1], device=device))
                    self.assertEqual(logits.shape, (2, 6, 5))
                    lengths = model.compute_output_lengths(
                        torch.tensor([25, 21], device=device))
                    self.assertEqual(lengths.tolist(), [6, 5])
                    logits.square().mean().backward()
                    for parameter in (model.frame_embed[1].weight,
                                      model.patch_embed[1].weight):
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(parameter.grad.abs().sum().item(), 0)
                    self.assertTrue(torch.isfinite(x.grad).all())

    def test_baseline_checkpoint_compatibility(self):
        baseline = self.make_model().eval()
        self.assertIsNone(baseline.frame_embed)
        self.assertFalse(any(key.startswith('frame_embed.')
                             for key in baseline.state_dict()))
        restored = self.make_model(frame_embed_dim=None).eval()
        restored.load_state_dict(baseline.state_dict(), strict=True)
        x = torch.randn(2, 25, 8)
        days = torch.tensor([0, 1])
        torch.testing.assert_close(restored(x, days), baseline(x, days))

    def test_config_resolution(self):
        config = {'model_type': 'tcn', 'tcn_frame_embed_dim': 16,
                  'tcn': {'frame_embed_dim': 4}}
        self.assertEqual(resolve_model_config(config)['tcn_frame_embed_dim'], 4)
        self.assertEqual(config['tcn_frame_embed_dim'], 16)
        config['tcn']['frame_embed_dim'] = None
        self.assertIsNone(resolve_model_config(config)['tcn_frame_embed_dim'])
        self.assertNotIn('tcn_frame_embed_dim', resolve_model_config(
            {'model_type': 'tcn'}))

    def test_invalid_width(self):
        for width in (0, -1, True, 1.5):
            with self.subTest(width=width), self.assertRaises(ValueError):
                self.make_model(frame_embed_dim=width)


class GRUFrameEmbeddingTests(unittest.TestCase):
    def make_model(self, **kwargs):
        return GRUDecoder(neural_dim=8, n_days=2, n_classes=5,
                          n_units=16, n_layers=2, **kwargs)

    def test_ctc_backward_and_state_with_and_without_patches(self):
        for size, stride in ((14, 4), (0, 0)):
            with self.subTest(size=size):
                model = self.make_model(frame_embed_dim=4,
                                        patch_size=size, patch_stride=stride)
                x = torch.randn(2, 30, 8, requires_grad=True)
                days = torch.tensor([0, 1])
                logits, state = model(x, days, return_state=True)
                expected_time = 5 if size else 30
                self.assertEqual(logits.shape, (2, expected_time, 5))
                self.assertEqual(state.shape, (2, 2, 16))
                self.assertEqual(model.gru.input_size, 16)
                lengths = model.compute_output_lengths(torch.tensor([30, 26]))
                self.assertEqual(lengths.tolist(), [5, 4] if size else [30, 26])
                loss = torch.nn.CTCLoss()(
                    logits.log_softmax(-1).transpose(0, 1),
                    torch.tensor([[1, 2], [2, 3]]), lengths,
                    torch.tensor([2, 2]))
                loss.backward()
                for parameter in (model.frame_embed[1].weight,
                                  model.patch_embed[1].weight,
                                  model.gru.weight_ih_l0):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
                self.assertTrue(torch.isfinite(x.grad).all())
                continued, _ = model(x, days, states=state.detach(), return_state=True)
                self.assertEqual(continued.shape, logits.shape)

    def test_baseline_checkpoint_compatibility(self):
        baseline = self.make_model(patch_size=14, patch_stride=4).eval()
        self.assertEqual(baseline.gru.input_size, 112)
        self.assertFalse(any(key.startswith(('frame_embed.', 'patch_embed.'))
                             for key in baseline.state_dict()))
        restored = self.make_model(patch_size=14, patch_stride=4,
                                   frame_embed_dim=None).eval()
        restored.load_state_dict(baseline.state_dict(), strict=True)
        x, days = torch.randn(2, 30, 8), torch.tensor([0, 1])
        torch.testing.assert_close(restored(x, days), baseline(x, days))

    def test_config_resolution_and_invalid_width(self):
        config = {'model_type': 'gru', 'gru_frame_embed_dim': 16,
                  'gru': {'frame_embed_dim': 4},
                  'tcn': {'frame_embed_dim': 32}}
        self.assertEqual(resolve_model_config(config)['gru_frame_embed_dim'], 4)
        self.assertEqual(config['gru_frame_embed_dim'], 16)
        config['gru']['frame_embed_dim'] = None
        self.assertIsNone(resolve_model_config(config)['gru_frame_embed_dim'])
        self.assertNotIn('gru_frame_embed_dim', resolve_model_config({}))
        for width in (0, -1, True, 1.5):
            with self.subTest(width=width), self.assertRaises(ValueError):
                self.make_model(frame_embed_dim=width)


if __name__ == '__main__':
    unittest.main()
