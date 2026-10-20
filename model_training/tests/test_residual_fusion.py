import unittest
from unittest.mock import patch

import torch

from model_config import resolve_model_config
from rnn_model import TCNDecoder


class ResidualFusionTests(unittest.TestCase):
    def make_model(self, **kwargs):
        return TCNDecoder(8, 1, 5, hidden_dim=16, tcn_num_blocks=2,
                          patch_stride=4, **kwargs)

    def test_ctc_gradients_shapes_and_raw_skip(self):
        model = self.make_model(frame_embed_dim=4, residual_fusion=True,
                                use_glu=True).eval()
        x, days = torch.randn(2, 25, 8), torch.tensor([0, 0])
        logits = model(x, days)
        self.assertEqual(logits.shape, (2, 6, 5))
        lengths = model.compute_output_lengths(torch.tensor([25, 21]))
        self.assertEqual(lengths.tolist(), [6, 5])
        loss = torch.nn.CTCLoss()(
            logits.log_softmax(-1).transpose(0, 1),
            torch.tensor([[1, 2], [2, 3]]), lengths, torch.tensor([2, 2]))
        loss.backward()
        for layer in (model.raw_patch_embed[1], model.frame_embed[1],
                      model.patch_embed[1], model.output_projection[0],
                      model.output_projection[2]):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        # A zero correction must send the raw embedding directly into the TCN.
        with patch.object(model.output_projection, 'forward',
                          side_effect=lambda z: z.new_zeros(z.shape[0], 16, z.shape[2])), \
             patch.object(model.raw_patch_embed, 'forward',
                          wraps=model.raw_patch_embed.forward) as raw, \
             patch.object(model.residual_blocks[0], 'forward',
                          wraps=model.residual_blocks[0].forward) as block:
            model(x, days)
            # Recompute the deterministic raw projection on its recorded input.
            expected = model.raw_patch_embed(raw.call_args.args[0]).transpose(1, 2)
            torch.testing.assert_close(block.call_args.args[0], expected)

    def test_disabled_compatibility_and_config(self):
        for width in (None, 4):
            original = self.make_model(frame_embed_dim=width).eval()
            restored = self.make_model(frame_embed_dim=width,
                                       residual_fusion=False).eval()
            restored.load_state_dict(original.state_dict(), strict=True)
            self.assertIsNone(restored.raw_patch_embed)
            x, days = torch.randn(2, 25, 8), torch.tensor([0, 0])
            torch.testing.assert_close(original(x, days), restored(x, days))
        with self.assertRaisesRegex(ValueError, 'requires frame_embed_dim'):
            self.make_model(residual_fusion=True)
        config = {'model_type': 'tcn', 'tcn_residual_fusion': False,
                  'tcn': {'residual_fusion': True}}
        self.assertTrue(resolve_model_config(config)['tcn_residual_fusion'])
        self.assertNotIn('tcn_residual_fusion', resolve_model_config({'model_type': 'tcn'}))


if __name__ == '__main__':
    unittest.main()
