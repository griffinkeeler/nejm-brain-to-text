import unittest
import torch
from rnn_model import affine_scan, GILRMemory, TCNDecoder
from model_config import resolve_model_config


class GILRTests(unittest.TestCase):
    def test_scan_outputs_and_gradients(self):
        for time in (1, 7, 16):
            a = torch.rand(2, time, 4, dtype=torch.float64, requires_grad=True)
            b = torch.randn_like(a, requires_grad=True)
            state = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
            h, reference = state, []
            for t in range(time):
                h = a[:, t] * h + b[:, t]
                reference.append(h)
            reference = torch.stack(reference, dim=1)
            actual = affine_scan(a, b, state)
            torch.testing.assert_close(actual, reference)
            weight = torch.randn_like(actual)
            expected_grads = torch.autograd.grad((reference * weight).sum(), (a, b, state))
            actual_grads = torch.autograd.grad((actual * weight).sum(), (a, b, state))
            for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                torch.testing.assert_close(actual_grad, expected_grad)

    def test_memory_padding_chunks_and_causality(self):
        memory = GILRMemory(8).eval()
        x = torch.randn(2, 13, 8)
        initial = torch.randn(2, 8)
        full, final = memory(x, initial)
        first, state = memory(x[:, :5], initial)
        second, chunk_final = memory(x[:, 5:], state)
        torch.testing.assert_close(torch.cat([first, second], 1), full)
        torch.testing.assert_close(chunk_final, final)
        changed = x.clone()
        changed[:, 5:] += 10
        torch.testing.assert_close(memory(changed, initial)[0][:, :5], full[:, :5])
        _, padded_state = memory(x, initial, torch.tensor([5, 13]))
        torch.testing.assert_close(padded_state[0], state[0])
        torch.testing.assert_close(padded_state[1], final[1])

    def test_decoder_ctc_and_disabled_checkpoint(self):
        kwargs = dict(neural_dim=8, n_days=1, n_classes=5, hidden_dim=16,
                      tcn_num_blocks=2, patch_stride=4, frame_embed_dim=4,
                      residual_fusion=True, use_glu=True)
        model = TCNDecoder(**kwargs, use_gilr=True)
        logits, state = model(torch.randn(2, 25, 8), torch.tensor([0, 0]),
                              return_state=True, input_lengths=torch.tensor([25, 21]))
        self.assertEqual(logits.shape, (2, 6, 5))
        self.assertEqual(state.shape, (2, 16))
        loss = torch.nn.CTCLoss()(logits.log_softmax(-1).transpose(0, 1),
                                  torch.tensor([[1, 2], [2, 3]]),
                                  torch.tensor([6, 5]), torch.tensor([2, 2]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        baseline = TCNDecoder(**kwargs)
        restored = TCNDecoder(**kwargs, use_gilr=False)
        restored.load_state_dict(baseline.state_dict(), strict=True)
        self.assertIsNone(restored.memory)
        self.assertTrue(resolve_model_config({'model_type': 'tcn',
                         'tcn': {'use_gilr': True}})['tcn_use_gilr'])


if __name__ == '__main__':
    unittest.main()
