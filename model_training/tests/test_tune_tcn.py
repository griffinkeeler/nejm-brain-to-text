import unittest

from omegaconf import OmegaConf
from tune_tcn import trial_config
from model_config import resolve_model_config


class FakeTrial:
    number = 7

    def suggest_categorical(self, name, choices):
        return choices[-1]

    def suggest_int(self, name, low, high):
        return high

    def suggest_float(self, name, low, high, **kwargs):
        return low


class TuningTests(unittest.TestCase):
    def test_search_preserves_alignment_split_and_base(self):
        base = OmegaConf.load('rnn_args.yaml')
        original = OmegaConf.to_container(base, resolve=True)
        args = trial_config(base, FakeTrial(), '/tmp/tcn_test')
        resolved = resolve_model_config(args.model)
        self.assertEqual(resolved['tcn_hidden_dim'], 384)
        self.assertEqual(resolved['tcn_num_blocks'], 8)
        self.assertEqual(args.model.tcn.patch_stride, base.model.tcn.patch_stride)
        self.assertEqual(args.model.tcn.patch_size, base.model.tcn.patch_size)
        self.assertEqual(args.dataset, base.dataset)
        self.assertEqual(args.seed, base.seed)
        self.assertEqual(OmegaConf.to_container(base, resolve=True), original)
        self.assertFalse(args.init_from_checkpoint)
        self.assertTrue(args.save_best_checkpoint)
        self.assertTrue(args.output_dir.endswith('trial_00007'))
        for suffix in ('', '_day'):
            self.assertAlmostEqual(
                args[f'lr_min{suffix}'] / args[f'lr_max{suffix}'],
                base[f'lr_min{suffix}'] / base[f'lr_max{suffix}'])


if __name__ == '__main__':
    unittest.main()
