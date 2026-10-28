"""Checkpoint alignment and scalar preservation for resumed runs."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('merge', Path(__file__).parents[1] / 'merge_tensorboard_runs.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MergeTests(unittest.TestCase):
    def test_checkpoint_trimming_and_time_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / 'first', root / 'second'
            first.mkdir()
            second.mkdir()
            (first / 'training_log').write_text(
                '2026-10-26 10:00:00,000: Val batch 1: PER (avg): 0.2\n'
                '2026-10-26 10:00:01,000: Saved model to checkpoint: /remote/first/checkpoint/best_checkpoint\n')
            (second / 'training_log').write_text(
                '2026-10-26 11:00:00,000: Loaded model from checkpoint: /remote/first/checkpoint/best_checkpoint\n')
            for run in (first, second):
                writer = module.EventFileWriter(str(run / 'tensorboard'))
                for step in (1, 2, 3):
                    for tag, value, axis in [('train/ctc_loss', 10 / step, step),
                                             ('timing/elapsed_seconds', step * 10, step),
                                             ('convergence/per_by_elapsed_second', .2, step * 10)]:
                        writer.add_event(module.Event(wall_time=100 + step, step=axis,
                            summary=module.Summary(value=[module.Summary.Value(tag=tag, simple_value=value)])))
                writer.close()
            output = root / 'merged'
            manifest = module.merge([first, second], output)
            points = {}
            for path in output.glob('events.out.tfevents.*'):
                for event in module.LegacyEventFileLoader(str(path)).Load():
                    for value in event.summary.value:
                        points[value.tag, event.step] = value.simple_value
            self.assertEqual(manifest[1]['step_offset'], 2)
            self.assertEqual(sorted(step for tag, step in points if tag == 'train/ctc_loss'), [1, 2, 3, 4, 5])
            self.assertEqual(points['train/ctc_loss', 3], 10)
            self.assertEqual(points['timing/elapsed_seconds', 5], 50)
            self.assertIn(('convergence/per_by_elapsed_second', 50), points)
            self.assertEqual(len(points), 15)
            with self.assertRaises(FileExistsError):
                module.merge([first, second], output)


if __name__ == '__main__':
    unittest.main()
