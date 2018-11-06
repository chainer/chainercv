import mock
import unittest

from chainer import testing
from chainercv.chainer_experimental.training.extensions import ManualScheduler


def schedule(updater):
    return updater.iteration % 5


class TestManualScheduler(unittest.TestCase):

    def test_manual_scheduler(self):
        extension = ManualScheduler('x', schedule)

        trainer = testing.get_trainer_with_mock_updater(
            iter_per_epoch=10, extensions=[extension])
        trainer.updater.get_optimizer.return_value = mock.MagicMock()
        trainer.updater.get_optimizer().x = 0

        for i in range(100):
            self.trainer.updater.update()
            self.assertEqual(trainer.updater.get_optimizer().x, i % 5)


testing.run_module(__name__, __file__)
