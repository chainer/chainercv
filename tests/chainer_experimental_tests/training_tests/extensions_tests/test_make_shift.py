import mock
import unittest

from chainer import testing
from chainercv.chainer_experimental.training.extensions import make_shift


@make_shift('x')
def mod5_shift(trainer):
    return trainer.updater.iteration % 5


class TestMakeShift(unittest.TestCase):

    def test_make_shift(self):
        trainer = testing.get_trainer_with_mock_updater(
            iter_per_epoch=10, extensions=[mod5_shift])
        trainer.updater.get_optimizer.return_value = mock.MagicMock()
        trainer.updater.get_optimizer().x = -1

        mod5_shift.initialize(trainer)
        for i in range(100):
            self.assertEqual(trainer.updater.get_optimizer().x, i % 5)
            trainer.updater.update()
            mod5_shift(trainer)


testing.run_module(__name__, __file__)
