import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.experiment_utils import summarize, train_model
from utils.model_utils import FullyConnectedNet, count_parameters, layer_weights, set_dropout


def count_layers(model, layer_type):
    return sum(isinstance(module, layer_type) for module in model.modules())


def toy_loader():
    torch.manual_seed(0)
    x = torch.randn(400, 2)
    y = (x[:, 0] + x[:, 1] > 0).long()
    return DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)


class TestModel(unittest.TestCase):
    def test_depth(self):
        for hidden_sizes, depth in [([], 1), ([8], 2), ([8, 8], 3), ([8] * 6, 7)]:
            self.assertEqual(count_layers(FullyConnectedNet(4, hidden_sizes), nn.Linear), depth)

    def test_output_shape(self):
        model = FullyConnectedNet(28 * 28, [32, 16])
        self.assertEqual(model(torch.randn(5, 1, 28, 28)).shape, (5, 10))

    def test_count_parameters(self):
        model = FullyConnectedNet(4, [3], num_classes=2)
        self.assertEqual(count_parameters(model), (4 * 3 + 3) + (3 * 2 + 2))

    def test_regularization_layers(self):
        model = FullyConnectedNet(4, [8, 8, 8], dropout=0.5, batch_norm=True)
        self.assertEqual(count_layers(model, nn.Dropout), 3)
        self.assertEqual(count_layers(model, nn.BatchNorm1d), 3)
        model = FullyConnectedNet(4, [8, 8, 8], dropout=[0.5, 0, 0], batch_norm=[False, False, True])
        self.assertEqual(count_layers(model, nn.Dropout), 1)
        self.assertEqual(count_layers(model, nn.BatchNorm1d), 1)

    def test_set_dropout(self):
        model = FullyConnectedNet(4, [8, 8], dropout=0.5)
        set_dropout(model, 0.1)
        self.assertTrue(all(m.p == 0.1 for m in model.modules() if isinstance(m, nn.Dropout)))


class TestTraining(unittest.TestCase):
    def test_learns_simple_task(self):
        loader = toy_loader()
        history = train_model(FullyConnectedNet(2, [16], num_classes=2), loader, loader, epochs=10, lr=0.01)
        self.assertEqual(len(history["test_acc"]), 10)
        self.assertGreater(history["test_acc"][-1], 0.95)

    def test_weight_decay_makes_weights_smaller(self):
        loader = toy_loader()
        norms = []
        for weight_decay in (0.0, 0.1):
            torch.manual_seed(0)
            model = FullyConnectedNet(2, [16], num_classes=2)
            train_model(model, loader, loader, epochs=10, weight_decay=weight_decay)
            norms.append(torch.cat(layer_weights(model)).norm().item())
        self.assertLess(norms[1], norms[0])

    def test_dropout_schedule(self):
        loader = toy_loader()
        model = FullyConnectedNet(2, [16], num_classes=2, dropout=0.5)
        train_model(model, loader, loader, epochs=5, dropout_schedule=(0.5, 0.1))
        self.assertAlmostEqual(model.layers[3].p, 0.1)

    def test_summarize(self):
        history = {"train_acc": [0.6, 0.8, 0.9, 0.95], "test_acc": [0.55, 0.7, 0.72, 0.7],
                   "train_loss": [1.2, 0.6, 0.4, 0.2], "test_loss": [1.0, 0.8, 0.7, 0.75], "time": 2.0}
        result = summarize(history)
        self.assertEqual(result["overfit_epoch"], 3)
        self.assertEqual(result["best_test_acc"], 0.72)
        self.assertAlmostEqual(result["gap"], 0.25)


if __name__ == "__main__":
    unittest.main(verbosity=2)
