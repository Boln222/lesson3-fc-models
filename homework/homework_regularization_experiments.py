import torch

from utils.experiment_utils import get_loaders, logger, make_table, save_results, set_seed, setup_logging, train_model
from utils.model_utils import FullyConnectedNet, layer_weights
from utils.visualization_utils import plot_histories, plot_weight_histograms

DATASET = "cifar"
HIDDEN = [512, 256, 128]

BASIC = {
    "без регуляризации": ({}, {}),
    "Dropout 0.1": ({"dropout": 0.1}, {}),
    "Dropout 0.3": ({"dropout": 0.3}, {}),
    "Dropout 0.5": ({"dropout": 0.5}, {}),
    "BatchNorm": ({"batch_norm": True}, {}),
    "Dropout 0.3 + BatchNorm": ({"dropout": 0.3, "batch_norm": True}, {}),
    "L2 (weight decay 0.001)": ({}, {"weight_decay": 0.001}),
}
ADAPTIVE = {
    "без регуляризации": ({}, {}),
    "Dropout 0.5 -> 0.1": ({"dropout": 0.5}, {"dropout_schedule": (0.5, 0.1)}),
    "BatchNorm momentum 0.01": ({"batch_norm": True, "bn_momentum": 0.01}, {}),
    "BatchNorm momentum 0.1": ({"batch_norm": True, "bn_momentum": 0.1}, {}),
    "BatchNorm momentum 0.9": ({"batch_norm": True, "bn_momentum": 0.9}, {}),
    "BatchNorm + Dropout 0.5 -> 0.1 + L2": ({"batch_norm": True, "dropout": 0.5},
                                           {"dropout_schedule": (0.5, 0.1), "weight_decay": 0.001}),
}
LAYERS = {
    "без регуляризации": ({}, {}),
    "Dropout 0.5 во всех слоях": ({"dropout": 0.5}, {}),
    "Dropout 0.5 только в 1-м": ({"dropout": [0.5, 0, 0]}, {}),
    "Dropout 0.5 только в 3-м": ({"dropout": [0, 0, 0.5]}, {}),
    "BatchNorm во всех слоях": ({"batch_norm": True}, {}),
    "BatchNorm только в 1-м": ({"batch_norm": [True, False, False]}, {}),
    "BatchNorm только в 3-м": ({"batch_norm": [False, False, True]}, {}),
}


def run_configs(dataset, group, configs):
    train_loader, test_loader, input_size = get_loaders(dataset)
    histories, weights = {}, {}
    for name, (model_args, train_args) in configs.items():
        logger.info("%s / %s", dataset, name)
        set_seed()
        model = FullyConnectedNet(input_size, HIDDEN, **model_args)
        histories[name] = train_model(model, train_loader, test_loader, **train_args)
        weights[name] = layer_weights(model)
    weight_std = [[round(w.std().item(), 4) for w in layers] for layers in weights.values()]
    table = make_table(histories, weight_std=weight_std)
    save_results("regularization_experiments", f"{dataset}_{group}", histories, table)
    plot_histories(histories, f"regularization_{dataset}_{group}.png", f"{dataset.upper()}, {group}")
    return weights


def run_basic_regularization(dataset):
    weights = run_configs(dataset, "basic", BASIC)
    all_weights = {name: torch.cat(layers) for name, layers in weights.items()}
    plot_weight_histograms(all_weights, f"regularization_{dataset}_weights.png", "Распределение весов")


def run_adaptive_regularization(dataset):
    run_configs(dataset, "adaptive", ADAPTIVE)
    run_configs(dataset, "layers", LAYERS)


if __name__ == "__main__":
    setup_logging("regularization_experiments")
    run_basic_regularization(DATASET)
    run_adaptive_regularization(DATASET)
