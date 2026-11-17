from utils.experiment_utils import get_loaders, logger, make_table, save_results, set_seed, setup_logging, train_model
from utils.model_utils import FullyConnectedNet, count_parameters
from utils.visualization_utils import plot_heatmap, plot_histories

WIDTHS = {
    "узкие": [64, 32, 16],
    "средние": [256, 128, 64],
    "широкие": [1024, 512, 256],
    "очень широкие": [2048, 1024, 512],
}
BASE_WIDTHS = [64, 128, 256, 512, 1024]
SCHEMES = {
    "расширение": lambda width: [width // 4, width // 2, width],
    "сужение": lambda width: [width, width // 2, width // 4],
    "постоянная": lambda width: [width, width, width],
}


def train_models(dataset, architectures):
    train_loader, test_loader, input_size = get_loaders(dataset)
    histories, params = {}, []
    for name, hidden_sizes in architectures.items():
        logger.info("%s / %s %s", dataset, name, hidden_sizes)
        set_seed()
        model = FullyConnectedNet(input_size, hidden_sizes)
        params.append(count_parameters(model))
        histories[name] = train_model(model, train_loader, test_loader)
    return histories, params


def run_width_comparison(dataset):
    architectures = {f"{name} {sizes}": sizes for name, sizes in WIDTHS.items()}
    histories, params = train_models(dataset, architectures)
    table = make_table(histories, params=params)
    save_results("width_experiments", f"{dataset}_widths", histories, table)
    plot_histories(histories, f"width_{dataset}.png", f"{dataset.upper()}, ширина")


def run_grid_search(dataset):
    architectures = {f"{scheme} {make(width)}": make(width) for scheme, make in SCHEMES.items() for width in BASE_WIDTHS}
    histories, params = train_models(dataset, architectures)
    table = make_table(histories, params=params)
    table["scheme"] = [scheme for scheme in SCHEMES for width in BASE_WIDTHS]
    table["base_width"] = BASE_WIDTHS * len(SCHEMES)
    save_results("width_experiments", f"{dataset}_grid_search", histories, table)
    best = table.loc[table["test_acc"].idxmax()]
    logger.info("%s: лучшая архитектура %s, test acc %.4f", dataset, best["model"], best["test_acc"])
    grid = table.pivot(index="scheme", columns="base_width", values="test_acc").loc[list(SCHEMES)]
    plot_heatmap(grid, f"width_{dataset}_heatmap.png", f"{dataset.upper()}: test accuracy",
                 "базовая ширина", "схема изменения ширины")


if __name__ == "__main__":
    setup_logging("width_experiments")
    for dataset in ["mnist", "cifar"]:
        run_width_comparison(dataset)
        run_grid_search(dataset)
