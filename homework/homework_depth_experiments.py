from utils.experiment_utils import get_loaders, logger, make_table, save_results, set_seed, setup_logging, train_model
from utils.model_utils import FullyConnectedNet, count_parameters
from utils.visualization_utils import plot_histories, plot_lines

DEPTHS = [1, 2, 3, 5, 7]
WIDTH = 256
VARIANTS = {
    "plain": {},
    "dropout_bn": {"dropout": 0.3, "batch_norm": True},
}


def run_depth_experiment(dataset, variant):
    train_loader, test_loader, input_size = get_loaders(dataset)
    histories, params = {}, []
    for depth in DEPTHS:
        logger.info("%s / %s / глубина %d", dataset, variant, depth)
        set_seed()
        model = FullyConnectedNet(input_size, [WIDTH] * (depth - 1), **VARIANTS[variant])
        params.append(count_parameters(model))
        histories[f"глубина {depth}"] = train_model(model, train_loader, test_loader)
    table = make_table(histories, params=params)
    save_results("depth_experiments", f"{dataset}_{variant}", histories, table)
    plot_histories(histories, f"depth_{dataset}_{variant}.png", f"{dataset.upper()}, {variant}")
    return table


def analyze_depth(dataset):
    tables = {variant: run_depth_experiment(dataset, variant) for variant in VARIANTS}
    lines = {}
    for variant, table in tables.items():
        best = table.loc[table["test_acc"].idxmax()]
        logger.info("%s / %s: оптимальная %s, test acc %.4f", dataset, variant, best["model"], best["test_acc"])
        lines[f"{variant}: train"] = table["train_acc"].tolist()
        lines[f"{variant}: test"] = table["test_acc"].tolist()
    plot_lines(DEPTHS, lines, f"depth_{dataset}_summary.png", f"{dataset.upper()}: accuracy от глубины",
               "число слоёв", "accuracy после обучения")


if __name__ == "__main__":
    setup_logging("depth_experiments")
    for dataset in ["mnist", "cifar"]:
        analyze_depth(dataset)
