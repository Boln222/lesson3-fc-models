import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils.experiment_utils import PLOTS_DIR

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"axes.prop_cycle": plt.cycler(color=COLORS), "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / name, dpi=120)
    plt.close(fig)


def plot_histories(histories, name, title):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for label, history in histories.items():
        epochs = range(1, len(history["train_acc"]) + 1)
        axes[0].plot(epochs, history["train_acc"], label=label)
        axes[1].plot(epochs, history["test_acc"], label=label)
        axes[2].plot(epochs, history["test_loss"], label=label)
    for ax, metric in zip(axes, ["train accuracy", "test accuracy", "test loss"]):
        ax.set(title=f"{title}: {metric}", xlabel="эпоха", ylabel=metric)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].legend(fontsize=8)
    save(fig, name)


def plot_lines(x, lines, name, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for label, y in lines.items():
        ax.plot(x, y, marker="o", label=label)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xticks=x)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    save(fig, name)


def plot_heatmap(table, name, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(table.values, cmap="Blues")
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.grid(False)
    middle = (table.values.min() + table.values.max()) / 2
    for i, row in enumerate(table.values):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.4f}", ha="center", va="center", color="white" if value > middle else "black")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.colorbar(image, label="test accuracy")
    save(fig, name)


def plot_weight_histograms(weights, name, title):
    columns = (len(weights) + 1) // 2
    fig, axes = plt.subplots(2, columns, figsize=(3.3 * columns, 6), sharex=True)
    for ax, (label, values) in zip(axes.flat, weights.items()):
        ax.hist(values, bins=100, range=(-0.1, 0.1), color=COLORS[0])
        ax.set_title(f"{label}\nstd = {values.std():.4f}", fontsize=9)
        ax.set_yticks([])
    for ax in axes.flat[len(weights):]:
        ax.axis("off")
    fig.suptitle(title)
    save(fig, name)
