import matplotlib.pyplot as plt
import numpy as np


def bar_compare(labels, values, title, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    return fig


def scatter_fairness_accuracy(acc_list, deo_list, labels, title):
    fig, ax = plt.subplots(figsize=(6, 5))

    for acc, deo, label in zip(acc_list, deo_list, labels):
        ax.scatter(deo, acc, label=label, s=60)

    ax.set_xlabel("DEOd (lower is better)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()

    plt.show()
    plt.close(fig)
