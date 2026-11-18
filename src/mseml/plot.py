import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    with open("results/results.json") as f:
        results = json.load(f)

    losses = results["training_losses"]
    epochs = range(1, len(losses) + 1)

    Path("results").mkdir(exist_ok=True)

    plt.figure()
    plt.plot(epochs, losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE")
    plt.title("1D CNN Training Curve")
    plt.tight_layout()
    plt.savefig("results/training_curve.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
