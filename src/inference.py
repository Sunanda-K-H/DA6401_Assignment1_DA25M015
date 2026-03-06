"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.utils.data_loader import load_dataset
from src.ann.neural_network import NeuralNetwork
from src.ann.objective_functions import ce_loss, mse_loss


def parse_arguments():
    """
    Parse command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument('--model_path', type=str, default='best_model.npy')
    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-b', '--batch_size', type=int, default=32)

    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128])
    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['relu', 'sigmoid', 'tanh'])
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier',
                        choices=['random', 'xavier'])
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'mse'])

    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model, X_test, y_test, batch_size=32, loss_name="cross_entropy"):
    """
    Evaluate model on test data.

    Returns Dictionary - logits, loss, accuracy, f1, precision, recall, y_true, y_pred
    """
    logits_list = []

    n = X_test.shape[0]
    for start in range(0, n, batch_size):
        end = start + batch_size
        X_batch = X_test[start:end]
        batch_logits = model.forward(X_batch)
        logits_list.append(batch_logits)

    logits = np.vstack(logits_list)

    if loss_name == "cross_entropy":
        loss = ce_loss(y_test, logits)
    else:
        loss = mse_loss(y_test, logits)

    y_pred = np.argmax(logits, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """
    Plot and save confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    plt.title("Confusion Matrix - Best Model on Test Set")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_misclassified_samples(X_test, y_true, y_pred,
                               save_path="misclassified_samples.png",
                               num_show=16):
    """
    Plot and save a grid of misclassified test samples.
    """
    mis_idx = np.where(y_true != y_pred)[0]

    if len(mis_idx) == 0:
        print("No misclassified samples found.")
        return

    num_show = min(num_show, len(mis_idx))
    chosen = mis_idx[:num_show]

    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten()

    for ax, idx in zip(axes, chosen):
        img = X_test[idx].reshape(28, 28)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"T:{y_true[idx]}  P:{y_pred[idx]}")
        ax.axis("off")

    # hide unused axes if fewer than 16
    for ax in axes[num_show:]:
        ax.axis("off")

    plt.suptitle("Misclassified Test Samples")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def main():
    """
    Main inference function.
    """
    args = parse_arguments()

    # load dataset; only test set is used here
    _, _, X_test, _, _, y_test = load_dataset(args.dataset)

    # rebuild model with same architecture settings
    model = NeuralNetwork(args)

    # load saved weights
    weights = load_model(args.model_path)
    model.set_weights(weights)

    # evaluate
    results = evaluate_model(
        model,
        X_test,
        y_test,
        batch_size=args.batch_size,
        loss_name=args.loss
    )

    # plots for Question 2.8
    plot_confusion_matrix(results["y_true"], results["y_pred"])
    plot_misclassified_samples(X_test, results["y_true"], results["y_pred"])

    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-score: {results['f1']:.4f}")
    print("Saved: confusion_matrix.png")
    print("Saved: misclassified_samples.png")

    print("Evaluation complete!")
    return results


if __name__ == '__main__':
    main()