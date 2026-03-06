"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import json
import numpy as np
import wandb
from utils.data_loader import load_dataset
from ann.neural_network import NeuralNetwork
from ann.optimizers import Optimizer
from ann.objective_functions import ce_loss, mse_loss, ce_der, mse_der


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                    choices=['mnist', 'fashion_mnist'])

    parser.add_argument('-e', '--epochs', type=int, default=10)

    parser.add_argument('-b', '--batch_size', type=int, default=32)

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)

    parser.add_argument('-o', '--optimizer', type=str, default='sgd',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop'])

    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128])

    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['relu', 'sigmoid', 'tanh'])

    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'mse'])

    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier',
                        choices=['random', 'xavier', 'zeros'])

    parser.add_argument('-w_p', '--wandb_project', type=str, default='my_project')

    parser.add_argument('--model_save_path', type=str, default='best_model.npy')

    parser.add_argument('--config_save_path', type=str, default='best_config.json')

    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()
    wandb.init(project=args.wandb_project, config=vars(args))
    config = wandb.config
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(args.dataset)

    model = NeuralNetwork(args)
    optimizer = Optimizer(
        name=args.optimizer,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    best_val_acc = -1.0

    iteration=0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        batch_count = 0
        epoch_grad_norm = 0.0
        n = X_train.shape[0]
        indices = np.random.permutation(n)

        for start in range(0, n, args.batch_size):
            end = start + args.batch_size
            batch_idx = indices[start:end]

            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            logits = model.forward(X_batch)

            if args.loss == "cross_entropy":
                loss = ce_loss(y_batch, logits)
                grad = ce_der(y_batch, logits)
            else:
                loss = mse_loss(y_batch, logits)
                grad = mse_der(y_batch, logits)

            model.backward(grad)
            if iteration < 50:
                grad_log = {}
                for j in range(5):
                    grad_log[f"neuron_{j}_grad_norm"] = np.linalg.norm(model.layers[0].grad_W[:, j])

                wandb.log({
                    "iteration": iteration,
                    **grad_log
                })

            iteration += 1
            epoch_grad_norm += np.linalg.norm(model.layers[0].grad_W)
            optimizer.step(model.layers)

            epoch_loss += loss
            batch_count += 1

        epoch_loss /= batch_count
        epoch_grad_norm /= batch_count
        train_acc = model.evaluate(X_train, y_train)
        val_acc = model.evaluate(X_val, y_val)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )
        wandb.log({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "first_layer_grad_norm": epoch_grad_norm
        })

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.get_weights()
            np.save(args.model_save_path, best_weights)

            config_dict = vars(args)
            with open(args.config_save_path, "w") as f:
                json.dump(config_dict, f, indent=4)

    # Final test evaluation only once after training
    best_weights = np.load(args.model_save_path, allow_pickle=True).item()
    model.set_weights(best_weights)

    test_acc = model.evaluate(X_test, y_test)
    print(f"Final Test Acc: {test_acc:.4f}")
    wandb.log({"final_test_acc": test_acc})
    wandb.finish()
    print("Training complete!")


if __name__ == '__main__':
    main()