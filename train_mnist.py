"""
MNIST Training & Visualization
================================
Trains a 784-128-10 neural network on MNIST entirely from scratch in NumPy.
Generates four result plots and saves the trained model.

Run:
    python train_mnist.py

Outputs in results/:
    mnist_training_curves.png    — loss + test accuracy per epoch
    mnist_weight_maps.png        — hidden unit weights as 28x28 feature detectors
    mnist_confusion_matrix.png   — 10x10 confusion matrix on the test set
    mnist_sample_predictions.png — sample test images with correct/wrong labels

Model saved to:
    model/mnist_model.npz
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from deep_neural_network import DeepNeuralNetwork
from mnist_loader import load_mnist

# ── Hyperparameters ──────────────────────────────────────────────────────────
LAYER_SIZES    = [784, 128, 10]    # 784 inputs -> 128 hidden (ReLU) -> 10 outputs (softmax)
EPOCHS         = 30
LEARNING_RATE  = 0.001             # Adam default
BATCH_SIZE     = 256
SAVE_DIR       = 'results'
MODEL_PATH     = 'model/mnist_dnn_model.npz'
HIDDEN_SIZE    = LAYER_SIZES[1]    # kept for plot titles


# ── Plotting helpers ─────────────────────────────────────────────────────────

def plot_training_curves(loss_history, val_acc_history, save_dir):
    epochs = range(1, len(loss_history) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(
        f'MNIST Training — 784-{HIDDEN_SIZE}-10 Network (NumPy, from scratch)',
        fontsize=13, fontweight='bold'
    )

    ax1.plot(epochs, loss_history, color='#4facfe', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('MSE Loss (avg over mini-batches)', fontsize=11)
    ax1.set_title('Training Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_acc_history, color='#22c55e', linewidth=2)
    ax2.axhline(y=val_acc_history[-1], color='red', linestyle='--', alpha=0.5,
                label=f'Final: {val_acc_history[-1]:.2f}%')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax2.set_title('Test Set Accuracy per Epoch', fontsize=12)
    ax2.set_ylim([max(50, min(val_acc_history) - 5), 100])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'mnist_training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_weight_maps(W1, save_dir, n_show=32):
    """Each hidden neuron's 784 weights reshaped to 28x28 reveal digit-stroke detectors."""
    n_cols = 8
    n_rows = (n_show + n_cols - 1) // n_cols
    n_show = min(n_show, W1.shape[1])
    vmax = np.percentile(np.abs(W1[:, :n_show]), 99)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.7, n_rows * 1.7))
    fig.suptitle(
        'What Each Hidden Neuron Learned — Weights Reshaped to 28×28\n'
        '(Red = positive weight, Blue = negative weight)',
        fontsize=12, fontweight='bold'
    )

    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols]
        if i < n_show:
            ax.imshow(W1[:, i].reshape(28, 28), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f'#{i + 1}', fontsize=7)
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(save_dir, 'mnist_weight_maps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, save_dir):
    n = 10
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    overall_acc = cm.diagonal().sum() / cm.sum() * 100
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(n))
    ax.set_yticklabels(range(n))
    ax.set_xlabel('Predicted Digit', fontsize=12)
    ax.set_ylabel('True Digit', fontsize=12)
    ax.set_title(
        f'Confusion Matrix — MNIST Test Set (10,000 samples)\nOverall Accuracy: {overall_acc:.2f}%',
        fontsize=12, fontweight='bold'
    )

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8,
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    path = os.path.join(save_dir, 'mnist_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    return overall_acc, per_class_acc


def plot_sample_predictions(X_test, y_true, y_pred, save_dir, n=25):
    """5x5 grid of test images — green title = correct, red = wrong."""
    wrong = np.where(y_pred != y_true)[0]
    right = np.where(y_pred == y_true)[0]
    # Show ~5 wrong + ~20 right to make mistakes visible
    idx = np.concatenate([wrong[:5], right[:20]])[:n]

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    fig.suptitle('Sample Test Predictions (green = correct, red = wrong)',
                 fontsize=13, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i >= len(idx):
            ax.axis('off')
            continue
        j = idx[i]
        ax.imshow(X_test[j].reshape(28, 28), cmap='gray', vmin=-1, vmax=1)
        color = 'red' if y_pred[j] != y_true[j] else 'green'
        ax.set_title(f'T:{y_true[j]}  P:{y_pred[j]}', color=color, fontsize=9, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(save_dir, 'mnist_sample_predictions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs('model', exist_ok=True)

    print("=" * 65)
    print("  MNIST - From-Scratch Neural Network (NumPy only)")
    print(f"  Architecture : 784 -> {HIDDEN_SIZE} -> 10")
    print(f"  Optimizer    : Mini-batch SGD  |  lr={LEARNING_RATE}  |  batch={BATCH_SIZE}")
    print(f"  Epochs       : {EPOCHS}")
    print("=" * 65)

    X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = load_mnist()

    nn = DeepNeuralNetwork(
        LAYER_SIZES,
        activation='relu',
        loss='cross_entropy',
        optimizer='adam',
        learning_rate=LEARNING_RATE,
    )
    nn.summary()
    print()

    nn.train(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        X_val=X_test,
        y_val=y_test,
        verbose=True,
        print_every=1,
    )

    train_acc = nn.get_accuracy(X_train, y_train)
    test_acc  = nn.get_accuracy(X_test, y_test)

    print(f"\nFinal Train Accuracy : {train_acc:.2f}%")
    print(f"Final Test  Accuracy : {test_acc:.2f}%")

    nn.save(MODEL_PATH)

    print("\nGenerating visualizations...")
    plot_training_curves(nn.loss_history, nn.val_accuracy_history, SAVE_DIR)
    plot_weight_maps(nn.weights[0], SAVE_DIR, n_show=32)

    y_pred, _ = nn.predict(X_test)
    overall_acc, per_class_acc = plot_confusion_matrix(y_test_labels, y_pred, SAVE_DIR)
    plot_sample_predictions(X_test, y_test_labels, y_pred, SAVE_DIR)

    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("-" * 65)
    print(f"  Test Accuracy  : {test_acc:.2f}%")
    print(f"  Train Accuracy : {train_acc:.2f}%")
    print(f"  Parameters     : {sum(W.size + b.size for W, b in zip(nn.weights, nn.biases)):,}")
    print("\n  Per-class accuracy on test set:")
    for i, acc in enumerate(per_class_acc):
        bar = '#' * int(acc / 5)
        print(f"    Digit {i}: {acc:5.1f}%  {bar}")
    print("=" * 65)
    print(f"\nAll results saved to  {SAVE_DIR}/")
    print(f"Model saved to        {MODEL_PATH}")


if __name__ == '__main__':
    main()
