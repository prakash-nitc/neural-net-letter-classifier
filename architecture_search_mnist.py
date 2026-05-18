"""
MNIST Architecture Search
==========================
Compares 784-X-10 networks for different hidden layer sizes X on MNIST.
Uses a proper train/test split throughout — accuracy is always measured on
the held-out test set, never the training set.

Run:
    python architecture_search_mnist.py

Outputs in results/:
    mnist_architecture_search.png  — accuracy vs. hidden size + convergence curves

Note: runtime ~5-8 minutes (6 architectures × 2 trials × 20 epochs each).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from deep_neural_network import DeepNeuralNetwork
from mnist_loader import load_mnist

# ── Config ────────────────────────────────────────────────────────────────────
HIDDEN_SIZES  = [16, 32, 64, 128, 256, 512]
EPOCHS        = 20
LEARNING_RATE = 0.001
BATCH_SIZE    = 256
N_TRIALS      = 2
SAVE_DIR      = 'results'


def run_search():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 65)
    print("  MNIST Architecture Search - 784-X-10 Networks")
    print(f"  Hidden sizes : {HIDDEN_SIZES}")
    print(f"  Epochs/trial : {EPOCHS}  |  Trials : {N_TRIALS}  |  Batch : {BATCH_SIZE}")
    print("=" * 65)

    X_train, X_test, y_train, y_test, _, _ = load_mnist()

    results = {}

    for h in HIDDEN_SIZES:
        n_params = (784 * h + h) + (h * 10 + 10)  # W1+b1 + W2+b2
        print(f"\n--- Hidden size: {h:4d}  ({n_params:,} parameters) ---")

        trial_accs = []
        best_val_history = None

        for trial in range(N_TRIALS):
            nn = DeepNeuralNetwork(
                [784, h, 10],
                activation='relu',
                loss='cross_entropy',
                optimizer='adam',
                learning_rate=LEARNING_RATE,
                seed=42 + trial,
            )
            nn.train(
                X_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                X_val=X_test,
                y_val=y_test,
                verbose=False,
            )
            # Use test accuracy — not training accuracy (that was the original bug)
            test_acc = nn.get_accuracy(X_test, y_test)
            trial_accs.append(test_acc)
            print(f"  Trial {trial + 1}: Test Acc = {test_acc:.2f}%")

            if best_val_history is None or test_acc >= max(trial_accs[:-1] or [0]):
                best_val_history = nn.val_accuracy_history[:]

        results[h] = {
            'mean':     np.mean(trial_accs),
            'std':      np.std(trial_accs),
            'best':     max(trial_accs),
            'n_params': n_params,
            'val_history': best_val_history,
        }
        print(f"  -> Mean: {results[h]['mean']:.2f}% +/- {results[h]['std']:.2f}%"
              f"   Best: {results[h]['best']:.2f}%")

    _plot(results, HIDDEN_SIZES)

    best_h = max(results, key=lambda h: results[h]['mean'])
    print(f"\n{'=' * 65}")
    print(f"  Best architecture : 784-{best_h}-10")
    print(f"  Mean test accuracy: {results[best_h]['mean']:.2f}% ± {results[best_h]['std']:.2f}%")
    print(f"  Parameters        : {results[best_h]['n_params']:,}")
    print("=" * 65)

    return results


def _plot(results, hidden_sizes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('MNIST Architecture Search: 784-X-10 Networks\n'
                 '(accuracy measured on held-out test set)',
                 fontsize=13, fontweight='bold')

    means    = [results[h]['mean']    for h in hidden_sizes]
    stds     = [results[h]['std']     for h in hidden_sizes]
    n_params = [results[h]['n_params'] for h in hidden_sizes]

    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(hidden_sizes)))
    ax1.bar(range(len(hidden_sizes)), means, yerr=stds, color=colors,
            edgecolor='black', linewidth=0.5, capsize=4, label='Mean test acc')
    ax1.scatter(range(len(hidden_sizes)),
                [results[h]['best'] for h in hidden_sizes],
                color='red', zorder=5, s=40, label='Best trial')

    ax1.set_xticks(range(len(hidden_sizes)))
    ax1.set_xticklabels([str(h) for h in hidden_sizes])
    ax1.set_xlabel('Hidden Layer Size (X)', fontsize=11)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax1.set_title('Accuracy vs. Architecture', fontsize=12)
    ax1.set_ylim([max(80, min(means) - 5), 100])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.text(i, m + s + 0.15, f'{m:.1f}%', ha='center', fontsize=8)

    # Parameter count on secondary axis
    ax1b = ax1.twinx()
    ax1b.plot(range(len(hidden_sizes)), [p / 1000 for p in n_params],
              'o--', color='orange', label='Params (K)')
    ax1b.set_ylabel('Parameters (thousands)', fontsize=10, color='orange')
    ax1b.tick_params(axis='y', labelcolor='orange')
    ax1b.legend(loc='lower right', fontsize=9)

    # Test accuracy convergence curves
    for h in hidden_sizes:
        hist = results[h]['val_history']
        if hist:
            ax2.plot(range(1, len(hist) + 1), hist, label=f'X={h}', linewidth=1.5)

    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax2.set_title('Test Accuracy Convergence by Architecture', fontsize=12)
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([80, 100])

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'mnist_architecture_search.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    run_search()
