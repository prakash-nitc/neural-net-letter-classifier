"""
CIFAR-10 Training
==================
Trains a 3072-512-256-10 MLP on CIFAR-10 from scratch in NumPy.
Demonstrates that the same framework scales to a real colour-image benchmark
and reveals why plain MLPs hit a ceiling (~52%) vs CNNs (93%+).

Run:
    python train_cifar10.py   (~5-8 minutes on CPU)

Outputs in results/:
    cifar10_training_curves.png   — loss + test accuracy per epoch
    cifar10_confusion_matrix.png  — 10x10 confusion matrix
    cifar10_sample_predictions.png

Model saved to:
    model/cifar10_model.npz
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from deep_neural_network import DeepNeuralNetwork
from cifar10_loader import load_cifar10, CLASSES

# ── Config ────────────────────────────────────────────────────────────────────
LAYER_SIZES   = [3072, 512, 256, 10]
EPOCHS        = 40
LEARNING_RATE = 0.001
BATCH_SIZE    = 512
SAVE_DIR      = 'results'
MODEL_PATH    = 'model/cifar10_model.npz'


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_training_curves(loss_history, val_acc_history, save_dir):
    epochs = range(1, len(loss_history) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('CIFAR-10 Training — 3072-512-256-10 MLP (NumPy, from scratch)',
                 fontsize=12, fontweight='bold')

    ax1.plot(epochs, loss_history, color='#4facfe', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_acc_history, color='#f97316', linewidth=2)
    ax2.axhline(y=val_acc_history[-1], color='red', linestyle='--', alpha=0.5,
                label=f'Final: {val_acc_history[-1]:.2f}%')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Test Set Accuracy per Epoch')
    ax2.set_ylim([20, 70])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'cifar10_training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_confusion_matrix(y_true, y_pred, save_dir):
    n = 10
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    acc = cm.diagonal().sum() / cm.sum() * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Oranges')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLASSES, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix — CIFAR-10 Test Set\nOverall Accuracy: {acc:.2f}%',
                 fontsize=12, fontweight='bold')

    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7,
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    path = os.path.join(save_dir, 'cifar10_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

    return acc, cm.diagonal() / cm.sum(axis=1) * 100


def plot_sample_predictions(X_test, y_true, y_pred, save_dir, n=25):
    wrong = np.where(y_pred != y_true)[0]
    right = np.where(y_pred == y_true)[0]
    idx   = np.concatenate([wrong[:5], right[:20]])[:n]

    fig, axes = plt.subplots(5, 5, figsize=(11, 11))
    fig.suptitle('Sample CIFAR-10 Predictions (green = correct, red = wrong)',
                 fontsize=12, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i >= len(idx):
            ax.axis('off')
            continue
        j = idx[i]
        # Reverse normalisation for display (approximate)
        img = X_test[j].reshape(3, 1024).T.reshape(32, 32, 3)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(img)
        color = 'red' if y_pred[j] != y_true[j] else 'green'
        ax.set_title(f'T:{CLASSES[y_true[j]][:4]}\nP:{CLASSES[y_pred[j]][:4]}',
                     color=color, fontsize=7, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(save_dir, 'cifar10_sample_predictions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs('model', exist_ok=True)

    print('=' * 65)
    print('  CIFAR-10 - From-Scratch MLP (NumPy only)')
    print(f'  Architecture : {" -> ".join(map(str, LAYER_SIZES))}')
    print(f'  Optimizer    : Adam  lr={LEARNING_RATE}  batch={BATCH_SIZE}')
    print(f'  Epochs       : {EPOCHS}')
    print('=' * 65)

    X_train, X_test, y_train, y_test, y_train_labels, y_test_labels, _ = load_cifar10()

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
    test_acc  = nn.get_accuracy(X_test,  y_test)
    print(f'\nFinal Train Accuracy : {train_acc:.2f}%')
    print(f'Final Test  Accuracy : {test_acc:.2f}%')

    nn.save(MODEL_PATH)

    print('\nGenerating visualisations ...')
    plot_training_curves(nn.loss_history, nn.val_accuracy_history, SAVE_DIR)
    y_pred, _ = nn.predict(X_test)
    overall_acc, per_class = plot_confusion_matrix(y_test_labels, y_pred, SAVE_DIR)
    plot_sample_predictions(X_test, y_test_labels, y_pred, SAVE_DIR)

    print('\n' + '=' * 65)
    print('  CIFAR-10 RESULTS')
    print('-' * 65)
    print(f'  Test Accuracy  : {test_acc:.2f}%')
    print(f'  (Random chance : 10.00% | Typical CNN : ~93%)')
    print()
    print('  Per-class accuracy:')
    for i, (cls, acc) in enumerate(zip(CLASSES, per_class)):
        bar = '#' * int(acc / 5)
        print(f'    {cls:12s}: {acc:5.1f}%  {bar}')
    print()
    print('  Why does MLP plateau at ~50%?')
    print('  MLPs treat each pixel independently, ignoring spatial structure.')
    print('  CNNs use local filters (weight sharing) to detect edges and')
    print('  textures at any position, which is why they reach 93%+.')
    print('=' * 65)


if __name__ == '__main__':
    main()
