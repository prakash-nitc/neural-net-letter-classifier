"""
Training and Visualization Script — Parts (a), (b), (c)
========================================================
(a) Trains a 64-3-3 neural network with bias on dataset D.
(b) Displays input-to-hidden weights as 8×8 images and interprets them.
(c) Displays hidden-to-output weights and interprets them.

All plots are saved to the results/ directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from data_generation import generate_dataset, get_letter_templates
from neural_network import NeuralNetwork


def plot_templates(templates, save_dir='results'):
    """
    Plot the clean 8x8 letter templates for reference.
    
    Args:
        templates (dict): Dictionary of letter templates.
        save_dir (str): Directory to save the plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.suptitle('Clean 8×8 Letter Templates (Black = -1, White = +1)', fontsize=14, fontweight='bold')
    
    class_names = ['B', '0', 'E']
    for i, name in enumerate(class_names):
        ax = axes[i]
        # Use a grayscale colormap: -1 (black) maps to dark, +1 (white) maps to light
        im = ax.imshow(templates[name], cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
        ax.set_title(f"Letter '{name}'", fontsize=12, fontweight='bold')
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.grid(True, linewidth=0.5, color='gray', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'templates.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: templates.png")


def plot_noisy_samples(X, y, class_names, save_dir='results'):
    """
    Plot a grid of noisy sample images for each class.
    Shows 5 examples per class to visualize the noise effect.
    
    Args:
        X (np.ndarray): Input data (n_samples, 64).
        y (np.ndarray): One-hot labels (n_samples, 3).
        class_names (list): List of class names.
        save_dir (str): Directory to save the plot.
    """
    fig, axes = plt.subplots(3, 5, figsize=(12, 8))
    fig.suptitle('Noisy Training Samples (5 per class)', fontsize=14, fontweight='bold')
    
    for i, name in enumerate(class_names):
        # Get indices for this class
        class_mask = y[:, i] == 1
        class_samples = X[class_mask]
        
        for j in range(5):
            ax = axes[i, j]
            img = class_samples[j].reshape(8, 8)
            ax.imshow(img, cmap='RdBu', interpolation='nearest')
            if j == 0:
                ax.set_ylabel(f"'{name}'", fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'noisy_samples.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: noisy_samples.png")


def plot_training_curves(loss_history, accuracy_history, save_dir='results'):
    """
    Plot the training loss and accuracy curves over epochs.
    
    Args:
        loss_history (list): Loss at each epoch.
        accuracy_history (list): Accuracy at each epoch.
        save_dir (str): Directory to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('64-3-3 Network Training Progress', fontsize=14, fontweight='bold')
    
    # Loss curve
    ax1.plot(loss_history, color='#e74c3c', linewidth=1.5)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('MSE Loss', fontsize=11)
    ax1.set_title('Training Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Accuracy curve
    ax2.plot(accuracy_history, color='#2ecc71', linewidth=1.5)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Training Accuracy', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: training_curves.png")


def plot_input_hidden_weights(nn, save_dir='results'):
    """
    Part (b): Display input-to-hidden weights as 8×8 images.
    
    Each hidden unit has 64 weights connecting it to the input layer.
    These weights can be reshaped to 8×8 to visualize what pattern
    each hidden neuron has learned to detect.
    
    Args:
        nn (NeuralNetwork): Trained neural network.
        save_dir (str): Directory to save the plot.
    """
    fig, axes = plt.subplots(1, nn.hidden_size, figsize=(4 * nn.hidden_size, 4))
    fig.suptitle('Input → Hidden Weights (reshaped to 8×8)\nPart (b): What each hidden neuron detects',
                 fontsize=14, fontweight='bold')
    
    if nn.hidden_size == 1:
        axes = [axes]
    
    for i in range(nn.hidden_size):
        ax = axes[i]
        # W1[:, i] contains the 64 weights from all inputs to hidden unit i
        weight_image = nn.W1[:, i].reshape(8, 8)
        
        # Use diverging colormap: blue = negative weights, red = positive weights
        vmax = max(abs(weight_image.min()), abs(weight_image.max()))
        im = ax.imshow(weight_image, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
        ax.set_title(f'Hidden Unit {i+1}\n(bias: {nn.b1[0, i]:.3f})', fontsize=11)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.grid(True, linewidth=0.5, color='gray', alpha=0.3)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'input_hidden_weights.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: input_hidden_weights.png")


def plot_hidden_output_weights(nn, class_names, save_dir='results'):
    """
    Part (c): Display hidden-to-output weights in a meaningful way.
    
    Visualizes how each output neuron (representing a class) combines the
    hidden unit activations. Shows both a heatmap and a grouped bar chart.
    
    Args:
        nn (NeuralNetwork): Trained neural network.
        class_names (list): Names of the output classes.
        save_dir (str): Directory to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Hidden → Output Weights\nPart (c): How hidden features combine to form class decisions',
                 fontsize=14, fontweight='bold')
    
    # --- Heatmap ---
    # W2 shape: (hidden_size, output_size)
    vmax = max(abs(nn.W2.min()), abs(nn.W2.max()))
    im = ax1.imshow(nn.W2, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax1.set_xlabel('Output Neuron (Class)', fontsize=11)
    ax1.set_ylabel('Hidden Unit', fontsize=11)
    ax1.set_xticks(range(nn.output_size))
    ax1.set_xticklabels([f"'{name}'" for name in class_names])
    ax1.set_yticks(range(nn.hidden_size))
    ax1.set_yticklabels([f'HU {i+1}' for i in range(nn.hidden_size)])
    ax1.set_title('Weight Heatmap', fontsize=12)
    
    # Annotate with actual values
    for i in range(nn.hidden_size):
        for j in range(nn.output_size):
            ax1.text(j, i, f'{nn.W2[i, j]:.2f}', ha='center', va='center',
                    color='white' if abs(nn.W2[i, j]) > vmax * 0.5 else 'black', fontsize=10)
    
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    
    # --- Grouped Bar Chart ---
    x = np.arange(nn.output_size)
    width = 0.8 / nn.hidden_size
    colors = plt.cm.Set2(np.linspace(0, 1, nn.hidden_size))
    
    for i in range(nn.hidden_size):
        offset = (i - nn.hidden_size / 2 + 0.5) * width
        bars = ax2.bar(x + offset, nn.W2[i, :], width, label=f'Hidden Unit {i+1}',
                      color=colors[i], edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Output Class', fontsize=11)
    ax2.set_ylabel('Weight Value', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"'{name}'" for name in class_names])
    ax2.set_title('Weight Bar Chart', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add bias info
    bias_text = ', '.join([f"'{class_names[i]}': {nn.b2[0, i]:.3f}" for i in range(nn.output_size)])
    ax2.text(0.5, -0.15, f'Output biases: {bias_text}',
             transform=ax2.transAxes, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hidden_output_weights.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: hidden_output_weights.png")


def plot_confusion_matrix(nn, X, y, class_names, save_dir='results'):
    """
    Plot a confusion matrix for the trained model's predictions.
    
    Args:
        nn (NeuralNetwork): Trained neural network.
        X (np.ndarray): Input data.
        y (np.ndarray): True labels (one-hot).
        class_names (list): Class names.
        save_dir (str): Directory to save the plot.
    """
    predictions, _ = nn.predict(X)
    true_labels = np.argmax(y, axis=1)
    
    # Build confusion matrix
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_labels, predictions):
        cm[t, p] += 1
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title('Confusion Matrix (64-3-3 Network)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels([f"'{n}'" for n in class_names])
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels([f"'{n}'" for n in class_names])
    
    # Annotate cells
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                   color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: confusion_matrix.png")


def main():
    """
    Main function to execute parts (a), (b), and (c) of the assignment.
    """
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    
    # ==========================================
    # Step 1: Generate Dataset D
    # ==========================================
    print("=" * 60)
    print("STEP 1: Generating Dataset D (300 samples)")
    print("=" * 60)
    X, y, class_names = generate_dataset(n_samples_per_class=100, noise_range=5.0, seed=42)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Classes: {class_names}")
    
    # Plot templates and noisy samples
    templates = get_letter_templates()
    plot_templates(templates)
    plot_noisy_samples(X, y, class_names)
    
    # ==========================================
    # Step 2: Part (a) — Train 64-3-3 Network
    # ==========================================
    print("\n" + "=" * 60)
    print("STEP 2: Part (a) — Training 64-3-3 Network")
    print("=" * 60)
    nn = NeuralNetwork(input_size=64, hidden_size=3, output_size=3, seed=42)
    
    # Train with a reasonable learning rate and sufficient epochs
    loss_history, accuracy_history = nn.train(
        X, y,
        epochs=5000,
        learning_rate=0.5,
        verbose=True,
        print_every=500
    )
    
    # Plot training curves
    plot_training_curves(loss_history, accuracy_history)
    
    # Final accuracy
    final_accuracy = nn.get_accuracy(X, y)
    print(f"\nFinal Training Accuracy: {final_accuracy:.1f}%")
    
    # Confusion matrix
    plot_confusion_matrix(nn, X, y, class_names)
    
    # Save model
    nn.save('model/model_64_3_3.npz')
    
    # ==========================================
    # Step 3: Part (b) — Visualize Input-Hidden Weights
    # ==========================================
    print("\n" + "=" * 60)
    print("STEP 3: Part (b) — Input → Hidden Weight Visualization")
    print("=" * 60)
    plot_input_hidden_weights(nn)
    
    print("\nInterpretation of Input-Hidden Weights:")
    print("-" * 40)
    print("Each hidden unit's weights, reshaped as an 8×8 image, reveal the")
    print("spatial pattern that the neuron has learned to detect. Since the")
    print("network must distinguish B, 0, and E using only 3 hidden neurons,")
    print("each neuron learns to act as a feature detector for distinguishing")
    print("features of the letters — such as vertical strokes, horizontal bars,")
    print("or the curved/closed regions that differentiate B from E and 0.")
    
    # ==========================================
    # Step 4: Part (c) — Visualize Hidden-Output Weights
    # ==========================================
    print("\n" + "=" * 60)
    print("STEP 4: Part (c) — Hidden → Output Weight Visualization")
    print("=" * 60)
    plot_hidden_output_weights(nn, class_names)
    
    print("\nInterpretation of Hidden-Output Weights:")
    print("-" * 40)
    print("The hidden-to-output weights show how each output neuron (class)")
    print("combines the features detected by the hidden units. A large positive")
    print("weight means the hidden unit's activation strongly supports that class,")
    print("while a large negative weight means it opposes that class. This creates")
    print("a decision rule: each class is recognized by a unique combination of")
    print("the three learned features.")
    
    print("\n" + "=" * 60)
    print("ALL PARTS (a), (b), (c) COMPLETED SUCCESSFULLY")
    print(f"Results saved in: results/")
    print(f"Model saved in: model/")
    print("=" * 60)


if __name__ == '__main__':
    main()
