"""
Architecture Search & Sample Complexity — Part (e)
===================================================
Finds the best architecture of a 64-X-3 neural network by trying different
values of X (hidden units). Also analyzes how sample complexity (number of
training samples) affects classification performance.

Generates supporting plots:
- Accuracy vs. number of hidden units
- Training loss vs. hidden units
- Sample complexity curves for different architectures
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from data_generation import generate_dataset
from neural_network import NeuralNetwork


def architecture_search(save_dir='results'):
    """
    Part (e) — Search for the best 64-X-3 architecture.
    
    Tests various hidden layer sizes and compares:
    - Final training accuracy
    - Convergence speed (loss curves)
    - Number of parameters
    
    Args:
        save_dir (str): Directory to save plots.
    
    Returns:
        dict: Results for each hidden size tested.
    """
    print("=" * 60)
    print("PART (e): Architecture Search — Finding Best 64-X-3")
    print("=" * 60)
    
    # Generate full dataset
    X, y, class_names = generate_dataset(n_samples_per_class=100, noise_range=5.0, seed=42)
    
    # Hidden sizes to test
    hidden_sizes = [1, 2, 3, 4, 5, 8, 10, 16, 32]
    
    results = {}
    epochs = 5000
    learning_rate = 0.5
    n_trials = 5  # Run multiple trials for statistical robustness
    
    for h in hidden_sizes:
        print(f"\n--- Testing hidden_size = {h} ---")
        trial_accuracies = []
        trial_losses = []
        best_acc = 0
        best_loss_history = None
        best_acc_history = None
        
        for trial in range(n_trials):
            nn = NeuralNetwork(input_size=64, hidden_size=h, output_size=3, seed=42 + trial)
            loss_history, acc_history = nn.train(
                X, y,
                epochs=epochs,
                learning_rate=learning_rate,
                verbose=False
            )
            
            final_acc = nn.get_accuracy(X, y)
            trial_accuracies.append(final_acc)
            trial_losses.append(loss_history[-1])
            
            if final_acc > best_acc:
                best_acc = final_acc
                best_loss_history = loss_history
                best_acc_history = acc_history
        
        mean_acc = np.mean(trial_accuracies)
        std_acc = np.std(trial_accuracies)
        n_params = (64 * h + h) + (h * 3 + 3)  # weights + biases
        
        results[h] = {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'best_accuracy': max(trial_accuracies),
            'mean_loss': np.mean(trial_losses),
            'n_params': n_params,
            'loss_history': best_loss_history,
            'acc_history': best_acc_history,
        }
        
        print(f"  Accuracy: {mean_acc:.1f}% ± {std_acc:.1f}% | Best: {max(trial_accuracies):.1f}% | Params: {n_params}")
    
    # ==========================================
    # Plot 1: Accuracy vs Hidden Units
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Architecture Search: 64-X-3 Network Performance', fontsize=14, fontweight='bold')
    
    h_list = list(results.keys())
    mean_accs = [results[h]['mean_accuracy'] for h in h_list]
    std_accs = [results[h]['std_accuracy'] for h in h_list]
    best_accs = [results[h]['best_accuracy'] for h in h_list]
    n_params_list = [results[h]['n_params'] for h in h_list]
    
    # Bar + error chart
    bars = ax1.bar(range(len(h_list)), mean_accs, yerr=std_accs,
                   color=plt.cm.viridis(np.linspace(0.2, 0.8, len(h_list))),
                   edgecolor='black', linewidth=0.5, capsize=3)
    ax1.scatter(range(len(h_list)), best_accs, color='red', zorder=5, s=40, label='Best trial')
    ax1.set_xticks(range(len(h_list)))
    ax1.set_xticklabels([str(h) for h in h_list])
    ax1.set_xlabel('Number of Hidden Units (X)', fontsize=11)
    ax1.set_ylabel('Training Accuracy (%)', fontsize=11)
    ax1.set_title('Accuracy vs. Architecture', fontsize=12)
    ax1.set_ylim([0, 105])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Secondary axis for parameter count
    ax1b = ax1.twinx()
    ax1b.plot(range(len(h_list)), n_params_list, 'o--', color='orange', label='# Parameters')
    ax1b.set_ylabel('Number of Parameters', fontsize=11, color='orange')
    ax1b.tick_params(axis='y', labelcolor='orange')
    ax1b.legend(loc='lower right', fontsize=9)
    
    # Loss curves for all architectures
    for h in h_list:
        ax2.plot(results[h]['loss_history'], label=f'X={h}', linewidth=1.2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('MSE Loss (log scale)', fontsize=11)
    ax2.set_title('Training Loss Convergence', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'architecture_search.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: architecture_search.png")
    
    # ==========================================
    # Plot 2: Accuracy curves over epochs for each architecture
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    for h in h_list:
        ax.plot(results[h]['acc_history'], label=f'X={h}', linewidth=1.2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Accuracy (%)', fontsize=11)
    ax.set_title('Accuracy Convergence for Different Architectures', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_convergence.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: accuracy_convergence.png")
    
    # Find the best architecture
    best_h = max(results, key=lambda h: results[h]['mean_accuracy'])
    print(f"\n{'='*60}")
    print(f"BEST ARCHITECTURE: 64-{best_h}-3")
    print(f"Mean Accuracy: {results[best_h]['mean_accuracy']:.1f}% ± {results[best_h]['std_accuracy']:.1f}%")
    print(f"Parameters: {results[best_h]['n_params']}")
    print(f"{'='*60}")
    
    return results, best_h


def sample_complexity_analysis(save_dir='results'):
    """
    Part (e) — Analyze how sample complexity affects performance.
    
    Varies the number of training samples per class and measures
    the resulting accuracy for different architectures.
    The sample complexity is the number of training examples needed
    to achieve a desired level of generalization.
    
    Args:
        save_dir (str): Directory to save plots.
    """
    print("\n" + "=" * 60)
    print("PART (e): Sample Complexity Analysis")
    print("=" * 60)
    
    # Sample sizes to test (per class)
    sample_sizes = [5, 10, 20, 30, 50, 75, 100, 150, 200]
    
    # Architectures to compare
    architectures = [2, 3, 5, 10, 16]
    
    epochs = 5000
    learning_rate = 0.5
    n_trials = 3
    
    results = {h: {'mean_accs': [], 'std_accs': []} for h in architectures}
    
    for n_samples in sample_sizes:
        print(f"\n--- Samples per class: {n_samples} (total: {n_samples*3}) ---")
        
        for h in architectures:
            trial_accs = []
            for trial in range(n_trials):
                # Generate training data
                X_train, y_train, _ = generate_dataset(
                    n_samples_per_class=n_samples, noise_range=5.0, seed=42 + trial
                )
                
                # Generate separate test data for measuring generalization
                X_test, y_test, _ = generate_dataset(
                    n_samples_per_class=100, noise_range=5.0, seed=1000 + trial
                )
                
                # Train
                nn = NeuralNetwork(input_size=64, hidden_size=h, output_size=3, seed=42 + trial)
                nn.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate, verbose=False)
                
                # Evaluate on test set (generalization)
                test_acc = nn.get_accuracy(X_test, y_test)
                trial_accs.append(test_acc)
            
            results[h]['mean_accs'].append(np.mean(trial_accs))
            results[h]['std_accs'].append(np.std(trial_accs))
            print(f"  X={h:2d}: Test Acc = {np.mean(trial_accs):.1f}% ± {np.std(trial_accs):.1f}%")
    
    # ==========================================
    # Plot: Sample Complexity Curves
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Sample Complexity Analysis', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(architectures)))
    
    for i, h in enumerate(architectures):
        mean_accs = results[h]['mean_accs']
        std_accs = results[h]['std_accs']
        
        ax1.plot(sample_sizes, mean_accs, 'o-', color=colors[i], label=f'X={h}', linewidth=1.5)
        ax1.fill_between(sample_sizes,
                         np.array(mean_accs) - np.array(std_accs),
                         np.array(mean_accs) + np.array(std_accs),
                         alpha=0.15, color=colors[i])
    
    ax1.set_xlabel('Training Samples per Class', fontsize=11)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax1.set_title('Generalization vs. Sample Size', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # Plot total samples on x-axis
    total_samples = [s * 3 for s in sample_sizes]
    for i, h in enumerate(architectures):
        n_params = (64 * h + h) + (h * 3 + 3)
        ax2.plot(total_samples, results[h]['mean_accs'], 'o-', color=colors[i],
                label=f'X={h} ({n_params} params)', linewidth=1.5)
    
    ax2.set_xlabel('Total Training Samples', fontsize=11)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax2.set_title('Accuracy vs. Total Samples\n(with parameter counts)', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_complexity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: sample_complexity.png")
    
    return results


def main():
    """Run the complete architecture search and sample complexity analysis."""
    os.makedirs('results', exist_ok=True)
    
    # Part 1: Architecture search
    arch_results, best_h = architecture_search()
    
    # Part 2: Sample complexity
    sample_results = sample_complexity_analysis()
    
    print("\n" + "=" * 60)
    print("PART (e) COMPLETED")
    print("All plots saved in results/")
    print("=" * 60)


if __name__ == '__main__':
    main()
