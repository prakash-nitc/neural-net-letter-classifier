"""
Data Generation Module for TFML Assignment 1
=============================================
Generates 8x8 pixel representations of letters B, 0 (zero), and E.
Each letter template uses -1.0 for black (ink) pixels and +1.0 for white (background) pixels.
100 noisy versions of each letter are created by adding uniform random noise in [-5.0, +5.0].
Total dataset D: 300 samples (100 per category).
"""

import numpy as np


def get_letter_templates():
    """
    Define the 8x8 pixel templates for letters B, 0 (zero), and E.
    Convention: -1.0 = black (letter stroke), +1.0 = white (background).
    
    Returns:
        dict: Dictionary mapping letter names to their 8x8 numpy arrays.
    """
    # Template for letter 'B'
    # Visual representation (X = black, . = white):
    # X X X X X X . .
    # X . . . . . X .
    # X . . . . . X .
    # X X X X X X . .
    # X . . . . . X .
    # X . . . . . X .
    # X X X X X X . .
    # . . . . . . . .
    B = np.array([
        [-1, -1, -1, -1, -1, -1,  1,  1],
        [-1,  1,  1,  1,  1,  1, -1,  1],
        [-1,  1,  1,  1,  1,  1, -1,  1],
        [-1, -1, -1, -1, -1, -1,  1,  1],
        [-1,  1,  1,  1,  1,  1, -1,  1],
        [-1,  1,  1,  1,  1,  1, -1,  1],
        [-1, -1, -1, -1, -1, -1,  1,  1],
        [ 1,  1,  1,  1,  1,  1,  1,  1],
    ], dtype=np.float64)

    # Template for digit '0' (zero)
    # Visual representation:
    # . X X X X X . .
    # X . . . . . X .
    # X . . . . . X .
    # X . . . . . X .
    # X . . . . . X .
    # X . . . . . X .
    # . X X X X X . .
    # . . . . . . . .
    zero = np.array([
        [ 1, -1, -1, -1, -1, -1,  1,  1],
        [-1,  1,  1,  1,  1,  1, -1,  1],
        [-1,  1,  1,  1,  1,  1, -1,  1],
        [-1,  1,  1,  1,  1,  1, -1,  1],
        [-1,  1,  1,  1,  1,  1, -1,  1],
        [-1,  1,  1,  1,  1,  1, -1,  1],
        [ 1, -1, -1, -1, -1, -1,  1,  1],
        [ 1,  1,  1,  1,  1,  1,  1,  1],
    ], dtype=np.float64)

    # Template for letter 'E'
    # Visual representation:
    # X X X X X X . .
    # X . . . . . . .
    # X . . . . . . .
    # X X X X X . . .
    # X . . . . . . .
    # X . . . . . . .
    # X X X X X X . .
    # . . . . . . . .
    E = np.array([
        [-1, -1, -1, -1, -1, -1,  1,  1],
        [-1,  1,  1,  1,  1,  1,  1,  1],
        [-1,  1,  1,  1,  1,  1,  1,  1],
        [-1, -1, -1, -1, -1,  1,  1,  1],
        [-1,  1,  1,  1,  1,  1,  1,  1],
        [-1,  1,  1,  1,  1,  1,  1,  1],
        [-1, -1, -1, -1, -1, -1,  1,  1],
        [ 1,  1,  1,  1,  1,  1,  1,  1],
    ], dtype=np.float64)

    return {'B': B, '0': zero, 'E': E}


def generate_noisy_samples(template, n_samples=100, noise_range=5.0, seed=None):
    """
    Generate noisy versions of a letter template.
    
    Each pixel gets independent uniform random noise added from [-noise_range, +noise_range].
    
    Args:
        template (np.ndarray): The 8x8 letter template.
        n_samples (int): Number of noisy samples to generate (default: 100).
        noise_range (float): Range of uniform noise, noise ~ U(-noise_range, +noise_range).
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        np.ndarray: Array of shape (n_samples, 64) containing flattened noisy samples.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Flatten the 8x8 template to a 64-dimensional vector
    flat_template = template.flatten()
    
    # Generate noise: shape (n_samples, 64), uniform in [-noise_range, +noise_range]
    noise = np.random.uniform(-noise_range, noise_range, size=(n_samples, 64))
    
    # Add noise to create noisy samples
    noisy_samples = flat_template + noise
    
    return noisy_samples


def generate_dataset(n_samples_per_class=100, noise_range=5.0, seed=42):
    """
    Generate the complete dataset D with 300 training samples.
    
    Creates n_samples_per_class noisy versions for each of B, 0, E.
    Labels are one-hot encoded: B=[1,0,0], 0=[0,1,0], E=[0,0,1].
    
    Args:
        n_samples_per_class (int): Number of samples per letter category.
        noise_range (float): Noise range for uniform distribution.
        seed (int): Random seed for reproducibility.
    
    Returns:
        tuple: (X, y, labels) where:
            X: np.ndarray of shape (total_samples, 64) — input features
            y: np.ndarray of shape (total_samples, 3) — one-hot encoded labels
            labels: list of str — class names ['B', '0', 'E']
    """
    templates = get_letter_templates()
    class_names = ['B', '0', 'E']
    
    X_list = []
    y_list = []
    
    for i, name in enumerate(class_names):
        # Generate noisy samples for this class
        # Use different seed offsets for each class to ensure different noise patterns
        samples = generate_noisy_samples(
            templates[name], 
            n_samples=n_samples_per_class, 
            noise_range=noise_range,
            seed=seed + i
        )
        X_list.append(samples)
        
        # Create one-hot encoded labels for this class
        one_hot = np.zeros((n_samples_per_class, 3))
        one_hot[:, i] = 1.0
        y_list.append(one_hot)
    
    # Concatenate all classes
    X = np.vstack(X_list)
    y = np.vstack(y_list)
    
    # Shuffle the dataset while keeping X and y aligned
    np.random.seed(seed)
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y, class_names


if __name__ == '__main__':
    # Quick verification: generate dataset and print statistics
    X, y, class_names = generate_dataset()
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Classes: {class_names}")
    print(f"Samples per class: {[np.sum(y[:, i]) for i in range(3)]}")
    print(f"X value range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"X mean: {X.mean():.4f}")
    
    # Display the clean templates
    templates = get_letter_templates()
    for name, template in templates.items():
        print(f"\nTemplate '{name}':")
        for row in template:
            print(' '.join(['█' if p == -1 else '·' for p in row]))
