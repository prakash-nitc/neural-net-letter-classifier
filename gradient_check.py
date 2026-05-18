"""
Gradient Check
===============
Verifies that backpropagation is mathematically correct by comparing
analytical gradients against numerical gradients computed via centered
finite differences:

    dL/dw  ~=  [ L(w + eps) - L(w - eps) ] / (2 * eps)

A relative error < 1e-5 confirms the implementation is correct.
This is a standard sanity check when building neural networks from scratch.

Run:
    python gradient_check.py
"""

import numpy as np
from deep_neural_network import DeepNeuralNetwork


def check_gradients(layer_sizes=None, activation='relu', loss='cross_entropy',
                    n_samples=8, eps=1e-5, seed=0):
    """
    Run gradient check on a small randomly-initialised network.

    Args:
        layer_sizes : network architecture (small for speed)
        activation  : hidden activation to test
        loss        : 'cross_entropy' or 'mse'
        n_samples   : number of data points (keep small — O(params) evaluations)
        eps         : finite-difference step size
        seed        : random seed

    Returns:
        bool: True if all relative errors are below threshold
    """
    if layer_sizes is None:
        layer_sizes = [6, 5, 4, 3]   # small 3-layer network

    np.random.seed(seed)
    n_out = layer_sizes[-1]

    nn = DeepNeuralNetwork(
        layer_sizes, activation=activation, loss=loss,
        optimizer='sgd', learning_rate=0.0, seed=seed
    )

    X = np.random.randn(n_samples, layer_sizes[0]) * 0.5
    y = np.eye(n_out)[np.random.randint(0, n_out, n_samples)]

    print('=' * 62)
    print('  Gradient Check — Verifying Backpropagation')
    print(f'  Network   : {layer_sizes}')
    print(f'  Activation: {activation}   Loss: {loss}')
    print(f'  Samples   : {n_samples}     eps = {eps}')
    print('=' * 62)

    # ── Analytical gradients via backprop ────────────────────────────────────
    nn.forward(X)
    analytical_w, analytical_b = nn._compute_grads(X, y)

    # ── Numerical gradients via finite differences ───────────────────────────
    numerical_w = []
    numerical_b = []

    for l in range(nn.n_layers):
        grad_W = np.zeros_like(nn.weights[l])
        for i in range(nn.weights[l].shape[0]):
            for j in range(nn.weights[l].shape[1]):
                nn.weights[l][i, j] += eps
                loss_p = nn.compute_loss(y, nn.forward(X))
                nn.weights[l][i, j] -= 2 * eps
                loss_m = nn.compute_loss(y, nn.forward(X))
                nn.weights[l][i, j] += eps          # restore
                grad_W[i, j] = (loss_p - loss_m) / (2 * eps)
        numerical_w.append(grad_W)

        grad_b = np.zeros_like(nn.biases[l])
        for j in range(nn.biases[l].shape[1]):
            nn.biases[l][0, j] += eps
            loss_p = nn.compute_loss(y, nn.forward(X))
            nn.biases[l][0, j] -= 2 * eps
            loss_m = nn.compute_loss(y, nn.forward(X))
            nn.biases[l][0, j] += eps               # restore
            grad_b[0, j] = (loss_p - loss_m) / (2 * eps)
        numerical_b.append(grad_b)

    # ── Compare ──────────────────────────────────────────────────────────────
    THRESHOLD = 1e-4
    all_pass  = True

    print(f'\n{"Layer":<7}{"Param":<8}{"Rel Error":<18}{"Max |diff|":<18}Status')
    print('-' * 62)

    for l in range(nn.n_layers):
        for a_grad, n_grad, name in [
            (analytical_w[l], numerical_w[l], f'W{l+1}'),
            (analytical_b[l], numerical_b[l], f'b{l+1}'),
        ]:
            diff     = a_grad - n_grad
            rel_err  = (np.linalg.norm(diff)
                        / (np.linalg.norm(a_grad) + np.linalg.norm(n_grad) + 1e-15))
            max_diff = np.max(np.abs(diff))
            status   = 'PASS' if rel_err < THRESHOLD else 'FAIL'
            if status == 'FAIL':
                all_pass = False
            print(f'  {l+1}     {name:<8}{rel_err:<18.2e}{max_diff:<18.2e}{status}')

    print('-' * 62)
    if all_pass:
        print('All gradients verified. Backpropagation is mathematically correct.')
    else:
        print('WARNING: gradient mismatch detected. Check implementation.')
    print()

    return all_pass


if __name__ == '__main__':
    print('\n--- Test 1: ReLU + Cross-Entropy ---')
    check_gradients(layer_sizes=[6, 5, 4, 3], activation='relu',    loss='cross_entropy')

    print('--- Test 2: Sigmoid + MSE ----------')
    check_gradients(layer_sizes=[5, 4, 3],    activation='sigmoid', loss='mse')

    print('--- Test 3: Tanh + Cross-Entropy ---')
    check_gradients(layer_sizes=[8, 6, 4, 2], activation='tanh',    loss='cross_entropy')
