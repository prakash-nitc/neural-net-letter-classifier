"""
Deep Neural Network — From Scratch in NumPy
=============================================
N-layer feedforward neural network supporting:
  - Arbitrary depth via layer_sizes list, e.g. [784, 256, 128, 10]
  - Activations : ReLU, Sigmoid, Tanh
  - Loss        : Cross-Entropy (+ Softmax output), MSE (+ Sigmoid output)
  - Optimizer   : Adam, SGD
  - Regularization : L2
  - Initialization : He (ReLU), Xavier (Sigmoid / Tanh)
  - Mini-batch training with optional validation tracking per epoch
"""

import numpy as np
import os


class DeepNeuralNetwork:
    """
    N-layer feedforward neural network built entirely from scratch in NumPy.

    Example usage:
        nn = DeepNeuralNetwork([784, 256, 128, 10])          # 3-layer MLP
        nn.train(X_train, y_train, epochs=30, X_val=X_test, y_val=y_test)
        acc = nn.get_accuracy(X_test, y_test)
        nn.save('model/my_model.npz')
    """

    def __init__(self, layer_sizes, activation='relu', loss='cross_entropy',
                 optimizer='adam', learning_rate=0.001, l2_lambda=0.0, seed=42):
        """
        Args:
            layer_sizes   : list of ints — [input, hidden..., output], e.g. [784, 128, 10]
            activation    : hidden-layer activation — 'relu', 'sigmoid', 'tanh'
            loss          : 'cross_entropy' (softmax output) or 'mse' (sigmoid output)
            optimizer     : 'adam' or 'sgd'
            learning_rate : step size for weight updates
            l2_lambda     : L2 regularisation strength (0 = off)
            seed          : random seed for reproducibility
        """
        assert len(layer_sizes) >= 2, "Need at least input + output layers"

        self.layer_sizes    = layer_sizes
        self.activation     = activation
        self.loss           = loss
        self.optimizer_type = optimizer
        self.lr             = learning_rate
        self.l2             = l2_lambda
        self.n_layers       = len(layer_sizes) - 1   # number of weight matrices

        # Output activation is determined by loss choice
        self.output_activation = 'softmax' if loss == 'cross_entropy' else 'sigmoid'

        np.random.seed(seed)
        self._init_weights()
        self._init_adam()

        self.loss_history         = []
        self.accuracy_history     = []
        self.val_accuracy_history = []

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        self.weights, self.biases = [], []
        for i in range(self.n_layers):
            fan_in  = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            # He initialisation for ReLU hidden layers; Xavier for Sigmoid/Tanh
            is_hidden = (i < self.n_layers - 1)
            scale = (np.sqrt(2.0 / fan_in)
                     if (self.activation == 'relu' and is_hidden)
                     else np.sqrt(2.0 / (fan_in + fan_out)))
            self.weights.append(np.random.randn(fan_in, fan_out) * scale)
            self.biases.append(np.zeros((1, fan_out)))

    # ── Adam state ────────────────────────────────────────────────────────────

    def _init_adam(self):
        self.adam_t   = 0
        self.adam_b1  = 0.9
        self.adam_b2  = 0.999
        self.adam_eps = 1e-8
        self.m_w = [np.zeros_like(W) for W in self.weights]
        self.v_w = [np.zeros_like(W) for W in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]

    # ── Activation functions ──────────────────────────────────────────────────

    @staticmethod
    def _act(z, fn):
        if fn == 'relu':
            return np.maximum(0.0, z)
        if fn == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        if fn == 'tanh':
            return np.tanh(z)
        if fn == 'softmax':
            z_s = z - np.max(z, axis=1, keepdims=True)   # numerical stability
            e   = np.exp(z_s)
            return e / np.sum(e, axis=1, keepdims=True)
        raise ValueError(f"Unknown activation: {fn}")

    @staticmethod
    def _act_deriv(a, fn):
        """Derivative of activation given post-activation value a = f(z)."""
        if fn == 'relu':
            return (a > 0).astype(np.float64)
        if fn == 'sigmoid':
            return a * (1.0 - a)
        if fn == 'tanh':
            return 1.0 - a ** 2
        # softmax derivative handled implicitly via CE+softmax combined gradient
        raise ValueError(f"Derivative not defined for: {fn}")

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, X):
        """
        Forward pass through all layers. Caches every activation for backprop.

        Returns output activations (shape: n_samples x output_size).
        """
        self._cache = [X]   # _cache[l] = output of layer l (0 = raw input)
        a = X
        for i in range(self.n_layers):
            z  = a @ self.weights[i] + self.biases[i]
            fn = self.output_activation if i == self.n_layers - 1 else self.activation
            a  = self._act(z, fn)
            self._cache.append(a)
        return a

    # ── Backward pass ────────────────────────────────────────────────────────

    def _compute_grads(self, X, y):
        """
        Compute weight gradients via backpropagation (without updating weights).
        Used by both backward() and gradient_check().
        """
        n = X.shape[0]
        L = self.n_layers

        a_out = self._cache[L]

        # Output layer delta (averaged over batch)
        # CE + softmax: combined gradient simplifies neatly to (a - y)/n
        if self.loss == 'cross_entropy':
            delta = (a_out - y) / n
        else:
            delta = 2.0 * (a_out - y) * self._act_deriv(a_out, self.output_activation) / n

        grads_w = [None] * L
        grads_b = [None] * L

        for i in range(L - 1, -1, -1):
            a_prev      = self._cache[i]
            grads_w[i]  = a_prev.T @ delta + (self.l2 / n) * self.weights[i]
            grads_b[i]  = np.sum(delta, axis=0, keepdims=True)

            if i > 0:
                # Chain rule: propagate error signal to previous layer
                delta = (delta @ self.weights[i].T) * self._act_deriv(
                    self._cache[i], self.activation
                )

        return grads_w, grads_b

    def backward(self, X, y):
        """Backpropagate and update weights (calls _compute_grads then applies update)."""
        grads_w, grads_b = self._compute_grads(X, y)

        if self.optimizer_type == 'adam':
            self._adam_step(grads_w, grads_b)
        else:
            for i in range(self.n_layers):
                self.weights[i] -= self.lr * grads_w[i]
                self.biases[i]  -= self.lr * grads_b[i]

    def _adam_step(self, grads_w, grads_b):
        self.adam_t += 1
        t, b1, b2, eps = self.adam_t, self.adam_b1, self.adam_b2, self.adam_eps

        for i in range(self.n_layers):
            for g, m, v, p in [
                (grads_w[i], self.m_w, self.v_w, self.weights),
                (grads_b[i], self.m_b, self.v_b, self.biases),
            ]:
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * g ** 2
                m_hat = m[i] / (1 - b1 ** t)
                v_hat = v[i] / (1 - b2 ** t)
                p[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    # ── Loss ─────────────────────────────────────────────────────────────────

    def compute_loss(self, y_true, y_pred):
        if self.loss == 'cross_entropy':
            base = -np.mean(np.sum(y_true * np.log(np.clip(y_pred, 1e-10, 1.0)), axis=1))
        else:
            # Mean over samples, sum over outputs — gradient is 2*(pred-true)/n
            base = np.mean(np.sum((y_true - y_pred) ** 2, axis=1))

        if self.l2 > 0:
            base += (self.l2 / 2) * sum(np.sum(W ** 2) for W in self.weights)
        return base

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, X, y, epochs=30, learning_rate=None, batch_size=256,
              X_val=None, y_val=None, verbose=True, print_every=1):
        """
        Train with mini-batch gradient descent.

        Args:
            X, y          : training inputs and one-hot labels
            epochs        : number of full passes over training data
            learning_rate : overrides self.lr if given
            batch_size    : mini-batch size (None = full-batch GD)
            X_val, y_val  : optional validation set — val accuracy tracked each epoch
            verbose       : print progress
            print_every   : print frequency (every N epochs)

        Returns:
            (loss_history, accuracy_history)
        """
        if learning_rate is not None:
            self.lr = learning_rate

        self.loss_history         = []
        self.accuracy_history     = []
        self.val_accuracy_history = []

        n      = X.shape[0]
        eff_bs = n if (batch_size is None or batch_size >= n) else batch_size
        mini   = eff_bs < n

        for epoch in range(epochs):
            if mini:
                idx     = np.random.permutation(n)
                Xs, ys  = X[idx], y[idx]
            else:
                Xs, ys  = X, y

            epoch_loss, n_batches = 0.0, 0
            for start in range(0, n, eff_bs):
                Xb  = Xs[start:start + eff_bs]
                yb  = ys[start:start + eff_bs]
                out = self.forward(Xb)
                epoch_loss += self.compute_loss(yb, out)
                n_batches  += 1
                self.backward(Xb, yb)

            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)

            # Fast training-accuracy estimate on first 5K samples
            sample_n  = min(5000, n)
            out_s     = self.forward(X[:sample_n])
            train_acc = np.mean(np.argmax(out_s, 1) == np.argmax(y[:sample_n], 1)) * 100
            self.accuracy_history.append(train_acc)

            val_acc = None
            if X_val is not None and y_val is not None:
                val_acc = self.get_accuracy(X_val, y_val)
                self.val_accuracy_history.append(val_acc)

            if verbose and (epoch % print_every == 0 or epoch == epochs - 1):
                vs = f" | Val: {val_acc:.2f}%" if val_acc is not None else ""
                print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.4f} "
                      f"| Train: {train_acc:.1f}%{vs}")

        return self.loss_history, self.accuracy_history

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, X):
        """Returns (predicted_class_indices, output_probabilities)."""
        probs = self.forward(X)
        return np.argmax(probs, axis=1), probs

    def get_accuracy(self, X, y):
        """Classification accuracy (%) on dataset (X, y) with one-hot labels."""
        preds, _ = self.predict(X)
        return np.mean(preds == np.argmax(y, axis=1)) * 100

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self):
        total = sum(W.size + b.size for W, b in zip(self.weights, self.biases))
        arch  = ' -> '.join(map(str, self.layer_sizes))
        print(f"\nArchitecture  : {arch}")
        print(f"  Hidden act  : {self.activation}")
        print(f"  Output act  : {self.output_activation}")
        print(f"  Loss        : {self.loss}")
        print(f"  Optimizer   : {self.optimizer_type}  (lr={self.lr})")
        print(f"  L2 lambda   : {self.l2}")
        print(f"  Parameters  : {total:,}")
        for i in range(self.n_layers):
            print(f"  Layer {i+1}     : W{self.weights[i].shape}  b{self.biases[i].shape}")

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, filepath):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        d = {
            'layer_sizes': np.array(self.layer_sizes),
            'n_layers':    np.array(self.n_layers),
            'activation':  np.array(self.activation),
            'loss':        np.array(self.loss),
        }
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            d[f'W{i}'] = W
            d[f'b{i}'] = b
        np.savez(filepath, **d)
        total = sum(W.size + b.size for W, b in zip(self.weights, self.biases))
        print(f"Saved to {filepath}  (arch: {self.layer_sizes}, params: {total:,})")

    @classmethod
    def load(cls, filepath):
        """Load a saved model. Returns a DeepNeuralNetwork instance."""
        data   = np.load(filepath, allow_pickle=True)
        n      = int(data['n_layers'])
        obj    = cls.__new__(cls)
        obj.layer_sizes       = list(data['layer_sizes'].astype(int))
        obj.n_layers          = n
        obj.activation        = str(data['activation'])
        obj.loss              = str(data['loss'])
        obj.output_activation = 'softmax' if obj.loss == 'cross_entropy' else 'sigmoid'
        obj.optimizer_type    = 'adam'
        obj.lr                = 0.001
        obj.l2                = 0.0
        obj.weights           = [data[f'W{i}'] for i in range(n)]
        obj.biases            = [data[f'b{i}'] for i in range(n)]
        obj._init_adam()
        obj.loss_history         = []
        obj.accuracy_history     = []
        obj.val_accuracy_history = []
        print(f"Loaded from {filepath}  (arch: {obj.layer_sizes})")
        return obj
