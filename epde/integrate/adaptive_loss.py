import deepxde as dde
import numpy as np


class AdaptiveLoss(dde.callbacks.Callback):
    def __init__(self, model, optimizer, lr,
                 update_every=1000, window_size=20,
                 alpha=0.9, max_delta=0.5, epsilon=1e-8):
        super().__init__()
        self.model = model
        self.optimizer = optimizer   # строка, например 'adam'
        self.lr = lr
        self.update_every = update_every
        self.window_size = window_size
        self.alpha = alpha
        self.max_delta = max_delta
        self.epsilon = epsilon

        self.weights = None
        self._loss_history = []

    def on_train_begin(self):
        # Достаём текущие веса, если DeepXDE их хранит
        try:
            lw = self.model.loss_weights
            self.weights = np.array(lw, dtype=float) if lw is not None else None
        except AttributeError:
            self.weights = None
        self._loss_history = []

    def on_epoch_end(self):
        try:
            epoch = self.model.train_state.epoch
        except AttributeError:
            return
        if epoch % self.update_every != 0:
            return

        try:
            current_losses = np.array(self.model.losshistory.loss_train[-1], dtype=float)
        except (IndexError, AttributeError):
            return

        if self._loss_history and np.allclose(self._loss_history[-1], current_losses):
            return

        if current_losses.ndim == 0 or current_losses.size < 2:
            print("[AdaptiveLoss] WARNING: only 1 loss component visible, "
                  "cannot balance. Skipping update.")
            return

        if self.weights is None or len(self.weights) != current_losses.size:
            self.weights = np.ones_like(current_losses)

        self._loss_history.append(current_losses)
        if len(self._loss_history) < self.window_size:
            return

        recent = np.array(self._loss_history[-self.window_size:])
        variances = np.var(recent, axis=0) + self.epsilon

        target = 1.0 / variances
        target = target / np.sum(target)

        smoothed = self.alpha * self.weights + (1.0 - self.alpha) * target
        delta = np.clip(smoothed - self.weights, -self.max_delta, self.max_delta)
        new_weights = self.weights + delta
        new_weights = new_weights / np.sum(new_weights)

        self.weights = new_weights

        try:
            self.model.compile(
                self.optimizer,
                lr=self.lr,
                loss_weights=self.weights.tolist(),
                verbose=False,
            )
            #print(f"[AdaptiveLoss] Updated weights: "
            #      f"{np.round(self.weights, 4).tolist()}")
        except Exception as e:
            print(f"[AdaptiveLoss] Recompile failed: {e}")