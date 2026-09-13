import deepxde as dde
import numpy as np


class AdaptiveLoss(dde.callbacks.Callback):
    def __init__(self, model, optimizer, lr,
                 update_every=1000, window_size=20,
                 alpha=0.9, max_delta=0.5, epsilon=1e-8,
                 priority=None,
                 weight_min=None,
                 weight_max=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.update_every = update_every
        self.window_size = window_size
        self.alpha = alpha
        self.max_delta = max_delta
        self.epsilon = epsilon
        self.priority = None if priority is None else np.atleast_1d(
            np.asarray(priority, dtype=float))
        self.weight_min = weight_min
        self.weight_max = weight_max

        self.weights = None
        self._loss_history = []

    def on_train_begin(self):
        try:
            lw = self.model.loss_weights
            self.weights = None if lw is None else np.atleast_1d(
                np.array(lw, dtype=float))
        except AttributeError:
            self.weights = None
        self._loss_history = []
        print(f"[AdaptiveLossBalancer] Initialized. "
              f"priority={self.priority}, "
              f"bounds=({self.weight_min}, {self.weight_max})")

    def on_epoch_end(self):
        try:
            epoch = self.model.train_state.epoch
        except AttributeError:
            return
        if epoch % self.update_every != 0:
            return

        try:
            current_losses = np.atleast_1d(
                np.array(self.model.losshistory.loss_train[-1], dtype=float))
        except (IndexError, AttributeError):
            return

        if self.weights is None or self.weights.size != current_losses.size:
            self.weights = np.ones_like(current_losses)

        self._loss_history.append(current_losses)
        if len(self._loss_history) < self.window_size:
            return

        recent = np.array(self._loss_history[-self.window_size:])
        # recent: (window_size, n_losses) или (window_size,) если n_losses==1
        variances = np.atleast_1d(np.var(recent, axis=0) + self.epsilon)

        print("[DEBUG balancer] epoch:", getattr(self.model.train_state, 'epoch', None))
        print("[DEBUG balancer] model.loss_weights:", self.model.loss_weights)
        try:
            print("[DEBUG balancer] losshistory.loss_train[-1]:",
                  self.model.losshistory.loss_train[-1])
            print("[DEBUG balancer] len(losshistory.loss_train[-1]):",
                  len(self.model.losshistory.loss_train[-1]))
        except Exception as e:
            print("[DEBUG balancer] losshistory err:", e)
        try:
            print("[DEBUG balancer] train_state.loss_train[-1]:",
                  self.model.train_state.loss_train[-1])
        except Exception as e:
            print("[DEBUG balancer] train_state err:", e)

        if self.priority is not None and self.priority.size == variances.size:
            prio = self.priority
        else:
            if self.priority is not None:
                print(f"[AdaptiveLossBalancer] priority size "
                      f"{self.priority.size} != n_losses {variances.size}, "
                      f"ignoring priority.")
            prio = np.ones_like(variances)

        target = prio / variances
        target = target / np.sum(target)

        smoothed = self.alpha * self.weights + (1.0 - self.alpha) * target
        delta = np.clip(smoothed - self.weights,
                        -self.max_delta, self.max_delta)
        new_weights = self.weights + delta

        if self.weight_min is not None:
            new_weights = np.maximum(new_weights, self.weight_min)
        if self.weight_max is not None:
            new_weights = np.minimum(new_weights, self.weight_max)

        new_weights = new_weights / np.sum(new_weights)
        self.weights = new_weights

        try:
            self.model.compile(
                self.optimizer,
                lr=self.lr,
                loss_weights=self.weights.tolist(),
            )
            print(f"[AdaptiveLossBalancer] Updated weights: "
                  f"{np.round(self.weights, 4).tolist()}")
        except Exception as e:
            print(f"[AdaptiveLossBalancer] Recompile failed: {e}")