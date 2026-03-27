import jax
import jax.numpy as jnp

from quadruped_pympc.controllers.dpc.dpc_solver import DPC


class DPC_Trainer:
    """Training scaffold for differentiable predictive control policies."""

    def __init__(self, dpc: DPC):
        self.dpc = dpc

    def init_params(self, key):
        """Initialize policy parameters through the associated DPC solver."""
        return self.dpc.init_policy_params(key)

    def loss(self, params, batch):
        """Delegate batch loss computation to the DPC solver."""
        return self.dpc.loss(params, batch)

    def train_step(self, params, batch, optimizer_state=None):
        """Placeholder for one optimization step."""
        raise NotImplementedError("train_step is not implemented yet.")

    def fit(self, params, dataset, optimizer_state=None, num_epochs=1):
        """Placeholder for the outer training loop."""
        raise NotImplementedError("fit is not implemented yet.")

    def evaluate(self, params, batch):
        """Placeholder for validation/evaluation logic."""
        raise NotImplementedError("evaluate is not implemented yet.")
