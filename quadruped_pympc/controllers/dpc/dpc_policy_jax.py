import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct


@struct.dataclass
class PolicyBounds:
    """Optional bounds for 2D controls [delta_v, a]."""

    delta_v_min: float
    delta_v_max: float
    a_min: float
    a_max: float


class NeuralGRFPolicy(nn.Module):
    """Simple MLP policy for DPC-style first-step GRF prediction.

    Input:
    - current centroidal state x          : 24
    - reference centroidal state x_ref    : 24
    - current contact flags               : 4

    Output:
    - first-step GRFs for FL/FR/RL/RR     : 12
      ordered as [fx_FL, fy_FL, fz_FL, fx_FR, fy_FR, fz_FR, ...]
    """
    num_layers: int = 2
    hidden_dim: int = 256
    activation: str = "gelu"
    # hard code those things for now
    max_fx: float = 30.0 
    max_fy: float = 30.0
    max_fz: float = 241.68897

    def pack_inputs(self, state, reference, current_contact):
        """Concatenate policy inputs into a single feature vector."""
        return jnp.concatenate((state, reference, current_contact.astype(jnp.float32)), axis=0)

    def _activation(self, x):
        if self.activation == "relu":
            return jax.nn.relu(x)
        if self.activation == "gelu":
            return jax.nn.gelu(x)
        if self.activation == "tanh":
            return jnp.tanh(x)
        raise ValueError(f"Unsupported activation='{self.activation}'")

    @nn.compact
    def __call__(self, state, reference, current_contact):
        """Predict first-step GRFs from state, reference, and contact flags."""
        x = self.pack_inputs(state, reference, current_contact)
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = self._activation(x)
        raw_output = nn.Dense(12)(x)
        return self.decode_action(raw_output, current_contact)

    def decode_action(self, raw_output, current_contact):
        """Map unconstrained network output to physically meaningful GRFs.
        Horizontal forces are bounded by tanh scaling.
        Vertical force is bounded directly to [0, max_fz].
        Swing-leg forces are masked out by the current contact vector.
        """
        raw_output = raw_output.reshape(4, 3)

        fx = self.max_fx * jnp.tanh(raw_output[:, 0])
        fy = self.max_fy * jnp.tanh(raw_output[:, 1])
        fz = self.max_fz * jax.nn.sigmoid(raw_output[:, 2])

        grfs = jnp.stack((fx, fy, fz), axis=1)
        grfs = grfs * current_contact[:, None]
        return grfs.reshape(12,)


class NeuralGRFDistributionPolicy(nn.Module):
    """Distributional version of NeuralGRFPolicy.

    It uses the same inputs and GRF bounds as the deterministic policy, but
    returns a bounded mean force together with a diagonal standard deviation
    for each GRF component.
    """

    num_layers: int = 2
    hidden_dim: int = 256
    activation: str = "gelu"
    min_std: float = 1e-3
    max_std: float = 1e2
    max_fx: float = 30.0
    max_fy: float = 30.0
    max_fz: float = 241.68897

    def pack_inputs(self, state, reference, current_contact):
        """Concatenate policy inputs into a single feature vector."""
        return jnp.concatenate((state, reference, current_contact.astype(jnp.float32)), axis=0)

    def _activation(self, x):
        if self.activation == "relu":
            return jax.nn.relu(x)
        if self.activation == "gelu":
            return jax.nn.gelu(x)
        if self.activation == "tanh":
            return jnp.tanh(x)
        raise ValueError(f"Unsupported activation='{self.activation}'")

    def decode_mean(self, raw_output, current_contact):
        """Bound the force mean exactly like NeuralGRFPolicy."""
        raw_output = raw_output.reshape(4, 3)

        fx = self.max_fx * jnp.tanh(raw_output[:, 0])
        fy = self.max_fy * jnp.tanh(raw_output[:, 1])
        fz = self.max_fz * jax.nn.sigmoid(raw_output[:, 2])

        grfs = jnp.stack((fx, fy, fz), axis=1)
        grfs = grfs * current_contact[:, None]
        return grfs.reshape(12,)

    @nn.compact
    def __call__(self, state, reference, current_contact):
        """Return bounded GRF mean and diagonal standard deviation.

        The backbone and mean-head Dense layer names intentionally mirror
        NeuralGRFPolicy so the mean branch can be warm-started from a
        deterministic policy checkpoint:
        - hidden layers: Dense_0 .. Dense_{num_layers-1}
        - mean head: Dense_{num_layers}
        """
        x = self.pack_inputs(state, reference, current_contact)
        for layer_idx in range(self.num_layers):
            x = nn.Dense(
                self.hidden_dim,
                name=f"Dense_{layer_idx}",
            )(x)
            x = self._activation(x)

        mean_raw = nn.Dense(
            12,
            name=f"Dense_{self.num_layers}",
        )(x)
        mean = self.decode_mean(mean_raw, current_contact)

        std = jax.nn.softplus(
            nn.Dense(
                12,
                name="StdDense",
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )(x)
        ) + self.min_std
        std = jnp.clip(std, a_min=self.min_std, a_max=self.max_std)
        std = std * jnp.repeat(current_contact, 3) + self.min_std * (1.0 - jnp.repeat(current_contact, 3))
        return mean, std
