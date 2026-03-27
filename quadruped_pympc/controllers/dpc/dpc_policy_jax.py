import jax
import jax.numpy as jnp
import flax.linen as nn


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
    max_fx: float = 150.0
    max_fy: float = 150.0
    nominal_fz: float = 80.0

    def pack_inputs(self, state, reference, current_contact):
        """Concatenate policy inputs into a single feature vector."""
        return jnp.concatenate((state, reference, current_contact.astype(jnp.float32)), axis=0)

    def _activation(self, x):
        if self.activation == "relu":
            return jax.nn.relu(x)
        if self.activation == "gelu":
            return jax.nn.gelu(x)
        return jnp.tanh(x)

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
        Vertical force is parameterized as positive via softplus.
        Swing-leg forces are masked out by the current contact vector.
        """
        raw_output = raw_output.reshape(4, 3)

        fx = self.max_fx * jnp.tanh(raw_output[:, 0])
        fy = self.max_fy * jnp.tanh(raw_output[:, 1])
        fz = jax.nn.softplus(raw_output[:, 2]) + self.nominal_fz

        grfs = jnp.stack((fx, fy, fz), axis=1)
        grfs = grfs * current_contact[:, None]
        return grfs.reshape(12,)


def project_grfs_with_friction(grfs, current_contact, mu, fz_min, fz_max):
    """Project predicted GRFs into a friction-cone-feasible set.

    This mirrors the same physical constraints used by the sampling MPC:
    - unilateral contact in z
    - vertical force bounds
    - friction cone bounds for x/y
    """
    grfs = grfs.reshape(4, 3)

    fx = grfs[:, 0]
    fy = grfs[:, 1]
    fz = grfs[:, 2]

    fz = jnp.clip(fz, fz_min, fz_max)
    fx = jnp.clip(fx, -mu * fz, mu * fz)
    fy = jnp.clip(fy, -mu * fz, mu * fz)

    projected = jnp.stack((fx, fy, fz), axis=1)
    projected = projected * current_contact[:, None]
    return projected.reshape(12,)
