import jax
import jax.numpy as jnp

from quadruped_pympc import config as cfg
from quadruped_pympc.controllers.dpc.dpc_policy_jax import NeuralGRFPolicy, project_grfs_with_friction
from quadruped_pympc.controllers.sampling.centroidal_model_jax import Centroidal_Model_JAX


class DPC:
    """Differentiable predictive control for GRF policies.

    This class is intended to train and evaluate a neural policy that maps the
    current centroidal state and reference to first-step ground reaction forces.
    The predictive model is the same centroidal dynamics.
    """

    def __init__(
        self,
        policy: NeuralGRFPolicy | None = None,
        device: str | None = None,
        horizon: int | None = None,
        dt: float | None = None,
    ):
        # ---- Global controller configuration -------------------------------------------------
        # solver uses the same prediction horizon, time step, and execution device.
        self.device_name = cfg.mpc_params["device"] if device is None else device
        self.dt = cfg.mpc_params["dt"] if dt is None else dt
        self.horizon = cfg.mpc_params["horizon"] if horizon is None else horizon

        self.state_dim = 24
        self.control_dim = 24
        self.reference_dim = self.state_dim

        self.control_parametrization = cfg.mpc_params["control_parametrization"]

        # These are not optimizer variables in DPC, but they are still useful
        # as output scales for the neural policy.
        self.max_sampling_forces_x = 10
        self.max_sampling_forces_y = 10
        self.max_sampling_forces_z = 30

        if self.control_parametrization in ("linear_spline", "cubic_spline"):
            self.num_spline = cfg.mpc_params["num_splines"]

        # ---- Select the JAX execution device ------------------------------------------------
        if self.device_name == "gpu":
            try:
                self.device = jax.devices("gpu")[0]
            except Exception:
                self.device = jax.devices("cpu")[0]
                print("GPU not available, using CPU")
        else:
            self.device = jax.devices("cpu")[0]

        # Neural policy            
        if policy is None:
            self.policy = NeuralGRFPolicy(
                max_fx=self.max_sampling_forces_x,
                max_fy=self.max_sampling_forces_y,
                nominal_fz=self.max_sampling_forces_z,
            )
        else:
            self.policy = policy

        # Prediction model
        # The same centroidal model used by the sampling MPC rollout.
        self.model = Centroidal_Model_JAX(self.dt, self.device)

        # Cost weights
        self.mu = cfg.mpc_params["mu"]
        self.f_z_min = cfg.mpc_params["grf_min"]
        self.f_z_max = cfg.mpc_params["grf_max"]

        self.Q = self._default_state_cost_weight()
        self.R = self._default_control_cost_weight()

    def _default_state_cost_weight(self):
        """State cost aligned with the sampling MPC default weighting."""
        Q = jnp.identity(self.state_dim, dtype=jnp.float32) * 0.0
        Q = Q.at[2, 2].set(1500.0)
        Q = Q.at[3, 3].set(200.0)
        Q = Q.at[4, 4].set(200.0)
        Q = Q.at[5, 5].set(200.0)
        Q = Q.at[6, 6].set(500.0)
        Q = Q.at[7, 7].set(500.0)
        Q = Q.at[9, 9].set(20.0)
        Q = Q.at[10, 10].set(20.0)
        Q = Q.at[11, 11].set(50.0)
        return Q

    def _default_control_cost_weight(self):
        """Control cost aligned with the sampling MPC default weighting."""
        R = jnp.identity(self.control_dim, dtype=jnp.float32)
        R = R.at[0, 0].set(0.0)
        R = R.at[1, 1].set(0.0)
        R = R.at[2, 2].set(0.0)
        R = R.at[3, 3].set(0.0)
        R = R.at[4, 4].set(0.0)
        R = R.at[5, 5].set(0.0)
        R = R.at[6, 6].set(0.0)
        R = R.at[7, 7].set(0.0)
        R = R.at[8, 8].set(0.0)
        R = R.at[9, 9].set(0.0)
        R = R.at[10, 10].set(0.0)
        R = R.at[11, 11].set(0.0)

        R = R.at[12, 12].set(0.1)
        R = R.at[13, 13].set(0.1)
        R = R.at[14, 14].set(0.001)
        R = R.at[15, 15].set(0.1)
        R = R.at[16, 16].set(0.1)
        R = R.at[17, 17].set(0.001)
        R = R.at[18, 18].set(0.1)
        R = R.at[19, 19].set(0.1)
        R = R.at[20, 20].set(0.001)
        R = R.at[21, 21].set(0.1)
        R = R.at[22, 22].set(0.1)
        R = R.at[23, 23].set(0.001)
        return R

    def compute_linear_spline(self, parameters, step, horizon_leg):
        """Linear-spline copied from Sampling_MPC."""
        chunk_boundaries = jnp.linspace(0, self.horizon, self.num_spline + 1)
        index = jnp.max(jnp.where(step >= chunk_boundaries, jnp.arange(self.num_spline + 1), 0))

        tau = step / (horizon_leg / self.num_spline)
        tau = tau - 1 * index
        q = (tau - 0.0) / (1.0 - 0.0)

        shift = self.num_spline + 1
        f_x = (1 - q) * parameters[index + 0] + q * parameters[index + 1]
        f_y = (1 - q) * parameters[index + shift] + q * parameters[index + shift + 1]
        f_z = (1 - q) * parameters[index + shift * 2] + q * parameters[index + shift * 2 + 1]
        return f_x, f_y, f_z

    def compute_cubic_spline(self, parameters, step, horizon_leg):
        """Cubic-spline copied from Sampling_MPC."""
        chunk_boundaries = jnp.linspace(0, self.horizon, self.num_spline + 1)
        index = jnp.max(jnp.where(step >= chunk_boundaries, jnp.arange(self.num_spline + 1), 0))

        tau = step / (horizon_leg / self.num_spline)
        tau = tau - 1 * index
        q = (tau - 0.0) / (1.0 - 0.0)

        start_index = 10 * index
        a = 2 * q * q * q - 3 * q * q + 1
        b = (q * q * q - 2 * q * q + q) * 1.0
        c = -2 * q * q * q + 3 * q * q
        d = (q * q * q - q * q) * 1.0

        phi = 0.5 * (
            ((parameters[start_index + 2] - parameters[start_index + 1]) / 1.0)
            + ((parameters[start_index + 1] - parameters[start_index + 0]) / 1.0)
        )
        phi_next = 0.5 * (
            ((parameters[start_index + 3] - parameters[start_index + 2]) / 1.0)
            + ((parameters[start_index + 2] - parameters[start_index + 1]) / 1.0)
        )
        f_x = a * parameters[start_index + 1] + b * phi + c * parameters[start_index + 2] + d * phi_next

        phi = 0.5 * (
            ((parameters[start_index + 6] - parameters[start_index + 5]) / 1.0)
            + ((parameters[start_index + 5] - parameters[start_index + 4]) / 1.0)
        )
        phi_next = 0.5 * (
            ((parameters[start_index + 7] - parameters[start_index + 6]) / 1.0)
            + ((parameters[start_index + 6] - parameters[start_index + 5]) / 1.0)
        )
        f_y = a * parameters[start_index + 5] + b * phi + c * parameters[start_index + 6] + d * phi_next

        phi = 0.5 * (
            ((parameters[start_index + 10] - parameters[start_index + 9]) / 1.0)
            + ((parameters[start_index + 9] - parameters[start_index + 8]) / 1.0)
        )
        phi_next = 0.5 * (
            ((parameters[start_index + 11] - parameters[start_index + 10]) / 1.0)
            + ((parameters[start_index + 10] - parameters[start_index + 9]) / 1.0)
        )
        f_z = a * parameters[start_index + 9] + b * phi + c * parameters[start_index + 10] + d * phi_next
        return f_x, f_y, f_z

    def compute_zero_order_spline(self, parameters, step, horizon_leg):
        """Zero-order-hold copied from Sampling_MPC."""
        del horizon_leg
        index = jnp.int16(step)
        f_x = parameters[index]
        f_y = parameters[index + self.horizon]
        f_z = parameters[index + self.horizon * 2]
        return f_x, f_y, f_z

    def enforce_force_constraints(
        self, f_x_FL, f_y_FL, f_z_FL, f_x_FR, f_y_FR, f_z_FR, f_x_RL, f_y_RL, f_z_RL, f_x_RR, f_y_RR, f_z_RR
    ):
        """Sampling_MPC contact-force projection utility."""
        f_z_FL = jnp.where(f_z_FL > self.f_z_min, f_z_FL, self.f_z_min)
        f_z_FR = jnp.where(f_z_FR > self.f_z_min, f_z_FR, self.f_z_min)
        f_z_RL = jnp.where(f_z_RL > self.f_z_min, f_z_RL, self.f_z_min)
        f_z_RR = jnp.where(f_z_RR > self.f_z_min, f_z_RR, self.f_z_min)

        f_z_FL = jnp.where(f_z_FL < self.f_z_max, f_z_FL, self.f_z_max)
        f_z_FR = jnp.where(f_z_FR < self.f_z_max, f_z_FR, self.f_z_max)
        f_z_RL = jnp.where(f_z_RL < self.f_z_max, f_z_RL, self.f_z_max)
        f_z_RR = jnp.where(f_z_RR < self.f_z_max, f_z_RR, self.f_z_max)

        f_x_FL = jnp.where(f_x_FL > -self.mu * f_z_FL, f_x_FL, -self.mu * f_z_FL)
        f_x_FL = jnp.where(f_x_FL < self.mu * f_z_FL, f_x_FL, self.mu * f_z_FL)
        f_y_FL = jnp.where(f_y_FL > -self.mu * f_z_FL, f_y_FL, -self.mu * f_z_FL)
        f_y_FL = jnp.where(f_y_FL < self.mu * f_z_FL, f_y_FL, self.mu * f_z_FL)

        f_x_FR = jnp.where(f_x_FR > -self.mu * f_z_FR, f_x_FR, -self.mu * f_z_FR)
        f_x_FR = jnp.where(f_x_FR < self.mu * f_z_FR, f_x_FR, self.mu * f_z_FR)
        f_y_FR = jnp.where(f_y_FR > -self.mu * f_z_FR, f_y_FR, -self.mu * f_z_FR)
        f_y_FR = jnp.where(f_y_FR < self.mu * f_z_FR, f_y_FR, self.mu * f_z_FR)

        f_x_RL = jnp.where(f_x_RL > -self.mu * f_z_RL, f_x_RL, -self.mu * f_z_RL)
        f_x_RL = jnp.where(f_x_RL < self.mu * f_z_RL, f_x_RL, self.mu * f_z_RL)
        f_y_RL = jnp.where(f_y_RL > -self.mu * f_z_RL, f_y_RL, -self.mu * f_z_RL)
        f_y_RL = jnp.where(f_y_RL < self.mu * f_z_RL, f_y_RL, self.mu * f_z_RL)

        f_x_RR = jnp.where(f_x_RR > -self.mu * f_z_RR, f_x_RR, -self.mu * f_z_RR)
        f_x_RR = jnp.where(f_x_RR < self.mu * f_z_RR, f_x_RR, self.mu * f_z_RR)
        f_y_RR = jnp.where(f_y_RR > -self.mu * f_z_RR, f_y_RR, -self.mu * f_z_RR)
        f_y_RR = jnp.where(f_y_RR < self.mu * f_z_RR, f_y_RR, self.mu * f_z_RR)

        return f_x_FL, f_y_FL, f_z_FL, f_x_FR, f_y_FR, f_z_FR, f_x_RL, f_y_RL, f_z_RL, f_x_RR, f_y_RR, f_z_RR

    def init_policy_params(self, key):
        """Initialize neural policy parameters."""
        return self.policy.init(key)

    def project_grfs(self, grfs, current_contact):
        """Project policy outputs into friction-constrained contact forces."""
        return project_grfs_with_friction(
            grfs=grfs,
            current_contact=current_contact,
            mu=self.mu,
            fz_min=self.f_z_min,
            fz_max=self.f_z_max,
        )

    def predict_first_step_grfs(self, params, state, reference, current_contact):
        """Run the policy and return feasible first-step GRFs."""
        grfs = self.policy.apply(params, state, reference, current_contact)
        return self.project_grfs(grfs, current_contact)

    def prepare_state_and_reference(self, state_current, reference_state, current_contact, previous_contact=None):
        """
        ``previous_contact`` is accepted for interface compatibility with
        ``SRBDControllerInterface`` but is not used by the current DPC solver.
        """
        del previous_contact
        state_current_jax = jnp.concatenate(
            (
                jnp.asarray(state_current["position"]),
                jnp.asarray(state_current["linear_velocity"]),
                jnp.asarray(state_current["orientation"]),
                jnp.asarray(state_current["angular_velocity"]),
                jnp.asarray(state_current["foot_FL"]).reshape((3,)),
                jnp.asarray(state_current["foot_FR"]).reshape((3,)),
                jnp.asarray(state_current["foot_RL"]).reshape((3,)),
                jnp.asarray(state_current["foot_RR"]).reshape((3,)),
            )
        ).reshape((24,))

        if current_contact[0] == 0.0:
            state_current_jax = state_current_jax.at[12:15].set(jnp.asarray(reference_state["ref_foot_FL"]).reshape((3,)))
        if current_contact[1] == 0.0:
            state_current_jax = state_current_jax.at[15:18].set(jnp.asarray(reference_state["ref_foot_FR"]).reshape((3,)))
        if current_contact[2] == 0.0:
            state_current_jax = state_current_jax.at[18:21].set(jnp.asarray(reference_state["ref_foot_RL"]).reshape((3,)))
        if current_contact[3] == 0.0:
            state_current_jax = state_current_jax.at[21:24].set(jnp.asarray(reference_state["ref_foot_RR"]).reshape((3,)))

        reference_state_jax = jnp.concatenate(
            (
                jnp.asarray(reference_state["ref_position"]),
                jnp.asarray(reference_state["ref_linear_velocity"]),
                jnp.asarray(reference_state["ref_orientation"]),
                jnp.asarray(reference_state["ref_angular_velocity"]),
                jnp.asarray(reference_state["ref_foot_FL"]).reshape((3,)),
                jnp.asarray(reference_state["ref_foot_FR"]).reshape((3,)),
                jnp.asarray(reference_state["ref_foot_RL"]).reshape((3,)),
                jnp.asarray(reference_state["ref_foot_RR"]).reshape((3,)),
            )
        ).reshape((24,))

        return state_current_jax, reference_state_jax

    def rollout_cost(self, params, initial_state, reference, contact_sequence):
        """Compute differentiable horizon rollout cost under the neural GRF policy.

        Args:
            params: Neural policy parameters.
            initial_state: Current centroidal state, shape (24,).
            reference: Reference centroidal state, shape (24,).
            contact_sequence: Binary stance/swing schedule, shape (4, horizon).

        Returns:
            Scalar rollout cost accumulated over the full horizon.
        """
        state = initial_state
        cost = jnp.float32(0.0)

        def iterate_fun(n, carry):
            running_cost, state = carry

            current_contact = jnp.array(
                [
                    contact_sequence[0][n],
                    contact_sequence[1][n],
                    contact_sequence[2][n],
                    contact_sequence[3][n],
                ],
                dtype=jnp.float32,
            )

            grfs = self.predict_first_step_grfs(params, state, reference, current_contact)
            grfs_matrix = grfs.reshape(4, 3)

            number_of_legs_in_stance = jnp.sum(current_contact)
            safe_num_stance = jnp.maximum(number_of_legs_in_stance, 1.0)
            reference_force_stance_legs = (self.model.mass * 9.81) / safe_num_stance

            input_vec = jnp.concatenate(
                (
                    jnp.zeros((12,), dtype=jnp.float32),
                    grfs,
                ),
                axis=0,
            )

            state_next = self.model.integrate_jax(state, input_vec, current_contact, n)

            state_error = state_next - reference[0:self.state_dim]
            state_cost = state_error.T @ self.Q @ state_error

            input_for_cost = jnp.concatenate(
                (
                    jnp.zeros((12,), dtype=jnp.float32),
                    jnp.stack(
                        (
                            grfs_matrix[0, 0],
                            grfs_matrix[0, 1],
                            grfs_matrix[0, 2] - reference_force_stance_legs,
                            grfs_matrix[1, 0],
                            grfs_matrix[1, 1],
                            grfs_matrix[1, 2] - reference_force_stance_legs,
                            grfs_matrix[2, 0],
                            grfs_matrix[2, 1],
                            grfs_matrix[2, 2] - reference_force_stance_legs,
                            grfs_matrix[3, 0],
                            grfs_matrix[3, 1],
                            grfs_matrix[3, 2] - reference_force_stance_legs,
                        ),
                        axis=0,
                    ),
                ),
                axis=0,
            )
            control_cost = input_for_cost.T @ self.R @ input_for_cost

            return running_cost + state_cost + control_cost, state_next

        cost, _ = jax.lax.fori_loop(0, self.horizon, iterate_fun, (cost, state))
        return cost

    def loss(self, params, batch):
        """Compute mean rollout cost over a batch.

        Supported batch formats:

        1. Packed tensors:
           {
               "initial_state": (B, 24),
               "reference": (B, 24),
               "contact_sequence": (B, 4, H),
           }

        2. SRBD-style dictionaries:
           {
               "state_current": sequence[dict],
               "reference_state": sequence[dict],
               "current_contact": (B, 4),
               "previous_contact": (B, 4),  # optional
               "contact_sequence": (B, 4, H),
           }
        """
        if {"initial_state", "reference", "contact_sequence"}.issubset(batch.keys()):
            initial_state = jnp.asarray(batch["initial_state"])
            reference = jnp.asarray(batch["reference"])
            contact_sequence = jnp.asarray(batch["contact_sequence"])
        elif {"state_current", "reference_state", "current_contact", "contact_sequence"}.issubset(batch.keys()):
            state_current_batch = batch["state_current"]
            reference_state_batch = batch["reference_state"]
            current_contact_batch = jnp.asarray(batch["current_contact"])
            previous_contact_batch = jnp.asarray(batch.get("previous_contact", current_contact_batch))
            contact_sequence = jnp.asarray(batch["contact_sequence"])

            packed_states = []
            packed_references = []
            for idx in range(len(state_current_batch)):
                packed_state, packed_reference = self.prepare_state_and_reference(
                    state_current_batch[idx],
                    reference_state_batch[idx],
                    current_contact_batch[idx],
                    previous_contact_batch[idx],
                )
                packed_states.append(packed_state)
                packed_references.append(packed_reference)

            initial_state = jnp.stack(packed_states, axis=0)
            reference = jnp.stack(packed_references, axis=0)
        else:
            raise ValueError(
                "Unsupported batch format. Expected either packed tensors "
                "('initial_state', 'reference', 'contact_sequence') or SRBD-style "
                "inputs ('state_current', 'reference_state', 'current_contact', 'contact_sequence')."
            )

        batched_rollout_cost = jax.vmap(
            lambda x0, x_ref, contacts: self.rollout_cost(params, x0, x_ref, contacts),
            in_axes=(0, 0, 0),
            out_axes=0,
        )
        rollout_costs = batched_rollout_cost(initial_state, reference, contact_sequence)
        return jnp.mean(rollout_costs)
