import jax
import jax.numpy as jnp
import optax

from quadruped_pympc.controllers.dpc.dpc_trainer import DPC_Trainer
from quadruped_pympc.controllers.dpc.dmppi_solver import DMPPI


class DMPPI_Trainer(DPC_Trainer):
    """Training for DMPPI policies built on top of DPC_Trainer."""

    def __init__(self, dmppi_solver: DMPPI, alpha: float = 1e-3):
        super().__init__(dmppi_solver)
        self.dmppi_solver = dmppi_solver
        self.alpha = float(alpha)
        self._eval_step = jax.jit(
            lambda params, batch, rollout_key: self._loss_and_metrics(params, batch, rollout_key)
        )

    def init_params(self, key):
        """Initialize policy parameters through the associated DMPPI solver."""
        return self.dmppi_solver.init_policy_params(key)

    def expl_loss(self, std_trajectory):
        """Entropy of the diagonal neural proposal distribution."""
        nu = std_trajectory.shape[-1]
        const = 0.5 * nu * (1.0 + jnp.log(2.0 * jnp.pi))
        safe_std = jnp.clip(std_trajectory, a_min=1e-8)
        h = jnp.sum(jnp.log(safe_std), axis=-1) + const
        return jnp.mean(h, axis=-1)

    def rollout_single(self, x0_single, R_single, C_single, params, rollout_key):
        """Run a single-sample rollout using the DMPPI."""

        def step_fn(state, inputs):
            reference, current_contact, step_idx = inputs

            nominal_grfs, nominal_std = self.dmppi_solver.policy.apply(
                {"params": params},
                state,
                reference,
                current_contact,
            )
            nominal_grfs = self.dmppi_solver.project_grfs(nominal_grfs, current_contact)
            step_key = jax.random.fold_in(rollout_key, step_idx)
            grfs = self.dmppi_solver.single_step_dmppi_correction(
                state,
                reference,
                current_contact,
                nominal_grfs,
                nominal_std,
                step_key,
            )
            grfs_matrix = grfs.reshape(4, 3)

            number_of_legs_in_stance = jnp.sum(current_contact)
            safe_num_stance = jnp.maximum(number_of_legs_in_stance, 1.0)
            reference_force_stance_legs = (self.dmppi_solver.model.mass * 9.81) / safe_num_stance

            input_vec = jnp.concatenate(
                (jnp.zeros((12,), dtype=jnp.float32), grfs),
                axis=0,
            )
            next_state = self.dmppi_solver.model.integrate_jax(state, input_vec, current_contact, step_idx)

            state_error = next_state - reference[: self.dmppi_solver.state_dim]
            track = state_error.T @ self.dmppi_solver.Q @ state_error

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
            pen = input_for_cost.T @ self.dmppi_solver.R @ input_for_cost
            loss = track + pen

            outputs = {
                "state": next_state,
                "control": grfs,
                "track": track,
                "pen": pen,
                "proposal_std": nominal_std,
                "loss": loss,
            }
            return next_state, outputs

        horizon = R_single.shape[0]
        step_indices = jnp.arange(horizon, dtype=jnp.int32)
        _, outputs = jax.lax.scan(step_fn, x0_single, (R_single, C_single, step_indices))

        X = jnp.concatenate((x0_single[None, :], outputs["state"]), axis=0)
        U = outputs["control"]
        track_mean = jnp.mean(outputs["track"])
        pen_mean = jnp.mean(outputs["pen"])
        expl_mean = self.expl_loss(outputs["proposal_std"][None, ...])[0]
        loss = track_mean + pen_mean - self.alpha * expl_mean
        return loss, track_mean, pen_mean, expl_mean, X, U

    def forward(self, params, x0, R, C, rollout_key):
        """Run one DMPPI rollout batch and include an entropy exploration bonus."""
        x0 = jnp.asarray(x0, dtype=jnp.float32)
        R = jnp.asarray(R, dtype=jnp.float32)
        C = jnp.asarray(C, dtype=jnp.float32)
        if C.ndim == 3 and C.shape[1] == 4:
            C = jnp.swapaxes(C, 1, 2)

        batch_size = x0.shape[0]
        rollout_keys = jax.random.split(rollout_key, batch_size)
        batched_forward = jax.vmap(
            lambda x0_single, R_single, C_single, sample_rollout_key: self.rollout_single(
                x0_single,
                R_single,
                C_single,
                params,
                sample_rollout_key,
            ),
            in_axes=(0, 0, 0, 0),
            out_axes=(0, 0, 0, 0, 0, 0),
        )
        loss, track, pen, expl, X, U = batched_forward(x0, R, C, rollout_keys)

        return {
            "loss": jnp.mean(loss),
            "track_mean": jnp.mean(track),
            "pen_mean": jnp.mean(pen),
            "expl_mean": jnp.mean(expl),
            "X": X,
            "U": U,
        }

    def _create_loss_batch(self, x0, R, C):
        """Keep the full reference horizon for DMPPI training."""
        return {
            "initial_state": jnp.asarray(x0, dtype=jnp.float32),
            "reference_state_horizon": jnp.asarray(R, dtype=jnp.float32),
            "contact_sequence": jnp.asarray(C, dtype=jnp.float32),
        }

    def _loss_and_metrics(self, params, batch, rollout_key):
        """Scalar batch loss and logging metrics for one stochastic rollout."""
        output = self.forward(
            params,
            batch["initial_state"],
            batch["reference_state_horizon"],
            batch["contact_sequence"],
            rollout_key,
        )
        metrics = {
            "track_mean": output["track_mean"],
            "pen_mean": output["pen_mean"],
            "expl_mean": output["expl_mean"],
        }
        return output["loss"], metrics

    def _build_train_step(self, optimizer):
        """Jitted train step that optimizes the entropy-augmented DMPPI loss."""
        def train_step(params, optimizer_state, batch, rollout_key):
            (loss_value, metrics), grads = jax.value_and_grad(
                lambda current_params: self._loss_and_metrics(current_params, batch, rollout_key),
                has_aux=True,
            )(params)
            updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
            params = optax.apply_updates(params, updates)
            return params, optimizer_state, loss_value, metrics

        return jax.jit(train_step)

    def train(self, training_params, params=None, optimizer_state=None, start_epoch: int = 0):
        """Train the DMPPI policy with one stochastic rollout per batch."""
        num_epochs = training_params.get("num_epochs", 50)
        steps_per_epoch = training_params.get("steps_per_epoch", 200)
        batch_size = training_params.get("batch_size", 64)
        lr = training_params.get("lr", 1e-3)
        weight_decay = training_params.get("weight_decay", 0.0)
        grad_clip = training_params.get("grad_clip", 1.0)
        log_every = training_params.get("log_every", 20)
        save_path = training_params.get("save_path", None)
        save_every = training_params.get("save_every", 10)

        plateau_factor = training_params.get("plateau_factor", 0.5)
        plateau_patience = training_params.get("plateau_patience", 5)
        plateau_min_lr = training_params.get("plateau_min_lr", 1e-6)
        plateau_threshold = training_params.get("plateau_threshold", 1e-4)
        plateau_cooldown = training_params.get("plateau_cooldown", 0)
        eval_batch_size = training_params.get("eval_batch_size", 256)
        early_stop_patience = training_params.get("early_stop_patience", 10)
        early_stop_min_delta = training_params.get("early_stop_min_delta", 1e-4)
        seed = training_params.get("seed", 0)
        eval_seed = training_params.get("eval_seed", 123)

        current_lr = lr
        optimizer = self._make_optimizer(current_lr, weight_decay, grad_clip)
        self._optimizer = optimizer
        self._train_step = self._build_train_step(optimizer)

        if params is None:
            params = self.init_params(jax.random.PRNGKey(seed))
        if optimizer_state is None:
            optimizer_state = optimizer.init(params)

        eval_batch_key = jax.random.PRNGKey(eval_seed)
        eval_x0, eval_R, eval_C = self.create_batch(eval_batch_size, key=eval_batch_key)
        eval_batch = self._create_loss_batch(eval_x0, eval_R, eval_C)

        best_eval_loss = float("inf")
        bad_epochs = 0
        bad_plateau_epochs = 0
        cooldown_counter = 0
        history = []
        train_rng = jax.random.PRNGKey(seed)
        eval_rng = jax.random.PRNGKey(eval_seed + 1)

        for epoch in range(start_epoch + 1, num_epochs + 1):
            loss_sum = 0.0
            expl_sum = 0.0

            for iteration in range(1, steps_per_epoch + 1):
                train_rng, batch_key, rollout_key = jax.random.split(train_rng, 3)
                x0_b, R_b, C_b = self.create_batch(batch_size, key=batch_key)
                batch = self._create_loss_batch(x0_b, R_b, C_b)

                params, optimizer_state, loss_value, train_metrics = self._train_step(
                    params,
                    optimizer_state,
                    batch,
                    rollout_key,
                )

                batch_loss = float(loss_value)
                batch_expl = float(train_metrics["expl_mean"])
                loss_sum += batch_loss
                expl_sum += batch_expl

                if iteration == 1 or (iteration % log_every) == 0:
                    print(
                        f"[ep {epoch:03d} it {iteration:04d}/{steps_per_epoch}] "
                        f"batch_loss={batch_loss:.6e} "
                        f"expl_mean={batch_expl:.6e} "
                    )

            denom = float(steps_per_epoch)
            train_loss = loss_sum / denom
            train_expl = expl_sum / denom

            eval_rng, eval_rollout_key = jax.random.split(eval_rng)
            eval_loss_value, eval_metrics = self._eval_step(params, eval_batch, eval_rollout_key)
            eval_loss = float(eval_loss_value)
            eval_expl = float(eval_metrics["expl_mean"])

            if cooldown_counter > 0:
                cooldown_counter -= 1

            if eval_loss < (best_eval_loss - plateau_threshold):
                bad_plateau_epochs = 0
            else:
                bad_plateau_epochs += 1

            if (
                bad_plateau_epochs >= plateau_patience
                and cooldown_counter == 0
                and current_lr > plateau_min_lr
            ):
                new_lr = max(current_lr * plateau_factor, plateau_min_lr)
                if new_lr < current_lr:
                    current_lr = new_lr
                    optimizer = self._make_optimizer(current_lr, weight_decay, grad_clip)
                    optimizer_state = optimizer.init(params)
                    self._optimizer = optimizer
                    self._train_step = self._build_train_step(optimizer)
                    cooldown_counter = plateau_cooldown
                    bad_plateau_epochs = 0
                    print(f"reduced learning rate to {current_lr:.2e}")

            print(
                f"[epoch {epoch:03d}] "
                f"train_loss={train_loss:.6e} | "
                f"train_expl={train_expl:.6e} | "
                f"eval_loss={eval_loss:.6e} | "
                f"eval_expl={eval_expl:.6e} | "
                f"lr={current_lr:.2e}"
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_expl": train_expl,
                    "eval_loss": eval_loss,
                    "eval_expl": eval_expl,
                    "lr": current_lr,
                }
            )

            improved = eval_loss < (best_eval_loss - early_stop_min_delta)
            if improved:
                best_eval_loss = eval_loss
                bad_epochs = 0
            else:
                bad_epochs += 1

            if save_path is not None and (epoch % save_every == 0 or epoch == num_epochs):
                self._save_checkpoint(
                    save_path,
                    epoch,
                    params,
                    optimizer_state,
                    training_params,
                    eval_loss,
                )

            if bad_epochs >= early_stop_patience:
                print(
                    f"early stopping at epoch {epoch:03d}: "
                    f"no eval_loss improvement > {early_stop_min_delta:.1e} "
                    f"for {bad_epochs} epoch(s)"
                )
                if save_path is not None:
                    self._save_checkpoint(
                        save_path,
                        epoch,
                        params,
                        optimizer_state,
                        training_params,
                        eval_loss,
                    )
                break

        return {
            "params": params,
            "optimizer_state": optimizer_state,
            "history": history,
            "best_eval_loss": best_eval_loss,
        }
