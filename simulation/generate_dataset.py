"""
generate_dataset.py is a data-collection script for DPC training. 
It uses the existing MPC controller to generate supervised training samples 
by rolling out many simulated episodes and logging the data.
"""

import copy
import argparse
import pathlib
from datetime import datetime

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.quadruped_utils import LegsAttr

from quadruped_pympc.controllers.dpc.dpc_solver import DPC
from quadruped_pympc.quadruped_pympc_wrapper import QuadrupedPyMPC_Wrapper


def build_heightmaps(env, qpympc_cfg):
    """Create heightmap sensors when foothold adaptation requires them."""
    if qpympc_cfg.simulation_params["visual_foothold_adaptation"] == "blind":
        return None

    from gym_quadruped.sensors.heightmap import HeightMap

    resolution_heightmap = 0.04
    num_rows_heightmap = 7
    num_cols_heightmap = 7
    return LegsAttr(
        FL=HeightMap(
            num_rows=num_rows_heightmap,
            num_cols=num_cols_heightmap,
            dist_x=resolution_heightmap,
            dist_y=resolution_heightmap,
            mj_model=env.mjModel,
            mj_data=env.mjData,
        ),
        FR=HeightMap(
            num_rows=num_rows_heightmap,
            num_cols=num_cols_heightmap,
            dist_x=resolution_heightmap,
            dist_y=resolution_heightmap,
            mj_model=env.mjModel,
            mj_data=env.mjData,
        ),
        RL=HeightMap(
            num_rows=num_rows_heightmap,
            num_cols=num_cols_heightmap,
            dist_x=resolution_heightmap,
            dist_y=resolution_heightmap,
            mj_model=env.mjModel,
            mj_data=env.mjData,
        ),
        RR=HeightMap(
            num_rows=num_rows_heightmap,
            num_cols=num_cols_heightmap,
            dist_x=resolution_heightmap,
            dist_y=resolution_heightmap,
            mj_model=env.mjModel,
            mj_data=env.mjData,
        ),
    )


def sample_initial_condition(env, qpympc_cfg, rng):
    """Sample a mild random initial state for dataset generation."""
    ref_z = qpympc_cfg.simulation_params["ref_z"]
    base_pos = np.array(
        [
            rng.uniform(-0.3, 0.3),
            rng.uniform(-0.3, 0.3),
            ref_z + rng.uniform(-0.02, 0.02),
        ],
        dtype=float,
    )
    base_rpy = np.array(
        [
            rng.uniform(-0.08, 0.08),
            rng.uniform(-0.08, 0.08),
            rng.uniform(-np.pi, np.pi),
        ],
        dtype=float,
    )
    base_lin_vel = np.array(
        [
            rng.uniform(-0.2, 0.2),
            rng.uniform(-0.2, 0.2),
            0.0,
        ],
        dtype=float,
    )
    base_ang_vel = np.array(
        [
            rng.uniform(-0.2, 0.2),
            rng.uniform(-0.2, 0.2),
            rng.uniform(-0.3, 0.3),
        ],
        dtype=float,
    )
    joint_pos = LegsAttr(
        FL=env.mjData.qpos[env.legs_qpos_idx.FL].copy() + rng.uniform(-0.12, 0.12, size=3),
        FR=env.mjData.qpos[env.legs_qpos_idx.FR].copy() + rng.uniform(-0.12, 0.12, size=3),
        RL=env.mjData.qpos[env.legs_qpos_idx.RL].copy() + rng.uniform(-0.12, 0.12, size=3),
        RR=env.mjData.qpos[env.legs_qpos_idx.RR].copy() + rng.uniform(-0.12, 0.12, size=3),
    )
    joint_vel = LegsAttr(
        FL=rng.uniform(-0.2, 0.2, size=3),
        FR=rng.uniform(-0.2, 0.2, size=3),
        RL=rng.uniform(-0.2, 0.2, size=3),
        RR=rng.uniform(-0.2, 0.2, size=3),
    )
    return {
        "base_pos": base_pos,
        "base_rpy": base_rpy,
        "base_lin_vel": base_lin_vel,
        "base_ang_vel": base_ang_vel,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
    }


def sample_goal(qpympc_cfg, rng, initial_base_pos):
    """Sample a planar goal around the rollout's initial position."""
    ref_z = qpympc_cfg.simulation_params["ref_z"]
    return np.array(
        [
            initial_base_pos[0] + rng.uniform(-5.0, 5.0),
            initial_base_pos[1] + rng.uniform(-3.0, 3.0),
            ref_z,
        ],
        dtype=float,
    )


def set_initial_state(env, initial_condition):
    """Apply a sampled initial state directly to Mujoco data."""
    quat_xyzw = Rotation.from_euler("xyz", initial_condition["base_rpy"]).as_quat()
    quat_wxyz = np.roll(quat_xyzw, 1)

    env.mjData.qpos[0:3] = initial_condition["base_pos"]
    env.mjData.qpos[3:7] = quat_wxyz
    env.mjData.qvel[0:3] = initial_condition["base_lin_vel"]
    env.mjData.qvel[3:6] = initial_condition["base_ang_vel"]

    env.mjData.qpos[env.legs_qpos_idx.FL] = initial_condition["joint_pos"].FL
    env.mjData.qpos[env.legs_qpos_idx.FR] = initial_condition["joint_pos"].FR
    env.mjData.qpos[env.legs_qpos_idx.RL] = initial_condition["joint_pos"].RL
    env.mjData.qpos[env.legs_qpos_idx.RR] = initial_condition["joint_pos"].RR

    env.mjData.qvel[env.legs_qvel_idx.FL] = initial_condition["joint_vel"].FL
    env.mjData.qvel[env.legs_qvel_idx.FR] = initial_condition["joint_vel"].FR
    env.mjData.qvel[env.legs_qvel_idx.RL] = initial_condition["joint_vel"].RL
    env.mjData.qvel[env.legs_qvel_idx.RR] = initial_condition["joint_vel"].RR
    mujoco.mj_forward(env.mjModel, env.mjData)


def pack_training_sample(dpc_solver, wrapper, current_contact):
    """Convert cached SRBD-style dictionaries into packed DPC tensors."""
    current_state, reference = dpc_solver.prepare_state_and_reference(
        wrapper.latest_state_current,
        wrapper.latest_ref_state,
        current_contact,
        current_contact,
    )
    contact_sequence = np.asarray(wrapper.latest_contact_sequence, dtype=np.float32)
    reference_trajectory = np.repeat(
        np.asarray(reference, dtype=np.float32)[None, :],
        contact_sequence.shape[1],
        axis=0,
    )
    return {
        "current_centroidal_state": np.asarray(current_state, dtype=np.float32),
        "reference_state_horizon": reference_trajectory,
        "reference": np.asarray(reference, dtype=np.float32),
        "contact_sequence": contact_sequence,
    }


def generate_dpc_dataset(
    qpympc_cfg,
    num_episodes=50,
    num_seconds_per_rollout=10.0,
    render=False,
    seed=0,
    output_path: str | pathlib.Path = "datasets/dpc_dataset.npz",
):
    """Generate DPC training data by rolling out the existing MPC controller.

    For each rollout:
    - sample a random initial condition
    - sample a random goal
    - run until the robot terminates, reaches the goal, or the safety timeout hits
    - log packed centroidal state, horizon reference, and contact sequence each step
    """
    rng = np.random.default_rng(seed)

    env = QuadrupedEnv(
        robot=qpympc_cfg.robot,
        scene=qpympc_cfg.simulation_params["scene"],
        sim_dt=qpympc_cfg.simulation_params["dt"],
        ref_base_lin_vel=(0.0, 0.0),
        ref_base_ang_vel=(0.0, 0.0),
        ground_friction_coeff=(0.5, 1.0),
        base_vel_command_type="forward",
        state_obs_names=tuple(),
    )
    env.mjModel.opt.gravity[2] = -qpympc_cfg.gravity_constant

    if qpympc_cfg.qpos0_js is not None:
        env.mjModel.qpos0 = np.concatenate((env.mjModel.qpos0[:7], qpympc_cfg.qpos0_js))

    env.reset(random=False)
    if render:
        env.render()
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    heightmaps = build_heightmaps(env, qpympc_cfg)

    tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in range(4)])
    tau_soft_limits_scalar = 0.9
    tau_limits = LegsAttr(
        FL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FL] * tau_soft_limits_scalar,
        FR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FR] * tau_soft_limits_scalar,
        RL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RL] * tau_soft_limits_scalar,
        RR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RR] * tau_soft_limits_scalar,
    )
    legs_order = ["FL", "FR", "RL", "RR"]

    wrapper = QuadrupedPyMPC_Wrapper(
        initial_feet_pos=env.feet_pos,
        legs_order=tuple(legs_order),
        feet_geom_id=env._feet_geom_id,
    )
    dpc_solver = DPC()

    simulation_dt = qpympc_cfg.simulation_params["dt"]
    max_steps = int(num_seconds_per_rollout // simulation_dt)

    dataset = {
        "current_centroidal_state": [],
        "reference_state_horizon": [],
        "reference": [],
        "contact_sequence": [],
        "goal_base_pos": [],
        "initial_base_pos": [],
        "initial_base_rpy": [],
        "rollout_id": [],
        "time_index": [],
        "termination_code": [],
    }

    for rollout_id in range(num_episodes):
        env.reset(random=False)
        initial_condition = sample_initial_condition(env, qpympc_cfg, rng)
        goal_base_pos = sample_goal(qpympc_cfg, rng, initial_condition["base_pos"])
        set_initial_state(env, initial_condition)
        wrapper.reset(initial_feet_pos=env.feet_pos(frame="world"))
        termination_code = 3
        num_logged_steps = 0

        for step_idx in tqdm(range(max_steps), desc=f"Rollout {rollout_id}", total=max_steps):
            feet_pos = env.feet_pos(frame="world")
            feet_vel = env.feet_vel(frame="world")
            hip_pos = env.hip_positions(frame="world")
            base_lin_vel = env.base_lin_vel(frame="world")
            base_ang_vel = env.base_ang_vel(frame="base")
            base_ori_euler_xyz = env.base_ori_euler_xyz
            base_pos = copy.deepcopy(env.base_pos)
            com_pos = copy.deepcopy(env.com)

            position_error = goal_base_pos - base_pos
            position_error[2] = 0.0
            distance_to_goal = np.linalg.norm(position_error[:2])
            if distance_to_goal <= 0.1:
                ref_base_lin_vel = np.zeros(3)
                ref_base_ang_vel = np.zeros(3)
            else:
                ref_base_lin_vel = 0.5 * position_error
                ref_base_lin_vel[2] = 0.0
                planar_speed = np.linalg.norm(ref_base_lin_vel[:2])
                if planar_speed > 0.5:
                    ref_base_lin_vel[:2] *= 0.5 / planar_speed
                ref_base_ang_vel = np.zeros(3)

            if qpympc_cfg.simulation_params["use_inertia_recomputation"]:
                inertia = env.get_base_inertia().flatten()
            else:
                inertia = qpympc_cfg.inertia.flatten()

            qpos, qvel = env.mjData.qpos, env.mjData.qvel
            legs_qvel_idx = env.legs_qvel_idx
            legs_qpos_idx = env.legs_qpos_idx
            joints_pos = LegsAttr(
                FL=qpos[legs_qpos_idx.FL].copy(),
                FR=qpos[legs_qpos_idx.FR].copy(),
                RL=qpos[legs_qpos_idx.RL].copy(),
                RR=qpos[legs_qpos_idx.RR].copy(),
            )

            legs_mass_matrix = env.legs_mass_matrix
            legs_qfrc_bias = env.legs_qfrc_bias
            legs_qfrc_passive = env.legs_qfrc_passive
            feet_jac = env.feet_jacobians(frame="world", return_rot_jac=False)
            feet_jac_dot = env.feet_jacobians_dot(frame="world", return_rot_jac=False)

            tau = wrapper.compute_actions(
                com_pos,
                base_pos,
                base_lin_vel,
                base_ori_euler_xyz,
                base_ang_vel,
                feet_pos,
                hip_pos,
                joints_pos,
                heightmaps,
                tuple(legs_order),
                simulation_dt,
                ref_base_lin_vel,
                ref_base_ang_vel,
                env.step_num,
                qpos,
                qvel,
                feet_jac,
                feet_jac_dot,
                feet_vel,
                legs_qfrc_passive,
                legs_qfrc_bias,
                legs_mass_matrix,
                legs_qpos_idx,
                legs_qvel_idx,
                tau,
                inertia,
                env.mjData.contact,
            )

            current_contact = np.array(
                [
                    wrapper.latest_contact_sequence[0][0],
                    wrapper.latest_contact_sequence[1][0],
                    wrapper.latest_contact_sequence[2][0],
                    wrapper.latest_contact_sequence[3][0],
                ],
                dtype=np.float32,
            )
            sample = pack_training_sample(dpc_solver, wrapper, current_contact)
            dataset["current_centroidal_state"].append(sample["current_centroidal_state"])
            dataset["reference_state_horizon"].append(sample["reference_state_horizon"])
            dataset["reference"].append(sample["reference"])
            dataset["contact_sequence"].append(sample["contact_sequence"])
            dataset["goal_base_pos"].append(goal_base_pos.astype(np.float32))
            dataset["initial_base_pos"].append(initial_condition["base_pos"].astype(np.float32))
            dataset["initial_base_rpy"].append(initial_condition["base_rpy"].astype(np.float32))
            dataset["rollout_id"].append(rollout_id)
            dataset["time_index"].append(step_idx)
            num_logged_steps += 1

            for leg in legs_order:
                tau_min, tau_max = tau_limits[leg][:, 0], tau_limits[leg][:, 1]
                tau[leg] = np.clip(tau[leg], tau_min, tau_max)

            action = np.zeros(env.mjModel.nu)
            action[env.legs_tau_idx.FL] = tau.FL
            action[env.legs_tau_idx.FR] = tau.FR
            action[env.legs_tau_idx.RL] = tau.RL
            action[env.legs_tau_idx.RR] = tau.RR

            _, _, is_terminated, is_truncated, _ = env.step(action=action)

            if render:
                env.render()

            if is_terminated:
                termination_code = 1
                break
            if distance_to_goal <= 0.1:
                termination_code = 0
                break
            if is_truncated:
                termination_code = 2
                break

        dataset["termination_code"].extend([termination_code] * num_logged_steps)

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        current_centroidal_state=np.stack(dataset["current_centroidal_state"], axis=0),
        reference_state_horizon=np.stack(dataset["reference_state_horizon"], axis=0),
        reference=np.stack(dataset["reference"], axis=0),
        contact_sequence=np.stack(dataset["contact_sequence"], axis=0),
        goal_base_pos=np.stack(dataset["goal_base_pos"], axis=0),
        initial_base_pos=np.stack(dataset["initial_base_pos"], axis=0),
        initial_base_rpy=np.stack(dataset["initial_base_rpy"], axis=0),
        rollout_id=np.asarray(dataset["rollout_id"], dtype=np.int32),
        time_index=np.asarray(dataset["time_index"], dtype=np.int32),
        termination_code=np.asarray(dataset["termination_code"], dtype=np.int32),
    )

    env.close()
    print(f"Collected {len(dataset['current_centroidal_state'])} data samples.")
    return output_path


if __name__ == "__main__":
    from quadruped_pympc import config as cfg

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output_path = pathlib.Path(__file__).parent.parent / "datasets" / f"dpc_dataset_{timestamp}.npz"

    parser = argparse.ArgumentParser(description="Generate a DPC training dataset from MPC rollouts.")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of rollout episodes to collect.")
    parser.add_argument(
        "--num-seconds-per-rollout",
        type=float,
        default=10.0,
        help="Maximum simulated seconds per rollout.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for dataset generation.")
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        default=default_output_path,
        help="Output .npz dataset path.",
    )
    parser.add_argument("--render", action="store_true", help="Render the Mujoco rollout while collecting data.")
    args = parser.parse_args()

    dataset_path = generate_dpc_dataset(
        qpympc_cfg=cfg,
        num_episodes=args.num_episodes,
        num_seconds_per_rollout=args.num_seconds_per_rollout,
        render=args.render,
        seed=args.seed,
        output_path=args.output_path,
    )
    print(f"Saved DPC dataset to: {dataset_path}")
