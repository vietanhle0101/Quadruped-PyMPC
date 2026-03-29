import argparse
import copy
import os
import pathlib
import sys
import time


def _requested_device_from_argv(argv):
    for idx, arg in enumerate(argv):
        if arg.startswith("--device="):
            return arg.split("=", 1)[1].lower()
        if arg == "--device" and idx + 1 < len(argv):
            return argv[idx + 1].lower()
    return "gpu"

if _requested_device_from_argv(sys.argv[1:]) == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import mujoco
import jax
import numpy as np
import yaml
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector
from gym_quadruped.utils.quadruped_utils import LegsAttr
from tqdm import tqdm

CURRENT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from quadruped_pympc import config as cfg
from quadruped_pympc.helpers.quadruped_utils import plot_swing_mujoco
from quadruped_pympc.quadruped_pympc_wrapper import QuadrupedPyMPC_Wrapper


def load_config(config_path: pathlib.Path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_repo_path(path: pathlib.Path):
    if path.is_absolute():
        return path
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate
    return CURRENT_DIR / path


def normalize_policy_config(policy_config: dict):
    normalized = dict(policy_config)
    if "num_layers" in normalized and normalized["num_layers"] is not None:
        normalized["num_layers"] = int(normalized["num_layers"])
    if "hidden_dim" in normalized and normalized["hidden_dim"] is not None:
        normalized["hidden_dim"] = int(normalized["hidden_dim"])
    if "activation" in normalized and normalized["activation"] is not None:
        normalized["activation"] = str(normalized["activation"])
    return normalized


def normalize_dmppi_config(dmppi_config: dict):
    normalized = dict(dmppi_config)
    if "dmppi_samples" in normalized and normalized["dmppi_samples"] is not None:
        normalized["dmppi_samples"] = int(normalized["dmppi_samples"])
    if "dmppi_temperature" in normalized and normalized["dmppi_temperature"] is not None:
        normalized["dmppi_temperature"] = float(normalized["dmppi_temperature"])
    return normalized


def configure_dmppi_controller(
    policy_file_path: pathlib.Path,
    policy_config: dict,
    device: str,
    dmppi_num_samples: int,
    dmppi_temperature: float,
):
    cfg.mpc_params["type"] = "dmppi"
    cfg.mpc_params["use_neural_updater"] = True
    cfg.mpc_params["device"] = device
    cfg.mpc_params["dpc_policy_path"] = str(policy_file_path)
    cfg.mpc_params["dpc_num_layers"] = policy_config.get("num_layers", 5)
    cfg.mpc_params["dpc_hidden_dim"] = policy_config.get("hidden_dim", 256)
    cfg.mpc_params["dpc_activation"] = policy_config.get("activation", "gelu")
    cfg.mpc_params["dmppi_samples"] = int(dmppi_num_samples)
    cfg.mpc_params["dmppi_temperature"] = float(dmppi_temperature)


def run_dmppi_test(
    policy_file_path,
    policy_config,
    qpympc_cfg,
    process=0,
    num_episodes=1,
    num_seconds_per_episode=30,
    ref_base_lin_vel=(0.0, 2.0),
    ref_base_ang_vel=(-0.4, 0.4),
    friction_coeff=(0.5, 1.0),
    base_vel_command_type="forward",
    goal_base_pos=None,
    goal_kp=0.5,
    goal_max_lin_vel=0.2,
    goal_position_tolerance=0.1,
    seed=0,
    render=True,
    device="gpu",
    dmppi_num_samples=64,
    dmppi_temperature=1.0,
):
    del process
    np.random.seed(seed)
    configure_dmppi_controller(
        policy_file_path,
        policy_config,
        device,
        dmppi_num_samples,
        dmppi_temperature,
    )

    print(f"Loaded policy checkpoint: {policy_file_path}")
    print(
        "Policy architecture: "
        f"NeuralGRFDistributionPolicy(num_layers={policy_config.get('num_layers', 5)}, "
        f"hidden_dim={policy_config.get('hidden_dim', 256)}, "
        f"activation='{policy_config.get('activation', 'gelu')}')"
    )
    print(f"Controller mode: {cfg.mpc_params['type']}")
    print(
        "DMPPI settings: "
        f"dmppi_samples={cfg.mpc_params['dmppi_samples']}, "
        f"dmppi_temperature={cfg.mpc_params['dmppi_temperature']}"
    )
    print("Neural updater: enabled")

    robot_name = qpympc_cfg.robot
    hip_height = qpympc_cfg.hip_height
    scene_name = qpympc_cfg.simulation_params["scene"]
    simulation_dt = qpympc_cfg.simulation_params["dt"]

    env = QuadrupedEnv(
        robot=robot_name,
        scene=scene_name,
        sim_dt=simulation_dt,
        ref_base_lin_vel=np.asarray(ref_base_lin_vel) * hip_height,
        ref_base_ang_vel=ref_base_ang_vel,
        ground_friction_coeff=friction_coeff,
        base_vel_command_type=base_vel_command_type,
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

    tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in range(4)])
    tau_soft_limits_scalar = 0.9
    tau_limits = LegsAttr(
        FL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FL] * tau_soft_limits_scalar,
        FR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FR] * tau_soft_limits_scalar,
        RL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RL] * tau_soft_limits_scalar,
        RR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RR] * tau_soft_limits_scalar,
    )

    feet_traj_geom_ids, feet_GRF_geom_ids = None, LegsAttr(FL=-1, FR=-1, RL=-1, RR=-1)
    legs_order = ("FL", "FR", "RL", "RR")
    goal_geom_id = -1

    wrapper = QuadrupedPyMPC_Wrapper(
        initial_feet_pos=env.feet_pos,
        legs_order=legs_order,
        feet_geom_id=env._feet_geom_id,
    )
    print(f"Controller device: {wrapper.srbd_controller_interface.controller.device}")

    goal_base_pos = None if goal_base_pos is None else np.asarray(goal_base_pos, dtype=float)
    if goal_base_pos is not None and goal_base_pos.shape not in {(2,), (3,)}:
        raise ValueError("goal_base_pos must be a 2D or 3D position.")

    if qpympc_cfg.simulation_params["visual_foothold_adaptation"] != "blind":
        from gym_quadruped.sensors.heightmap import HeightMap

        resolution_heightmap = 0.04
        num_rows_heightmap = 7
        num_cols_heightmap = 7
        heightmaps = LegsAttr(
            FL=HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, dist_x=resolution_heightmap, dist_y=resolution_heightmap, mj_model=env.mjModel, mj_data=env.mjData),
            FR=HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, dist_x=resolution_heightmap, dist_y=resolution_heightmap, mj_model=env.mjModel, mj_data=env.mjData),
            RL=HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, dist_x=resolution_heightmap, dist_y=resolution_heightmap, mj_model=env.mjModel, mj_data=env.mjData),
            RR=HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, dist_x=resolution_heightmap, dist_y=resolution_heightmap, mj_model=env.mjModel, mj_data=env.mjData),
        )
    else:
        heightmaps = None

    render_freq = 30
    steps_per_episode = int(num_seconds_per_episode // simulation_dt)
    last_render_time = time.time()

    for episode_num in range(num_episodes):
        for _ in tqdm(range(steps_per_episode), desc=f"Ep:{episode_num:d}-steps:", total=steps_per_episode):
            feet_pos = env.feet_pos(frame="world")
            feet_vel = env.feet_vel(frame="world")
            hip_pos = env.hip_positions(frame="world")
            base_lin_vel = env.base_lin_vel(frame="world")
            base_ang_vel = env.base_ang_vel(frame="base")
            base_ori_euler_xyz = env.base_ori_euler_xyz
            base_pos = copy.deepcopy(env.base_pos)
            com_pos = copy.deepcopy(env.com)

            if goal_base_pos is not None:
                goal_position_world = goal_base_pos if goal_base_pos.shape == (3,) else np.array(
                    [goal_base_pos[0], goal_base_pos[1], base_pos[2]],
                    dtype=float,
                )
                position_error = goal_position_world - base_pos
                position_error[2] = 0.0
                distance_to_goal = np.linalg.norm(position_error[:2])

                if distance_to_goal <= goal_position_tolerance:
                    ref_base_lin_vel_cmd = np.zeros(3)
                    ref_base_ang_vel_cmd = np.zeros(3)
                else:
                    ref_base_lin_vel_cmd = goal_kp * position_error
                    ref_base_lin_vel_cmd[2] = 0.0
                    planar_speed = np.linalg.norm(ref_base_lin_vel_cmd[:2])
                    if planar_speed > goal_max_lin_vel:
                        ref_base_lin_vel_cmd[:2] *= goal_max_lin_vel / planar_speed
                    ref_base_ang_vel_cmd = np.zeros(3)
            else:
                ref_base_lin_vel_cmd, ref_base_ang_vel_cmd = env.target_base_vel()
                distance_to_goal = None
                goal_position_world = None

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
                legs_order,
                simulation_dt,
                ref_base_lin_vel_cmd,
                ref_base_ang_vel_cmd,
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

            for leg in legs_order:
                tau_min, tau_max = tau_limits[leg][:, 0], tau_limits[leg][:, 1]
                tau[leg] = np.clip(tau[leg], tau_min, tau_max)

            action = np.zeros(env.mjModel.nu)
            action[env.legs_tau_idx.FL] = tau.FL
            action[env.legs_tau_idx.FR] = tau.FR
            action[env.legs_tau_idx.RL] = tau.RL
            action[env.legs_tau_idx.RR] = tau.RR

            _, _, is_terminated, is_truncated, _ = env.step(action=action)

            if render and (time.time() - last_render_time > 1.0 / render_freq or env.step_num == 1):
                _, _, feet_GRF = env.feet_contact_state(ground_reaction_forces=True)

                feet_traj_geom_ids = plot_swing_mujoco(
                    viewer=env.viewer,
                    swing_traj_controller=wrapper.wb_interface.stc,
                    swing_period=wrapper.wb_interface.stc.swing_period,
                    swing_time=LegsAttr(
                        FL=wrapper.wb_interface.stc.swing_time[0],
                        FR=wrapper.wb_interface.stc.swing_time[1],
                        RL=wrapper.wb_interface.stc.swing_time[2],
                        RR=wrapper.wb_interface.stc.swing_time[3],
                    ),
                    lift_off_positions=wrapper.wb_interface.frg.lift_off_positions,
                    nmpc_footholds=wrapper.nmpc_footholds,
                    ref_feet_pos=wrapper.nmpc_footholds,
                    early_stance_detector=wrapper.wb_interface.esd,
                    geom_ids=feet_traj_geom_ids,
                )

                for leg_name in legs_order:
                    feet_GRF_geom_ids[leg_name] = render_vector(
                        env.viewer,
                        vector=feet_GRF[leg_name],
                        pos=feet_pos[leg_name],
                        scale=np.linalg.norm(feet_GRF[leg_name]) * 0.005,
                        color=np.array([0, 1, 0, 0.5]),
                        geom_id=feet_GRF_geom_ids[leg_name],
                    )

                if goal_position_world is not None:
                    goal_geom_id = render_sphere(
                        viewer=env.viewer,
                        position=goal_position_world,
                        diameter=0.18,
                        color=[1, 0, 0, 0.75],
                        geom_id=goal_geom_id,
                    )

                env.render()
                last_render_time = time.time()

            reached_goal = goal_base_pos is not None and distance_to_goal is not None and distance_to_goal <= goal_position_tolerance
            if reached_goal:
                print(f"Goal reached at {base_pos[:2]}")
                env.close()
                return

            if is_terminated or is_truncated or env.step_num >= steps_per_episode:
                if is_terminated:
                    print("Environment terminated")
                    env.close()
                    return
                env.reset(random=True)
                wrapper.reset(initial_feet_pos=env.feet_pos(frame="world"))
                break

    env.close()


def main():
    default_policy_file = REPO_ROOT / "training" / "neural_dmppi_constrained_policy.pkl"
    default_config_path = REPO_ROOT / "training" / "dmppi_neural_config.yaml"
    parser = argparse.ArgumentParser(description="Test a trained neural-updater DMPPI policy through QuadrupedPyMPC_Wrapper.")
    parser.add_argument("--policy_file", type=pathlib.Path, default=default_policy_file, help="Path to the trained neural DMPPI checkpoint.")
    parser.add_argument("--config", type=pathlib.Path, default=default_config_path, help="Path to the neural DMPPI YAML config.")
    parser.add_argument("--device", type=str, default="gpu", choices=("cpu", "gpu"), help="JAX device preference for the DMPPI policy.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--goal-x", type=float, default=1.0, help="Goal x position in world frame.")
    parser.add_argument("--goal-y", type=float, default=0.0, help="Goal y position in world frame.")
    parser.add_argument(
        "--goal-max-lin-vel",
        type=float,
        default=0.1,
        help="Maximum planar linear velocity command used when driving toward the goal.",
    )
    parser.add_argument("--dmppi-num-samples", type=int, default=None, help="Override the number of one-step DMPPI force samples.")
    parser.add_argument("--dmppi-temperature", type=float, default=None, help="Override the softmax temperature for DMPPI weighting.")
    parser.add_argument("--no-render", action="store_true", help="Disable Mujoco rendering.")
    args = parser.parse_args()

    policy_file_path = resolve_repo_path(args.policy_file)
    config_path = resolve_repo_path(args.config)

    config = load_config(config_path)
    policy_config = normalize_policy_config(dict(config.get("policy", {})))
    dmppi_config = normalize_dmppi_config(dict(config.get("dmppi", {})))
    goal_base_pos = np.array([args.goal_x, args.goal_y, cfg.simulation_params["ref_z"]], dtype=float)

    run_dmppi_test(
        policy_file_path=policy_file_path,
        policy_config=policy_config,
        qpympc_cfg=cfg,
        goal_base_pos=goal_base_pos,
        goal_max_lin_vel=args.goal_max_lin_vel,
        seed=args.seed,
        render=not args.no_render,
        device=args.device,
        dmppi_num_samples=(
            args.dmppi_num_samples
            if args.dmppi_num_samples is not None
            else dmppi_config.get("dmppi_samples", 64)
        ),
        dmppi_temperature=(
            args.dmppi_temperature
            if args.dmppi_temperature is not None
            else dmppi_config.get("dmppi_temperature", 0.2)
        ),
    )


if __name__ == "__main__":
    main()
