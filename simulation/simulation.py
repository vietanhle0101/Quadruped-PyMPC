# Description: This script is used to simulate the full model of the robot in mujoco
import argparse
import pathlib

# Authors:
# Giulio Turrisi, Daniel Ordonez
import time
from os import PathLike
from pprint import pprint

import copy
import numpy as np

import mujoco

# Gym and Simulation related imports
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector
from gym_quadruped.utils.quadruped_utils import LegsAttr
from tqdm import tqdm

# Helper functions for plotting
from quadruped_pympc.helpers.quadruped_utils import plot_swing_mujoco

# PyMPC controller imports
from quadruped_pympc.quadruped_pympc_wrapper import QuadrupedPyMPC_Wrapper


def print_controller_config(qpympc_cfg):
    """Print the active controller configuration for quick consistency checks."""
    print("Active controller configuration:")
    print(f"  robot: {qpympc_cfg.robot}")
    print(f"  mass: {qpympc_cfg.mass if hasattr(qpympc_cfg, 'mass') else 'n/a'}")
    print(f"  gravity_constant: {qpympc_cfg.gravity_constant}")
    print(f"  simulation_dt: {qpympc_cfg.simulation_params['dt']}")
    print(f"  mpc_frequency: {qpympc_cfg.simulation_params['mpc_frequency']}")
    print("  mpc_params:")
    pprint(qpympc_cfg.mpc_params, sort_dicts=False)


def run_simulation(
    qpympc_cfg,
    process=0,
    num_episodes=500,
    num_seconds_per_episode=60,
    ref_base_lin_vel=(0.0, 4.0),
    ref_base_ang_vel=(-0.4, 0.4),
    friction_coeff=(0.5, 1.0),
    base_vel_command_type="human", 
    goal_base_pos=None, # If ``goal_base_pos`` is provided, the function does not rely on keyboard arrows.
    goal_kp=0.5,
    goal_max_lin_vel=0.2,
    goal_position_tolerance=0.1,
    stop_at_goal=False,
    seed=0,
    render=True,
    recording_path: PathLike = None,
):
    """
    Run the Mujoco simulation with the PyMPC controller.
    """
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(seed)

    # Extract the robot- and scene-specific parameters once so the inner loop
    # only deals with state updates and control.
    robot_name = qpympc_cfg.robot
    hip_height = qpympc_cfg.hip_height
    robot_leg_joints = qpympc_cfg.robot_leg_joints
    robot_feet_geom_names = qpympc_cfg.robot_feet_geom_names
    scene_name = qpympc_cfg.simulation_params["scene"]
    simulation_dt = qpympc_cfg.simulation_params["dt"]

    # Request only the observables that are needed externally. The controller
    # pulls most of its inputs directly from the environment object.
    state_obs_names = [] #list(QuadrupedEnv.ALL_OBS) # + list(IMU.ALL_OBS)

    # Build the Mujoco environment. This wraps the robot model, terrain,
    # simulator timing, and the command interface used by the controller.
    env = QuadrupedEnv(
        robot=robot_name,
        scene=scene_name,
        sim_dt=simulation_dt,
        ref_base_lin_vel=np.asarray(ref_base_lin_vel) * hip_height,  # pass a float for a fixed value
        ref_base_ang_vel=ref_base_ang_vel,  # pass a float for a fixed value
        ground_friction_coeff=friction_coeff,  # pass a float for a fixed value
        base_vel_command_type=base_vel_command_type,  # "forward", "random", "forward+rotate", "human"
        state_obs_names=tuple(state_obs_names),  # Desired quantities in the 'state' vec
    )
    # pprint(env.get_hyperparameters())
    env.mjModel.opt.gravity[2] = -qpympc_cfg.gravity_constant

    # Some robots require a change in the zero joint-space configuration. If provided apply it
    if qpympc_cfg.qpos0_js is not None:
        env.mjModel.qpos0 = np.concatenate((env.mjModel.qpos0[:7], qpympc_cfg.qpos0_js))

    env.reset(random=False)
    if render:
        env.render()  # Pass in the first render call any mujoco.viewer.KeyCallbackType
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

    # ---- Runtime buffers and visualization state ---------------------------------------------

    # Per-leg torque buffer reused across iterations.
    tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in range(4)])
    # Soft actuator limits keep the commanded torques slightly below the hard bounds.
    tau_soft_limits_scalar = 0.9
    tau_limits = LegsAttr(
        FL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FL] * tau_soft_limits_scalar,
        FR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FR] * tau_soft_limits_scalar,
        RL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RL] * tau_soft_limits_scalar,
        RR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RR] * tau_soft_limits_scalar,
    )

    # Keep a fixed leg ordering because the controller, visualizer, and logs all
    # assume the same FL/FR/RL/RR convention.
    feet_traj_geom_ids, feet_GRF_geom_ids = None, LegsAttr(FL=-1, FR=-1, RL=-1, RR=-1)
    legs_order = ["FL", "FR", "RL", "RR"]
    goal_geom_id = -1

    # Accept both [x, y] and [x, y, z] goals. When only XY is provided, the
    # current base height is reused so the command remains planar.
    goal_base_pos = None if goal_base_pos is None else np.asarray(goal_base_pos, dtype=float)
    if goal_base_pos is not None and goal_base_pos.shape not in {(2,), (3,)}:
        raise ValueError("goal_base_pos must be a 2D or 3D position.")

    # Create HeightMap -----------------------------------------------------------------------
    if qpympc_cfg.simulation_params["visual_foothold_adaptation"] != "blind":
        from gym_quadruped.sensors.heightmap import HeightMap

        resolution_heightmap = 0.04
        num_rows_heightmap = 7
        num_cols_heightmap = 7
        heightmaps = LegsAttr(
            FL=HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, 
                         dist_x=resolution_heightmap, dist_y=resolution_heightmap, 
                         mj_model=env.mjModel, mj_data=env.mjData),
            FR=HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, 
                         dist_x=resolution_heightmap, dist_y=resolution_heightmap, 
                         mj_model=env.mjModel, mj_data=env.mjData),
            RL=HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, 
                         dist_x=resolution_heightmap, dist_y=resolution_heightmap, 
                         mj_model=env.mjModel, mj_data=env.mjData),
            RR=HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, 
                         dist_x=resolution_heightmap, dist_y=resolution_heightmap, 
                         mj_model=env.mjModel, mj_data=env.mjData),
        )
    else:
        heightmaps = None

    # Define which controller-side quantities should be exposed for plotting and logging.
    quadrupedpympc_observables_names = (
        "ref_base_height",
        "ref_base_angles",
        "ref_feet_pos",
        "nmpc_GRFs",
        "nmpc_footholds",
        "swing_time",
        "phase_signal",
        "lift_off_positions",
        # "base_lin_vel_err",
        # "base_ang_vel_err",
        # "base_poz_z_err",
    )

    quadrupedpympc_wrapper = QuadrupedPyMPC_Wrapper(
        initial_feet_pos=env.feet_pos,
        legs_order=tuple(legs_order),
        feet_geom_id=env._feet_geom_id,
        quadrupedpympc_observables_names=quadrupedpympc_observables_names,
    )

    # Optional trajectory logging to HDF5. This is outside the control path and
    # only activated when the caller requests a recording directory.
    if recording_path is not None:
        from gym_quadruped.utils.data.h5py import H5Writer

        root_path = pathlib.Path(recording_path)
        root_path.mkdir(exist_ok=True)
        dataset_path = (
            root_path
            / f"{robot_name}/{scene_name}"
            / f"lin_vel={ref_base_lin_vel} ang_vel={ref_base_ang_vel} friction={friction_coeff}"
            / f"ep={num_episodes}_steps={int(num_seconds_per_episode // simulation_dt):d}.h5"
        )
        h5py_writer = H5Writer(
            file_path=dataset_path,
            env=env,
            extra_obs=None,  # TODO: Make this automatically configured. Not hardcoded
        )
        print(f"\n Recording data to: {dataset_path.absolute()}")
    else:
        h5py_writer = None

    # ---- Episode and render scheduling -------------------------------------------------------
    RENDER_FREQ = 30  # Hz
    N_EPISODES = num_episodes
    N_STEPS_PER_EPISODE = int(num_seconds_per_episode // simulation_dt)
    last_render_time = time.time()

    state_obs_history, ctrl_state_history = [], []
    goal_reached = False
    terminated_early = False
    for episode_num in range(N_EPISODES):
        ep_state_history, ep_ctrl_state_history, ep_time = [], [], []
        for _ in tqdm(range(N_STEPS_PER_EPISODE), desc=f"Ep:{episode_num:d}-steps:", total=N_STEPS_PER_EPISODE):
            # ---- Read the current simulator state ------------------------------------------------
            feet_pos = env.feet_pos(frame="world")
            feet_vel = env.feet_vel(frame='world')
            hip_pos = env.hip_positions(frame="world")
            base_lin_vel = env.base_lin_vel(frame="world")
            base_ang_vel = env.base_ang_vel(frame="base")
            base_ori_euler_xyz = env.base_ori_euler_xyz
            base_pos = copy.deepcopy(env.base_pos)
            com_pos = copy.deepcopy(env.com)

            # ---- Build the reference command -----------------------------------------------------
            # The downstream controller expects velocity references. In goal mode we
            # convert position error into a bounded planar velocity command.
            if goal_base_pos is not None:
                goal_position_world = goal_base_pos if goal_base_pos.shape == (3,) else np.array(
                    [goal_base_pos[0], goal_base_pos[1], base_pos[2]],
                    dtype=float,
                )
                position_error = goal_position_world - base_pos
                position_error[2] = 0.0
                distance_to_goal = np.linalg.norm(position_error[:2])

                if distance_to_goal <= goal_position_tolerance:
                    # Inside the tolerance ball, command zero velocity so the robot settles.
                    ref_base_lin_vel = np.zeros(3)
                    ref_base_ang_vel = np.zeros(3)
                else:
                    # Simple proportional guidance with a speed cap.
                    ref_base_lin_vel = goal_kp * position_error
                    ref_base_lin_vel[2] = 0.0
                    planar_speed = np.linalg.norm(ref_base_lin_vel[:2])
                    if planar_speed > goal_max_lin_vel:
                        ref_base_lin_vel[:2] *= goal_max_lin_vel / planar_speed
                    ref_base_ang_vel = np.zeros(3)
            else:
                ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()

            # ---- Gather dynamics terms required by the controller --------------------------------
            if qpympc_cfg.simulation_params["use_inertia_recomputation"]:
                inertia = env.get_base_inertia().flatten()  # Reflected inertia of base at qpos, in world frame
            else:
                inertia = qpympc_cfg.inertia.flatten()

            # Get the qpos and qvel
            qpos, qvel = env.mjData.qpos, env.mjData.qvel
            # Idx of the leg
            legs_qvel_idx = env.legs_qvel_idx  # leg_name: [idx1, idx2, idx3] ...
            legs_qpos_idx = env.legs_qpos_idx  # leg_name: [idx1, idx2, idx3] ...
            joints_pos = LegsAttr(
                FL=qpos[legs_qpos_idx.FL].copy(),
                FR=qpos[legs_qpos_idx.FR].copy(),
                RL=qpos[legs_qpos_idx.RL].copy(),
                RR=qpos[legs_qpos_idx.RR].copy(),
            )

            # Get Centrifugal, Coriolis, Gravity, Friction for the swing controller
            legs_mass_matrix = env.legs_mass_matrix
            legs_qfrc_bias = env.legs_qfrc_bias
            legs_qfrc_passive = env.legs_qfrc_passive

            # Compute feet jacobians
            feet_jac = env.feet_jacobians(frame='world', return_rot_jac=False)
            feet_jac_dot = env.feet_jacobians_dot(frame='world', return_rot_jac=False)

            # ---- Solve the control problem -------------------------------------------------------
            # The wrapper encapsulates gait generation, foothold planning, and MPC.
            tau = quadrupedpympc_wrapper.compute_actions(
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
            # Saturate the torques before sending them to Mujoco.
            for leg in ["FL", "FR", "RL", "RR"]:
                tau_min, tau_max = tau_limits[leg][:, 0], tau_limits[leg][:, 1]
                tau[leg] = np.clip(tau[leg], tau_min, tau_max)

            # Convert the per-leg torques into the flat actuator vector expected by the env.
            action = np.zeros(env.mjModel.nu)
            action[env.legs_tau_idx.FL] = tau.FL
            action[env.legs_tau_idx.FR] = tau.FR
            action[env.legs_tau_idx.RL] = tau.RL
            action[env.legs_tau_idx.RR] = tau.RR

            # Advance the simulation by one step using the newly computed torques.
            state, reward, is_terminated, is_truncated, info = env.step(action=action)

            # Read back controller internals for plotting and optional recording.
            ctrl_state = quadrupedpympc_wrapper.get_obs()

            # Persist both raw env state and controller-side observables for this episode.
            base_poz_z_err = ctrl_state["ref_base_height"] - base_pos[2]
            ctrl_state["base_poz_z_err"] = base_poz_z_err

            ep_state_history.append(state)
            ep_time.append(env.simulation_time)
            ep_ctrl_state_history.append(ctrl_state)

            # ---- Visualization ------------------------------------------------------------------
            if render and (time.time() - last_render_time > 1.0 / RENDER_FREQ or env.step_num == 1):
                _, _, feet_GRF = env.feet_contact_state(ground_reaction_forces=True)

                # Plot the swing trajectory
                feet_traj_geom_ids = plot_swing_mujoco(
                    viewer=env.viewer,
                    swing_traj_controller=quadrupedpympc_wrapper.wb_interface.stc,
                    swing_period=quadrupedpympc_wrapper.wb_interface.stc.swing_period,
                    swing_time=LegsAttr(
                        FL=ctrl_state["swing_time"][0],
                        FR=ctrl_state["swing_time"][1],
                        RL=ctrl_state["swing_time"][2],
                        RR=ctrl_state["swing_time"][3],
                    ),
                    lift_off_positions=ctrl_state["lift_off_positions"],
                    nmpc_footholds=ctrl_state["nmpc_footholds"],
                    ref_feet_pos=ctrl_state["ref_feet_pos"],
                    early_stance_detector=quadrupedpympc_wrapper.wb_interface.esd,
                    geom_ids=feet_traj_geom_ids,
                )

                # Update and Plot the heightmap
                if qpympc_cfg.simulation_params["visual_foothold_adaptation"] != "blind":
                    # if(stc.check_apex_condition(current_contact, interval=0.01)):
                    for leg_id, leg_name in enumerate(legs_order):
                        data = heightmaps[
                            leg_name
                        ].data  # .update_height_map(ref_feet_pos[leg_name], yaw=env.base_ori_euler_xyz[2])
                        if data is not None:
                            for i in range(data.shape[0]):
                                for j in range(data.shape[1]):
                                    heightmaps[leg_name].geom_ids[i, j] = render_sphere(
                                        viewer=env.viewer,
                                        position=([data[i][j][0][0], data[i][j][0][1], data[i][j][0][2]]),
                                        diameter=0.01,
                                        color=[0, 1, 0, 0.5],
                                        geom_id=heightmaps[leg_name].geom_ids[i, j],
                                    )

                # Plot the GRF
                for leg_id, leg_name in enumerate(legs_order):
                    feet_GRF_geom_ids[leg_name] = render_vector(
                        env.viewer,
                        vector=feet_GRF[leg_name],
                        pos=feet_pos[leg_name],
                        scale=np.linalg.norm(feet_GRF[leg_name]) * 0.005,
                        color=np.array([0, 1, 0, 0.5]),
                        geom_id=feet_GRF_geom_ids[leg_name],
                    )

                if goal_base_pos is not None:
                    # Draw the target point so goal-driven motion is visible in the viewer.
                    goal_geom_id = render_sphere(
                        viewer=env.viewer,
                        position=goal_position_world,
                        diameter=0.15,
                        color=[1, 0, 0, 0.75],
                        geom_id=goal_geom_id,
                    )

                env.render()
                last_render_time = time.time()

            # ---- Stop or reset conditions -------------------------------------------------------
            reached_goal = goal_base_pos is not None and distance_to_goal <= goal_position_tolerance
            if stop_at_goal and reached_goal:
                print(f"Goal reached at {base_pos[:2]}")
                state_obs_history.append(ep_state_history)
                ctrl_state_history.append(ep_ctrl_state_history)
                goal_reached = True
                break

            if env.step_num >= N_STEPS_PER_EPISODE or is_terminated or is_truncated:
                if is_terminated:
                    print("Environment terminated")
                    state_obs_history.append(ep_state_history)
                    ctrl_state_history.append(ep_ctrl_state_history)
                    terminated_early = True
                    break
                else:
                    state_obs_history.append(ep_state_history)
                    ctrl_state_history.append(ep_ctrl_state_history)

                env.reset(random=True)
                quadrupedpympc_wrapper.reset(initial_feet_pos=env.feet_pos(frame="world"))

        # Flush the episode trajectory once the inner rollout loop finishes.
        if h5py_writer is not None:
            ep_obs_history = collate_obs(ep_state_history)  # | collate_obs(ep_ctrl_state_history)
            ep_traj_time = np.asarray(ep_time)[:, np.newaxis]
            h5py_writer.append_trajectory(state_obs_traj=ep_obs_history, time=ep_traj_time)

        if goal_reached or terminated_early:
            break

    env.close()
    if h5py_writer is not None:
        return h5py_writer.file_path

def collate_obs(list_of_dicts) -> dict[str, np.ndarray]:
    """Collates a list of dictionaries containing observation names and numpy arrays
    into a single dictionary of stacked numpy arrays.
    """
    if not list_of_dicts:
        raise ValueError("Input list is empty.")

    # Get all keys (assumes all dicts have the same keys)
    keys = list_of_dicts[0].keys()

    # Stack the values per key
    collated = {key: np.stack([d[key] for d in list_of_dicts], axis=0) for key in keys}
    collated = {key: v[:, None] if v.ndim == 1 else v for key, v in collated.items()}
    return collated


if __name__ == "__main__":
    from quadruped_pympc import config as cfg

    parser = argparse.ArgumentParser(description="Run the quadruped Mujoco simulation.")
    parser.add_argument("--goal-x", type=float, default=1.0, help="Goal x position in world frame.")
    parser.add_argument("--goal-y", type=float, default=0.0, help="Goal y position in world frame.")
    args = parser.parse_args()

    qpympc_cfg = cfg
    # print_controller_config(qpympc_cfg)
    goal_base_pos = np.array([args.goal_x, args.goal_y, qpympc_cfg.simulation_params["ref_z"]])

    run_simulation(
        qpympc_cfg=qpympc_cfg,
        num_episodes=1,
        base_vel_command_type="forward",
        goal_base_pos=goal_base_pos,
        stop_at_goal=True,
    )

    # run_simulation(num_episodes=1, render=False)
