"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 \
        --camera_names agentview robot0_eye_in_hand 

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs 

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""
import argparse
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation

import torch
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy


def set_controller_delta(node):
    changed = False
    if isinstance(node, dict):
        if node.get("type", None) in ("OSC_POSE", "OSC_POSITION"):
            node["input_type"] = "delta"
            changed = True
        for key, value in node.items():
            if key == "control_delta":
                node[key] = True
                changed = True
            else:
                changed = set_controller_delta(value) or changed
    elif isinstance(node, list):
        for value in node:
            changed = set_controller_delta(value) or changed
    return changed


def arm_controllers(env):
    controllers = []
    for robot in env.unwrapped.env.robots:
        if hasattr(robot, "controller"):
            controller = robot.controller
            if hasattr(controller, "goal_pos") and hasattr(controller, "goal_ori"):
                controllers.append(controller)
            elif hasattr(controller, "part_controllers"):
                controllers.extend(
                    part_controller
                    for part_controller in controller.part_controllers.values()
                    if hasattr(part_controller, "goal_pos") and hasattr(part_controller, "goal_ori")
                )
            else:
                raise AssertionError(type(controller))
        elif hasattr(robot, "part_controllers"):
            controllers.extend(
                part_controller
                for part_controller in robot.part_controllers.values()
                if hasattr(part_controller, "goal_pos") and hasattr(part_controller, "goal_ori")
            )
        else:
            raise AssertionError(type(robot))
    assert len(controllers) in (1, 2), len(controllers)
    return controllers


def controller_goal_pos(controller):
    if controller.goal_pos is not None:
        return controller.goal_pos
    if controller.input_ref_frame == "base":
        return controller.world_to_origin_frame(controller.ref_pos)
    if controller.input_ref_frame == "world":
        return controller.ref_pos
    raise AssertionError(controller.input_ref_frame)


def controller_goal_ori(controller):
    if controller.goal_ori is not None:
        return controller.goal_ori
    if controller.input_ref_frame == "base":
        return controller.goal_origin_to_eef_pose()[:3, :3]
    if controller.input_ref_frame == "world":
        return controller.ref_ori_mat
    raise AssertionError(controller.input_ref_frame)


def controller_achieved_pos(controller):
    if controller.input_ref_frame == "base":
        return controller.world_to_origin_frame(controller.ref_pos)
    if controller.input_ref_frame == "world":
        return controller.ref_pos
    raise AssertionError(controller.input_ref_frame)


def controller_achieved_ori(controller):
    if controller.input_ref_frame == "base":
        return controller.goal_origin_to_eef_pose()[:3, :3]
    if controller.input_ref_frame == "world":
        return controller.ref_ori_mat
    raise AssertionError(controller.input_ref_frame)


def inverse_scale_controller_action(controller, scaled_action):
    if controller.action_scale is None:
        controller.scale_action(np.zeros_like(scaled_action))
    action = (
        (scaled_action - controller.action_output_transform)
        / controller.action_scale
        + controller.action_input_transform
    )
    return np.clip(action, controller.input_min, controller.input_max)


def abs_action_to_delta_action(env, abs_action):
    abs_action = np.asarray(abs_action, dtype=np.float64)
    controllers = arm_controllers(env)
    stacked_abs = abs_action.reshape(len(controllers), 7)
    stacked_delta = np.zeros_like(stacked_abs)
    for i, controller in enumerate(controllers):
        controller.update()
        target_pos = stacked_abs[i, :3]
        target_ori = Rotation.from_rotvec(stacked_abs[i, 3:6]).as_matrix()
        if controller._goal_update_mode == "desired":
            base_pos = controller_goal_pos(controller)
            base_ori = controller_goal_ori(controller)
        elif controller._goal_update_mode == "achieved":
            base_pos = controller_achieved_pos(controller)
            base_ori = controller_achieved_ori(controller)
        else:
            raise AssertionError(controller._goal_update_mode)
        scaled_pos = target_pos - base_pos
        scaled_ori = (Rotation.from_matrix(target_ori) * Rotation.from_matrix(base_ori).inv()).as_rotvec()
        stacked_delta[i, :6] = inverse_scale_controller_action(
            controller,
            np.concatenate([scaled_pos, scaled_ori]),
        )
        stacked_delta[i, 6] = stacked_abs[i, 6]
    return stacked_delta.reshape(abs_action.shape).astype(np.float32)


def write_rollout_episode(ep_data_grp, traj, include_obs=False):
    """
    Write a rollout episode in the standard robomimic rollout dataset format.
    """
    ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
    if "actions_abs" in traj:
        ep_data_grp.create_dataset("actions_abs", data=np.array(traj["actions_abs"]))
    ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
    ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
    ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
    if include_obs:
        for k in traj["obs"]:
            ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
            ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))


def write_demo_episode(ep_data_grp, traj):
    """
    Write a policy rollout in the raw PH demo.hdf5 schema.
    """
    actions = np.array(traj["actions"])
    actions_abs = np.array(traj["actions_abs"]) if "actions_abs" in traj else None
    states = np.array(traj["states"])
    ep_len = actions.shape[0]
    counters = np.arange(ep_len, dtype=np.int64)
    identity_rotations = np.tile(np.eye(3, dtype=np.float64), (ep_len, 1, 1))

    ep_data_grp.create_dataset("actions", data=actions)
    if actions_abs is not None:
        ep_data_grp.create_dataset("actions_abs", data=actions_abs)
    ep_data_grp.create_dataset("states", data=states)
    ep_data_grp.create_dataset("interventions", data=np.zeros((ep_len, 1), dtype=bool))
    ep_data_grp.create_dataset("policy_acting", data=np.ones((ep_len,), dtype=bool))
    ep_data_grp.create_dataset("user_acting", data=np.zeros((ep_len, 1), dtype=bool))

    controller_grp = ep_data_grp.create_group("controller_info")
    controller_grp.create_dataset("counters", data=counters)
    controller_grp.create_dataset("enabled", data=np.zeros((ep_len, 1), dtype=bool))
    controller_grp.create_dataset("user_timestamps", data=np.zeros((ep_len, 1), dtype=np.float64))

    robot_controls_grp = controller_grp.create_group("robot_controls")
    robot_controls_grp.create_dataset("action", data=actions)
    robot_controls_grp.create_dataset("counters", data=counters)

    teleop_commands_grp = controller_grp.create_group("teleop_commands")
    teleop_commands_arm_grp = teleop_commands_grp.create_group("arm0")
    teleop_commands_arm_grp.create_dataset("gripper", data=np.zeros((ep_len, 1), dtype=np.float64))
    teleop_commands_arm_grp.create_dataset("position", data=np.zeros((ep_len, 3), dtype=np.float64))
    teleop_commands_arm_grp.create_dataset("rotation", data=identity_rotations)
    teleop_commands_grp.create_dataset("arm0_counters", data=counters)

    teleop_controls_grp = controller_grp.create_group("teleop_controls")
    teleop_controls_arm_grp = teleop_controls_grp.create_group("arm0")
    teleop_controls_arm_grp.create_dataset("gripper", data=np.zeros((ep_len, 1), dtype=np.float64))
    teleop_controls_arm_grp.create_dataset("position", data=np.zeros((ep_len, 3), dtype=np.float64))
    teleop_controls_arm_grp.create_dataset("rotation", data=np.zeros((ep_len, 3), dtype=np.float32))
    teleop_controls_grp.create_dataset("arm0_counters", data=counters)

    user_info_grp = ep_data_grp.create_group("user_info").create_group("0")
    user_info_grp.create_dataset("clutch", data=np.zeros((ep_len,), dtype=bool))
    user_info_grp.create_dataset("dpos", data=np.zeros((ep_len, 3), dtype=np.float64))
    user_info_grp.create_dataset("engaged", data=np.zeros((ep_len,), dtype=bool))
    user_info_grp.create_dataset("grasp", data=np.zeros((ep_len,), dtype=bool))
    user_info_grp.create_dataset("intervene", data=np.zeros((ep_len,), dtype=bool))
    user_info_grp.create_dataset("reset_mode", data=np.zeros((ep_len,), dtype=bool))
    user_info_grp.create_dataset("rotation", data=identity_rotations)
    user_info_grp.create_dataset("sensitivity", data=np.zeros((ep_len,), dtype=np.int64))
    user_info_grp.create_dataset("taskState", data=np.zeros((ep_len,), dtype=np.int64))
    user_info_grp.create_dataset("timestamp", data=np.zeros((ep_len,), dtype=np.float64))
    user_info_grp.create_dataset("valid", data=np.zeros((ep_len,), dtype=bool))
    user_info_grp.create_dataset("zoom", data=np.zeros((ep_len,), dtype=np.float64))


def write_demo_masks(data_writer, num_demos):
    """
    Add the same mask keys used by the PH demo_v15.hdf5 file.
    """
    mask_grp = data_writer.create_group("mask")
    demo_keys = np.array(["demo_{}".format(i).encode("utf-8") for i in range(num_demos)])
    train_count = int(0.9 * num_demos)
    percent_20_count = int(0.2 * num_demos)
    percent_50_count = int(0.5 * num_demos)

    mask_grp.create_dataset("train", data=demo_keys[:train_count])
    mask_grp.create_dataset("valid", data=demo_keys[train_count:])
    mask_grp.create_dataset("20_percent", data=demo_keys[:percent_20_count])
    mask_grp.create_dataset("20_percent_train", data=demo_keys[: int(0.9 * percent_20_count)])
    mask_grp.create_dataset("20_percent_valid", data=demo_keys[int(0.9 * percent_20_count) : percent_20_count])
    mask_grp.create_dataset("50_percent", data=demo_keys[:percent_50_count])
    mask_grp.create_dataset("50_percent_train", data=demo_keys[: int(0.9 * percent_50_count)])
    mask_grp.create_dataset("50_percent_valid", data=demo_keys[int(0.9 * percent_50_count) : percent_50_count])


def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu. 
            They are excluded by default because the low-dimensional simulation states should be a minimal 
            representation of the environment. 
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):

            # get action from policy
            act = policy(ob=obs)

            # play action
            next_obs, r, done, _ = env.step(act)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                traj["obs"].append(obs)
                traj["next_obs"].append(next_obs)

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


def rollout_abs_policy_with_delta_env(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None):
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    results = {}
    video_count = 0
    total_reward = 0.
    traj = dict(actions=[], actions_abs=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):
            abs_act = policy(ob=obs)
            delta_act = abs_action_to_delta_action(env, abs_act)
            next_obs, r, done, _ = env.step(delta_act)
            total_reward += r
            success = env.is_success()["task"]

            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1)
                    video_writer.append_data(video_img)
                video_count += 1

            traj["actions"].append(delta_act)
            traj["actions_abs"].append(abs_act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                traj["obs"].append(obs)
                traj["next_obs"].append(next_obs)

            if done or success:
                break

            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if return_obs:
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


def run_trained_agent(args):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.camera_names) == 1

    # relative path to agent
    ckpt_path = args.agent

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
    if args.rollout_camera_view_mapping is not None:
        config_dict = json.loads(ckpt_dict["config"])
        camera_view_mapping = {}
        for mapping in args.rollout_camera_view_mapping:
            pair = mapping.split("=")
            assert len(pair) == 2, mapping
            assert pair[0] != "" and pair[1] != "", mapping
            camera_view_mapping[pair[0]] = pair[1]
        config_dict["experiment"]["rollout"]["camera_view_mapping"] = camera_view_mapping
        policy_camera_names = []
        for obs_key in config_dict["observation"]["modalities"]["obs"]["rgb"]:
            assert obs_key[-6:] == "_image", obs_key
            camera_name = obs_key[:-6]
            if camera_name not in policy_camera_names:
                policy_camera_names.append(camera_name)
        ckpt_dict["env_metadata"] = deepcopy(ckpt_dict["env_metadata"])
        ckpt_dict["env_metadata"]["env_kwargs"]["camera_names"] = policy_camera_names
        ckpt_dict["config"] = json.dumps(config_dict)
        print("Using rollout camera view mapping: {}".format(camera_view_mapping))
        print("Using policy camera names before mapping: {}".format(policy_camera_names))

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        # read horizon from config
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    if getattr(args, "abs_policy_delta_env", False):
        ckpt_dict = deepcopy(ckpt_dict)
        changed = set_controller_delta(ckpt_dict["env_metadata"]["env_kwargs"]["controller_configs"])
        assert changed, ckpt_dict["env_metadata"]["env_kwargs"]["controller_configs"]

    # create environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=args.env, 
        render=args.render, 
        render_offscreen=(args.video_path is not None), 
        verbose=True,
    )

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    dataset_format = getattr(args, "dataset_format", "rollout")
    if dataset_format == "demo" and args.dataset_obs:
        raise ValueError("--dataset_format demo is only supported without --dataset_obs")
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    target_successes = getattr(args, "n_successful_rollouts", None)
    max_attempts = getattr(args, "max_rollout_attempts", None)
    successful_rollouts_only = target_successes is not None
    if successful_rollouts_only:
        if target_successes <= 0:
            raise ValueError("--n_successful_rollouts must be positive")
        if args.dataset_path is None:
            raise ValueError("--n_successful_rollouts requires --dataset_path")
        rollout_num_episodes = target_successes
        if max_attempts is None:
            max_attempts = 10 * target_successes
        if max_attempts < target_successes:
            raise ValueError("--max_rollout_attempts must be at least --n_successful_rollouts")

    rollout_stats = []
    rollout_iter = range(max_attempts if successful_rollouts_only else rollout_num_episodes)
    if tqdm is not None:
        rollout_iter = tqdm(
            rollout_iter,
            total=max_attempts if successful_rollouts_only else rollout_num_episodes,
            desc="Evaluating rollouts",
            dynamic_ncols=True,
        )
    num_successes = 0
    for i in rollout_iter:
        rollout_fn = rollout_abs_policy_with_delta_env if getattr(args, "abs_policy_delta_env", False) else rollout
        stats, traj = rollout_fn(
            policy=policy, 
            env=env, 
            horizon=rollout_horizon, 
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip, 
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
        )
        rollout_stats.append(stats)
        success = stats["Success_Rate"] > 0.
        if successful_rollouts_only and not success:
            if tqdm is not None:
                rollout_iter.set_postfix(successes=num_successes, attempts=i + 1)
            continue

        if write_dataset:
            # store transitions
            ep_index = num_successes if successful_rollouts_only else i
            ep_data_grp = data_grp.create_group("demo_{}".format(ep_index))
            if dataset_format == "demo":
                write_demo_episode(ep_data_grp=ep_data_grp, traj=traj)
            else:
                write_rollout_episode(ep_data_grp=ep_data_grp, traj=traj, include_obs=args.dataset_obs)

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

        if success:
            num_successes += 1
        if successful_rollouts_only:
            if tqdm is not None:
                rollout_iter.set_postfix(successes=num_successes, attempts=i + 1)
            if num_successes >= target_successes:
                break

    if successful_rollouts_only and num_successes < target_successes:
        raise RuntimeError(
            "Only collected {} successful rollouts after {} attempts; target was {}".format(
                num_successes, len(rollout_stats), target_successes
            )
        )

    num_attempts = len(rollout_stats)
    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    avg_rollout_stats["Num_Attempts"] = num_attempts
    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        if dataset_format == "demo":
            num_written = num_successes if successful_rollouts_only else rollout_num_episodes
            write_demo_masks(data_writer=data_writer, num_demos=num_written)
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path to trained model
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=27,
        help="number of rollouts",
    )

    # number of successful rollouts to save
    parser.add_argument(
        "--n_successful_rollouts",
        type=int,
        default=None,
        help=(
            "if provided, keep rolling out until this many successful trajectories "
            "have been written to --dataset_path"
        ),
    )

    # max rollout attempts for successful-only collection
    parser.add_argument(
        "--max_rollout_attempts",
        type=int,
        default=None,
        help=(
            "maximum rollout attempts when using --n_successful_rollouts "
            "(defaults to 10x the requested successes)"
        ),
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Env Name (to override the one stored in model checkpoint)
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override name of env from the one in the checkpoint, and use\
            it for rollouts",
    )

    # Whether to render rollouts to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview"],
        help="(optional) camera name(s) to use for rendering on-screen or to video",
    )

    # camera view mapping for online rollout policy observations
    parser.add_argument(
        "--rollout_camera_view_mapping",
        type=str,
        nargs='+',
        default=None,
        help=(
            "optional policy_camera=render_camera mapping for online rollout observations "
            "(for example: thirdperson_1=agentview)"
        ),
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # Dataset file structure to write.
    parser.add_argument(
        "--dataset_format",
        type=str,
        default="rollout",
        choices=["rollout", "demo"],
        help=(
            "output HDF5 format. 'rollout' keeps rewards / dones, while 'demo' "
            "matches the raw PH demo.hdf5 schema"
        ),
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

    parser.add_argument(
        "--abs_policy_delta_env",
        action="store_true",
        help=(
            "interpret policy outputs as absolute actions, convert them to original "
            "delta-controller actions, and step the simulator with the deltas"
        ),
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) set seed for rollouts",
    )

    args = parser.parse_args()
    run_trained_agent(args)
