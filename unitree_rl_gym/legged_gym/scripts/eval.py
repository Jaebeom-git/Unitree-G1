# legged_gym/scripts/eval.py

import os
import time
import csv
import copy
import argparse
import subprocess
import sys
import numpy as np

import isaacgym  # noqa: F401 (needed to load Isaac Gym)
from isaacgym import gymapi
import torch
from isaacgym.torch_utils import get_euler_xyz, quat_apply

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


# ---- Camera follow config ----
CAMERA_FOLLOW = True          # False로 바꾸면 추적 끔
CAMERA_OFFSET = (-2.5, -2.0, 1.0)  # 로봇 기준 (x,y,z) 오프셋


def update_camera_follow(env, base_pos_tensor, offset=CAMERA_OFFSET):
    """
    env의 viewer 카메라를 base_pos 주변으로 옮겨서 로봇을 추적.
    base_pos_tensor: shape (3,) tensor (x,y,z)
    """
    if not CAMERA_FOLLOW:
        return
    if not hasattr(env, "viewer") or env.viewer is None:
        return
    if not hasattr(env, "gym") or not hasattr(env, "envs") or len(env.envs) == 0:
        return

    x = float(base_pos_tensor[0].item())
    y = float(base_pos_tensor[1].item())
    z = float(base_pos_tensor[2].item())

    target = gymapi.Vec3(x, y, z)
    pos = gymapi.Vec3(x + offset[0], y + offset[1], z + offset[2])

    env.gym.viewer_camera_look_at(env.viewer, env.envs[0], pos, target)


# ---- Recording ----

RECORD_ENABLED = True
CAM_W, CAM_H = 800, 600
RECORD_FPS = 30

class FFMPEGVideoRecorder:
    def __init__(self, path, width, height, fps=30):
        self.path = path
        self.w = width
        self.h = height
        self.fps = fps
        self.proc = None

    def start(self):
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.w}x{self.h}",
            "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            self.path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write_frame(self, frame_rgb_uint8):
        # frame_rgb_uint8: shape (H, W, 3), dtype=uint8
        if self.proc is None or self.proc.stdin is None:
            return
        self.proc.stdin.write(frame_rgb_uint8.tobytes())

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
            self.proc = None

# -------------------------
#   Time-series CSV logger
# -------------------------

class TimeSeriesLogger:
    """Collect per-step robot/RL data and save as CSV."""

    def __init__(self, env, save_dir, filename="g1_walk_timeseries.csv"):
        self.env = env
        self.save_dir = save_dir
        self.filename = filename
        self.header = None
        self.rows = []

        self.joint_names = getattr(env, "dof_names", None)
        self.body_names  = getattr(env, "body_names", None)

    def _build_header(self, state, actions, commands):
        """Build CSV header from first sample."""
        header = [
            "step",
            "time",
            "reward",
            "done",
            "distance",
        ]

        # base pose / twist (env 0)
        for axis in ["x", "y", "z"]:
            header.append(f"base_pos_{axis}")
        for name in ["roll", "pitch", "yaw"]:
            header.append(f"base_rpy_{name}")
        for axis in ["x", "y", "z"]:
            header.append(f"base_lin_vel_{axis}")
        for axis in ["x", "y", "z"]:
            header.append(f"base_ang_vel_{axis}")

        # joint states
        if state.get("dof_pos", None) is not None:
            n_dof = state["dof_pos"].shape[1]
            use_names = (
                self.joint_names is not None
                and len(self.joint_names) == n_dof
            )
            for i in range(n_dof):
                jname = self.joint_names[i] if use_names else str(i)
                header.append(f"dof_pos_{jname}")

        if state.get("dof_vel", None) is not None:
            n_dof = state["dof_vel"].shape[1]
            use_names = (
                self.joint_names is not None
                and len(self.joint_names) == n_dof
            )
            for i in range(n_dof):
                jname = self.joint_names[i] if use_names else str(i)
                header.append(f"dof_vel_{jname}")
        # if state.get("dof_pos", None) is not None:
        #     n_dof = state["dof_pos"].shape[1]
        #     for i in range(n_dof):
        #         header.append(f"dof_pos_{i}")
        # if state.get("dof_vel", None) is not None:
        #     n_dof = state["dof_vel"].shape[1]
        #     for i in range(n_dof):
        #         header.append(f"dof_vel_{i}")

        # target joint positions (for P control)
        if state.get("target_q", None) is not None and state["target_q"] is not None:
            n_dof = state["target_q"].shape[1]
            use_names = (
                self.joint_names is not None
                and len(self.joint_names) == n_dof
            )
            for i in range(n_dof):
                jname = self.joint_names[i] if use_names else str(i)
                header.append(f"target_q_{jname}")

        # joint torques
        if state.get("torques", None) is not None and state["torques"] is not None:
            n_dof = state["torques"].shape[1]
            use_names = (
                self.joint_names is not None
                and len(self.joint_names) == n_dof
            )
            for i in range(n_dof):
                jname = self.joint_names[i] if use_names else str(i)
                header.append(f"torque_{jname}")

        # RL actions (actor outputs)
        if actions is not None:
            n_act = actions.shape[1]
            use_names = (
                self.joint_names is not None
                and len(self.joint_names) == n_act
            )
            for i in range(n_act):
                aname = self.joint_names[i] if use_names else str(i)
                header.append(f"action_{aname}")
            # n_act = actions.shape[1]
            # for i in range(n_act):
            #     header.append(f"action_{i}")

        # high-level commands (vx, vy, yaw_rate/heading, ...)
        if commands is not None:
            n_cmd = commands.shape[1]
            for i in range(n_cmd):
                header.append(f"command_{i}")

        # rigid body states (flattened)
        if state.get("body_pos", None) is not None:
            n_bodies = state["body_pos"].shape[0]
            use_names = (
                self.body_names is not None
                and len(self.body_names) == n_bodies
            )
            axes = ["x", "y", "z"]
            for b in range(n_bodies):
                bname = self.body_names[b] if use_names else str(b)
                for ax in axes:
                    header.append(f"body_{bname}_pos_{ax}")
        if state.get("body_quat", None) is not None:
            n_bodies = state["body_quat"].shape[0]
            use_names = (
                self.body_names is not None
                and len(self.body_names) == n_bodies
            )
            comps = ["w", "x", "y", "z"]
            for b in range(n_bodies):
                bname = self.body_names[b] if use_names else str(b)
                for c in comps:
                    header.append(f"body_{bname}_quat_{c}")
        if state.get("body_lin_vel", None) is not None:
            n_bodies = state["body_lin_vel"].shape[0]
            use_names = (
                self.body_names is not None
                and len(self.body_names) == n_bodies
            )
            axes = ["x", "y", "z"]
            for b in range(n_bodies):
                bname = self.body_names[b] if use_names else str(b)
                for ax in axes:
                    header.append(f"body_{bname}_lin_vel_{ax}")
        if state.get("body_ang_vel", None) is not None:
            n_bodies = state["body_ang_vel"].shape[0]
            use_names = (
                self.body_names is not None
                and len(self.body_names) == n_bodies
            )
            axes = ["x", "y", "z"]
            for b in range(n_bodies):
                bname = self.body_names[b] if use_names else str(b)
                for ax in axes:
                    header.append(f"body_{bname}_ang_vel_{ax}")

        # # contact forces (flattened)
        if state.get("contact_forces", None) is not None:
            n_bodies = state["contact_forces"].shape[1]
            use_names = (
                self.body_names is not None
                and len(self.body_names) == n_bodies
            )
            axes = ["x", "y", "z"]
            for b in range(n_bodies):
                bname = self.body_names[b] if use_names else str(b)
                for ax in axes:
                    header.append(f"contact_{bname}_{ax}")
        # if state.get("contact_forces", None) is not None:
        #     n = state["contact_forces"][0].numel()
        #     for i in range(n):
        #         header.append(f"contact_flat_{i}")

        self.header = header

    def log(self, step, t, state, actions, commands, reward, done, distance):
        """Append one row (env 0) to the buffer."""
        if self.header is None:
            self._build_header(state, actions, commands)

        row = [int(step), float(t), float(reward), int(done), float(distance)]

        # base pose / twist
        base_pos = state["base_pos"][0].detach().cpu().numpy()
        base_rpy = state["base_rpy"][0].detach().cpu().numpy()
        base_lin_vel = state["base_lin_vel"][0].detach().cpu().numpy()
        base_ang_vel = state["base_ang_vel"][0].detach().cpu().numpy()

        row.extend(base_pos.tolist())
        row.extend(base_rpy.tolist())
        row.extend(base_lin_vel.tolist())
        row.extend(base_ang_vel.tolist())

        # joint states
        if state.get("dof_pos", None) is not None:
            q = state["dof_pos"][0].detach().cpu().numpy()
            row.extend(q.tolist())
        if state.get("dof_vel", None) is not None:
            dq = state["dof_vel"][0].detach().cpu().numpy()
            row.extend(dq.tolist())

        # target_q
        if state.get("target_q", None) is not None and state["target_q"] is not None:
            tq = state["target_q"][0].detach().cpu().numpy()
            row.extend(tq.tolist())

        # torques
        if state.get("torques", None) is not None and state["torques"] is not None:
            tau = state["torques"][0].detach().cpu().numpy()
            row.extend(tau.tolist())

        # actions (RL output)
        if actions is not None:
            a = actions[0].detach().cpu().numpy()
            row.extend(a.tolist())

        # commands
        if commands is not None:
            c = commands[0].detach().cpu().numpy()
            row.extend(c.tolist())

        # # rigid body states
        # if state.get("body_pos", None) is not None:
        #     v = state["body_pos"][0].detach().cpu().numpy().reshape(-1)
        #     row.extend(v.tolist())
        # if state.get("body_quat", None) is not None:
        #     v = state["body_quat"][0].detach().cpu().numpy().reshape(-1)
        #     row.extend(v.tolist())
        # if state.get("body_lin_vel", None) is not None:
        #     v = state["body_lin_vel"][0].detach().cpu().numpy().reshape(-1)
        #     row.extend(v.tolist())
        # if state.get("body_ang_vel", None) is not None:
        #     v = state["body_ang_vel"][0].detach().cpu().numpy().reshape(-1)
        #     row.extend(v.tolist())

        # # contact forces
        # if state.get("contact_forces", None) is not None:
        #     v = state["contact_forces"][0].detach().cpu().numpy().reshape(-1)
        #     row.extend(v.tolist())

        self.rows.append(row)

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, self.filename)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
            writer.writerows(self.rows)
        print(f"[Eval] Saved time-series CSV to: {path}")


# -------------------------
#   Eval base utilities
# -------------------------

class EvalBase:
    """Common eval utils: disable command resampling, read robot state."""

    def __init__(self, env, args=None):
        self.env = env
        self.args = args
        self.device = env.device
        self._disable_command_resampling()
        self.start_base_pos = None

    def _disable_command_resampling(self):
        """Make _resample_commands a no-op for deterministic eval."""
        def _no_resample_commands(this, env_ids):
            return
        self.env._resample_commands = _no_resample_commands.__get__(self.env, type(self.env))

    def _init_start_pos(self):
        """Store initial x position for distance measurement."""
        root_states = self.env.root_states
        self.start_base_pos = root_states[:, 0:3].clone()

    @torch.no_grad()
    def get_robot_state(self):
        """Return current robot state (tensors)."""
        root_states = self.env.root_states  # (N, 13)
        base_pos = root_states[:, 0:3]
        base_quat = root_states[:, 3:7]
        base_lin_vel = root_states[:, 7:10]
        base_ang_vel = root_states[:, 10:13]

        roll, pitch, yaw = get_euler_xyz(base_quat)
        base_rpy = torch.stack([roll, pitch, yaw], dim=-1)

        state = {
            "base_pos": base_pos.clone(),
            "base_quat": base_quat.clone(),
            "base_rpy": base_rpy.clone(),
            "base_lin_vel": base_lin_vel.clone(),
            "base_ang_vel": base_ang_vel.clone(),
        }

        # joint states
        if hasattr(self.env, "dof_pos") and self.env.dof_pos is not None:
            state["dof_pos"] = self.env.dof_pos.clone()
        else:
            state["dof_pos"] = None

        if hasattr(self.env, "dof_vel") and self.env.dof_vel is not None:
            state["dof_vel"] = self.env.dof_vel.clone()
        else:
            state["dof_vel"] = None

        if hasattr(self.env, "torques") and self.env.torques is not None:
            state["torques"] = self.env.torques.clone()
        else:
            state["torques"] = None

        # RL actions
        if hasattr(self.env, "actions") and self.env.actions is not None:
            state["actions"] = self.env.actions.clone()
        else:
            state["actions"] = None

        target_q = None
        try:
            if (
                hasattr(self.env, "default_dof_pos")
                and hasattr(self.env, "actions") and self.env.actions is not None
                and hasattr(self.env, "cfg")
                and hasattr(self.env.cfg, "control")
                and getattr(self.env.cfg.control, "control_type", "") == "P"
            ):
                action_scale = float(self.env.cfg.control.action_scale)
                target_q = self.env.default_dof_pos + action_scale * self.env.actions
        except Exception:
            target_q = None

        state["target_q"] = target_q.clone() if target_q is not None else None

        # # rigid body states
        # if hasattr(self.env, "rigid_body_states") and self.env.rigid_body_states is not None:
        #     rb = self.env.rigid_body_states  # (N, B, 13)
        #     state["body_pos"] = rb[..., 0:3].clone()
        #     state["body_quat"] = rb[..., 3:7].clone()
        #     state["body_lin_vel"] = rb[..., 7:10].clone()
        #     state["body_ang_vel"] = rb[..., 10:13].clone()
        # else:
        #     state["body_pos"] = None
        #     state["body_quat"] = None
        #     state["body_lin_vel"] = None
        #     state["body_ang_vel"] = None

        # # contact forces
        # if hasattr(self.env, "contact_forces") and self.env.contact_forces is not None:
        #     state["contact_forces"] = self.env.contact_forces.clone()
        # else:
        #     state["contact_forces"] = None

        return state

    @torch.no_grad()
    def step(self):
        raise NotImplementedError


# -------------------------
#   10 m straight walk
# -------------------------
class Walk(EvalBase):
    """
    10 m straight walking.
    commands:
      0: forward velocity (vx)
      1: lateral velocity (vy)
      2: yaw rate
      3: heading (unused at eval, kept at 0)
    Optional line-tracking feedback (y, heading) with P or PD.
    """

    def __init__(
        self,
        env,
        target_distance: float = 10.0,
        walk_speed: float = 1.0,
        feedback_mode: str = "",  # "P" or "PD"
        args=None,
    ):
        super().__init__(env, args=args)

        self.target_distance = float(target_distance)
        self.walk_speed = float(walk_speed)

        self.feedback_mode = feedback_mode.upper()
        self.use_feedback = self.feedback_mode in ("P", "PD")

        self.num_envs, self.num_commands = env.commands.shape
        assert self.num_commands >= 4, "G1 eval assumes 4D commands: [vx, vy, yaw_rate, heading]"

        # line tracking gains (world-frame y & heading)
        self.kp_y = 2.0
        self.kd_y = 1.0
        self.kp_yaw = 10
        self.kd_yaw = 1.0

        # saturation
        self.max_vy = 0.5
        self.max_yaw_rate = 0.8

        # target heading in world frame (+x)
        self.target_heading = 0.0

    @torch.no_grad()
    def _apply_feedback(self, state, cmds, mask):
        """Apply line-tracking feedback (y, heading) to commands."""
        if not self.use_feedback:
            return

        base_pos = state["base_pos"]         # (N,3), world
        base_quat = state["base_quat"]       # (N,4)
        base_lin_vel = state["base_lin_vel"] # (N,3), world
        base_ang_vel = state["base_ang_vel"] # (N,3), world

        # heading from forward vector in world frame
        forward = quat_apply(base_quat, self.env.forward_vec)  # (N,3)
        heading = torch.atan2(forward[:, 1], forward[:, 0])    # rad

        # tracking signals (y = 0 line, heading = +x)
        y = base_pos[:, 1]
        vy = base_lin_vel[:, 1]
        yaw_rate = base_ang_vel[:, 2]

        heading_err = heading - self.target_heading

        if self.feedback_mode == "P":
            lateral_cmd = -self.kp_y * y
            yaw_cmd = -self.kp_yaw * heading_err
        else:  # "PD"
            lateral_cmd = -self.kp_y * y - self.kd_y * vy
            yaw_cmd = -self.kp_yaw * heading_err - self.kd_yaw * yaw_rate

        lateral_cmd = lateral_cmd.clamp(-self.max_vy, self.max_vy)
        yaw_cmd = yaw_cmd.clamp(-self.max_yaw_rate, self.max_yaw_rate)

        # fixed command layout: [vx, vy, yaw_rate, heading]
        cmds[mask, 1] = lateral_cmd[mask]
        cmds[mask, 2] = yaw_cmd[mask]
        cmds[mask, 3] = 0.0  # keep desired heading = 0 (along +x)

    @torch.no_grad()
    def step(self):
        """Set env.commands for straight walking. Return per-env x-distance from start."""
        if self.start_base_pos is None:
            self._init_start_pos()

        state = self.get_robot_state()
        base_pos = state["base_pos"]

        current_x = base_pos[:, 0]
        dist = current_x - self.start_base_pos[:, 0]
        walk_mask = dist < self.target_distance
        done_by_distance = ~walk_mask

        cmds = self.env.commands
        cmds[:, :] = 0.0

        # forward command along +x
        cmds[walk_mask, 0] = self.walk_speed

        if self.use_feedback:
            self._apply_feedback(state, cmds, walk_mask)
        else:
            cmds[walk_mask, 1] = 0.0  # vy
            cmds[walk_mask, 2] = 0.0  # yaw rate
            cmds[walk_mask, 3] = 0.0  # heading

        self.env.commands[:] = cmds
        return dist, state, done_by_distance


# -------------------------
#   G1 walk eval entrypoint
# -------------------------

def eval_single(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    if args.task != "g1":
        print(f"[Warning] eval_g1_walk is written for task='g1' (current: '{args.task}').")

    # clean eval setup
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True

    # disable heading command for eval walk (heading off)
    env_cfg.commands.heading_command = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    dt = env_cfg.env.episode_length_s / env.max_episode_length
    wait_steps = int(3.0 / dt)
    print(f"[Eval] dt = {dt:.4f} s, 3 sec ≈ {wait_steps} steps")

    # walker: use PD line tracking by default
    walker = Walk(
        env,
        target_distance=4.0,
        walk_speed=1.0,
        feedback_mode="P",
        args=args,
    )

    # CSV logger (only during walking phase)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    eval_log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        "eval",
        train_cfg.runner.experiment_name,
    )
    if hasattr(args, "load_run") and args.load_run is not None:
        train_cfg.runner.run_name = args.load_run

    run_dir = os.path.join(eval_log_root, train_cfg.runner.run_name)
    os.makedirs(run_dir, exist_ok=True)
    # CSV logger ...
    csv_name = f"walk_{train_cfg.runner.run_name}.csv"
    ts_logger = TimeSeriesLogger(env, save_dir=run_dir, filename=csv_name)

    # ---- Recording camera & video recorder ----
    if RECORD_ENABLED:
        accumulated_time = 0.0
        frame_interval = 1.0 / RECORD_FPS

        mp4_path = os.path.join(run_dir, "record.mp4")

        camera_props = gymapi.CameraProperties()
        camera_props.width = CAM_W
        camera_props.height = CAM_H
        camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)

        video_recorder = FFMPEGVideoRecorder(mp4_path, CAM_W, CAM_H, fps=RECORD_FPS)
        video_recorder.start()
        
        def move_record_camera(env, base_pos, offset=(-2.0, -2.0, 1.0)):
            x = float(base_pos[0].item())
            y = float(base_pos[1].item())
            z = float(base_pos[2].item())
            target = gymapi.Vec3(x, y, z)
            pos = gymapi.Vec3(x + offset[0], y + offset[1], z + offset[2])
            env.gym.set_camera_location(camera_handle, env.envs[0], pos, target)
    else:
        video_recorder = None
        camera_handle = None

    # warm-up: stand still
    print("[Eval] Warmup ~3 seconds (stand still)")
    for _ in range(wait_steps):
        env.commands[:] = 0.0
        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        if CAMERA_FOLLOW:
            base_pos0 = env.root_states[0, 0:3]
            update_camera_follow(env, base_pos0)

    # walking phase
    max_steps = 1000 * int(env.max_episode_length)
    reached = False

    for step_idx in range(max_steps):
        dists, state, done_by_distance = walker.step()
        mean_dist = dists.mean().item()

        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        if CAMERA_FOLLOW:
            base_pos0 = state["base_pos"][0]
            update_camera_follow(env, base_pos0)

        reward = rews[0].item()
        done = bool(dones[0].item())
        distance0 = dists[0].item()
        t_now = step_idx * dt

        # log time-series (only env 0)
        ts_logger.log(
            step=step_idx,
            t=t_now,
            state=state,
            actions=actions,
            commands=env.commands,
            reward=reward,
            done=done if not done_by_distance[0].item() else done_by_distance[0].item(),
            distance=distance0,
        )
        
        if RECORD_ENABLED:
            t_now = step_idx * dt
            accumulated_time += dt
            
            if accumulated_time >= frame_interval:
                accumulated_time -= frame_interval

                move_record_camera(env, state["base_pos"][0])

                env.gym.render_all_camera_sensors(env.sim)
                img = env.gym.get_camera_image(
                    env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR
                )
                frame = np.frombuffer(img, dtype=np.uint8).reshape(CAM_H, CAM_W, 4)
                frame_rgb = frame[:, :, :3]

                video_recorder.write_frame(frame_rgb)

        if done_by_distance.all():
            print(
                f"[Eval] Reached {mean_dist:.2f} m "
                f"(target {walker.target_distance} m) at step {step_idx}."
            )
            reached = True
            break

        if torch.any(dones):
            print(
                f"[Eval] Episode terminated early at step {step_idx} "
                f"before reaching {walker.target_distance} m."
            )
            break

    if not reached:
        print(
            f"[Eval] Finished without reaching {walker.target_distance} m "
            f"(last distance {mean_dist:.2f} m)."
        )

    # cool-down: stand still
    print("[Eval] Cooldown ~3 seconds (stand still)")
    for _ in range(wait_steps):
        env.commands[:] = 0.0
        with torch.no_grad():
            actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        if CAMERA_FOLLOW:
            base_pos0 = env.root_states[0, 0:3]
            update_camera_follow(env, base_pos0)

    # save
    ts_logger.save()
    if RECORD_ENABLED and video_recorder is not None:
        video_recorder.close()
        print(f"[Record] Video saved to: {mp4_path}")

def eval_dir(args, runs_dir):
    runs_dir = os.path.abspath(runs_dir)
    if not os.path.isdir(runs_dir):
        raise NotADirectoryError(f"dir is not a directory: {runs_dir}")

    # 서브폴더 나열 (예: Nov27_..., exported 등)
    all_subdirs = sorted(
        d for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
    )

    skip_names = {"exported"}
    run_names = [d for d in all_subdirs if d not in skip_names]

    if not run_names:
        print(f"[Eval-All] No run folders found in: {runs_dir}")
        return

    print(f"[Eval-All] Found {len(run_names)} runs in {runs_dir}:")
    for rn in run_names:
        print(f"  - {rn}")

    # 이 스크립트 경로
    script_path = os.path.abspath(__file__)

    # 공통 인자 (task, device, headless 등 필요한 것만 전달)
    base_cmd = [sys.executable, script_path, "--task", args.task]
    if hasattr(args, "device") and args.device is not None:
        base_cmd += ["--device", args.device]
    if hasattr(args, "headless") and args.headless:
        base_cmd += ["--headless"]

    for rn in run_names:
        print(f"\n[Eval-All] ===== Evaluating run: {rn} =====")

        cmd = base_cmd + ["--load_run", rn]
        print("[Eval-All] Running:", " ".join(cmd))

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[Eval-All] Run {rn} failed with code {result.returncode}, stop.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="주어진 폴더 안의 모든 run 폴더를 순회하면서 eval 수행. 예: --dir logs/g1",
    )
    extra_args, remaining_argv = parser.parse_known_args()
    runs_dir = extra_args.dir

    # 2) sys.argv를 --dir 제거된 상태로 덮어쓰고 get_args() 호출
    import sys
    sys.argv = [sys.argv[0]] + remaining_argv
    args = get_args()

    # 3) args에 dir 정보 붙여서 사용
    args.dir = runs_dir

    if args.dir:
        # 폴더 모드: 폴더 내 모든 run을 평가
        eval_dir(args, args.dir)
    else:
        # 기존 모드: 단일 run만 평가
        eval_single(args)
