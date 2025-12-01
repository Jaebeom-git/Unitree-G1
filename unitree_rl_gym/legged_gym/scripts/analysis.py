# legged_gym/scripts/analysis.py

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- G1 config import for joint names & default angles ---
from legged_gym.envs.g1.g1_config import G1RoughCfg

# G1 default joint angles (rad) and action scale
G1_DEFAULT_ANGLES = G1RoughCfg.init_state.default_joint_angles
G1_ACTION_SCALE = G1RoughCfg.control.action_scale

# ---- Pretty labels for plots ----
_BASE_JOINT_LABELS = {
    "hip_yaw_joint": "Hip Yaw",
    "hip_roll_joint": "Hip Roll",
    "hip_pitch_joint": "Hip Pitch",
    "knee_joint": "Knee",
    "ankle_pitch_joint": "Ankle Pitch",
    "ankle_roll_joint": "Ankle Roll",
}

# body names: ['pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link']

# joint names: ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint']


_SIDE_LABELS = {
    "left": "Left",
    "right": "Right",
}

JOINT_GRID = [
    ["left_hip_roll_joint",   "left_hip_pitch_joint",    "left_hip_yaw_joint"],
    ["right_hip_roll_joint",  "right_hip_pitch_joint",   "right_hip_yaw_joint"],
    ["left_knee_joint",       "left_ankle_roll_joint",  "left_ankle_pitch_joint"],
    ["right_knee_joint",      "right_ankle_roll_joint", "right_ankle_pitch_joint"],
]

def get_joint_display_labels(jname: str):
    """
    Convert internal joint name (e.g. 'left_hip_yaw_joint')
    to:
      - title: 'Left Hip Yaw'
      - ylabel: 'Hip Yaw [deg]'
    """
    parts = jname.split("_", 1)
    if len(parts) == 2 and parts[0] in _SIDE_LABELS:
        side = parts[0]
        rest = parts[1]  # e.g. 'hip_yaw_joint'
        base_label = _BASE_JOINT_LABELS.get(rest, jname)
        side_label = _SIDE_LABELS[side]
        title = f"{side_label} {base_label}"
        ylabel = f"{base_label} [deg]"
    else:
        # fallback: use raw name
        base_label = jname
        title = jname
        ylabel = f"{base_label} [deg]"
    return title, ylabel


# --- Global plot style config ---
FIG_TITLE_FONTSIZE = 16
AX_TITLE_FONTSIZE = 11
AX_LABEL_FONTSIZE = 10
LEGEND_FONTSIZE = 10

# ------------------ CSV loader ------------------ #

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    print(f"[Analysis] Loaded CSV: {path} ({len(df)} rows, {len(df.columns)} columns)")
    return df


# ------------------ Joint vs action (4x3 layout) ------------------ #

def plot_joint_pos_action_group(df: pd.DataFrame, out_dir: str):
    """
    4x3 grid:

        L_hip_roll      L_hip_pitch      L_hip_yaw
        R_hip_roll      R_hip_pitch      R_hip_yaw
        L_knee         L_ankle_roll   L_ankle_pitch
        R_knee         R_ankle_roll   R_ankle_pitch

    Each subplot:
      - blue:  joint pos [deg]
      - red--: target pos = default + action_scale*action [deg]
    """    
    os.makedirs(out_dir, exist_ok=True)

    has_dof = any(c.startswith("dof_pos_") for c in df.columns)
    has_act = any(c.startswith("action_") for c in df.columns)
    if not (has_dof and has_act):
        print("[Analysis] No dof_pos_* or action_* columns found, skip joint/action plots.")
        return
    
    
    t = df["time"].values

    n_rows = len(JOINT_GRID)
    n_cols = len(JOINT_GRID[0])

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), sharex=True
    )

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            jname = JOINT_GRID[r][c]

            q_col = f"dof_pos_{jname}"
            a_col = f"action_{jname}"
            tq_col = f"target_q_{jname}"

            if q_col not in df.columns or a_col not in df.columns:
                ax.set_title(f"{jname} (missing data)", fontsize=AX_TITLE_FONTSIZE)
                ax.grid(True)
                continue

            q_rad = df[q_col].values
            a = df[a_col].values
            target_q_rad = df[tq_col].values

            q_deg = np.rad2deg(q_rad)
            target_q_deg = np.rad2deg(target_q_rad)

            ax.plot(t, q_deg, label="pos [deg]", color="b")
            ax.plot(t, target_q_deg, label="target [deg]", color="r", linestyle="--")

            pretty_title, pretty_ylabel = get_joint_display_labels(jname)
            ax.set_title(pretty_title, fontsize=AX_TITLE_FONTSIZE)
            ax.set_ylabel(pretty_ylabel, fontsize=AX_LABEL_FONTSIZE)
            ax.grid(True)

    for c in range(n_cols):
        axes[-1, c].set_xlabel("time [s]", fontsize=AX_LABEL_FONTSIZE)

    handles, labels = None, None
    for r in range(n_rows):
        for c in range(n_cols):
            h, l = axes[r, c].get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
        if handles is not None:
            break

    if handles is not None:
        fig.legend(
            handles,
            labels,
            loc="upper center",            # anchor의 위쪽 중앙을 기준
            bbox_to_anchor=(0.5, 0.05),    # figure 하단 쪽으로 내리기
            ncol=len(labels),
            fontsize=LEGEND_FONTSIZE,
        )
        
    fig.suptitle("G1 joint position vs target (deg)", fontsize=FIG_TITLE_FONTSIZE)
    
    fig.tight_layout(rect=[0, 0.05, 0.98, 0.96])
    fig.savefig(os.path.join(out_dir, "joint_pos_action_group.png"), dpi=300)
    plt.close(fig)


# ------------------ Torque vs joint velocity (4x3 layout) ------------------ #
def plot_torque_vs_vel_group(df: pd.DataFrame, out_dir: str):
    """
    4x3 grid, 각 subplot에
        x-axis: joint velocity [deg/s]  (dof_vel_{joint_name})
        y-axis: torque [Nm]             (torque_{joint_name})

    축은 원점 기준 좌우/상하 대칭이 되도록 설정.
    """
    os.makedirs(out_dir, exist_ok=True)

    has_vel = any(c.startswith("dof_vel_") for c in df.columns)
    has_tau = any(c.startswith("torque_") for c in df.columns)
    if not (has_vel and has_tau):
        print("[Analysis] No dof_vel_* or torque_* columns found, skip torque/velocity plots.")
        return

    n_rows = len(JOINT_GRID)
    n_cols = len(JOINT_GRID[0])

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), sharex=False, sharey=False
    )

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            jname = JOINT_GRID[r][c]

            dq_col = f"dof_vel_{jname}"
            tau_col = f"torque_{jname}"

            if dq_col not in df.columns or tau_col not in df.columns:
                ax.set_title(f"{jname} (missing data)", fontsize=AX_TITLE_FONTSIZE)
                ax.grid(True)
                continue

            dq_rad_s = df[dq_col].values
            tau = df[tau_col].values

            dq_deg_s = np.rad2deg(dq_rad_s)

            # scatter로 τ–ω 분포 시각화
            ax.scatter(dq_deg_s, tau, s=3, alpha=0.4)

            # --- 축을 원점 기준 대칭으로 맞추기 ---
            vmax = float(np.max(np.abs(dq_deg_s))) if dq_deg_s.size > 0 else 1.0
            tmax = float(np.max(np.abs(tau)))      if tau.size > 0 else 1.0

            # 0만 있는 경우를 방지해서 최소 범위 확보
            if vmax == 0:
                vmax = 1.0
            if tmax == 0:
                tmax = 1.0

            ax.set_xlim(-vmax, vmax)
            ax.set_ylim(-tmax, tmax)

            pretty_title, _ = get_joint_display_labels(jname)
            ax.set_title(pretty_title, fontsize=AX_TITLE_FONTSIZE)
            ax.set_xlabel("vel [deg/s]", fontsize=AX_LABEL_FONTSIZE)
            ax.set_ylabel("torque [Nm]", fontsize=AX_LABEL_FONTSIZE)
            ax.grid(True)

    fig.suptitle("G1 torque vs joint velocity", fontsize=FIG_TITLE_FONTSIZE)
    fig.tight_layout(rect=[0, 0.02, 0.98, 0.95])
    fig.savefig(os.path.join(out_dir, "torque_vs_vel_group.png"), dpi=300)
    plt.close(fig)

# ------------------ SUMMARY: distance, vx, y, z, yaw, reward ------------------ #

def plot_summary_group(df: pd.DataFrame, out_dir: str):
    """
    Single figure with 2x3 subplots:
        (1,1) x-y trajectory
        (1,2) z vs time
        (1,3) reward vs time
        (2,1) vx (actual) & command_0 vs time
        (2,2) vy (actual) & command_1 vs time
        (2,3) yaw (actual) & command_2 vs time
    """
    os.makedirs(out_dir, exist_ok=True)

    t = df["time"].values

    # base position
    x = df.get("base_pos_x", pd.Series(np.zeros_like(t))).values
    y = df.get("base_pos_y", pd.Series(np.zeros_like(t))).values
    z = df.get("base_pos_z", pd.Series(np.zeros_like(t))).values

    # velocities
    vx = df.get("base_lin_vel_x", pd.Series(np.zeros_like(t))).values
    vy = df.get("base_lin_vel_y", pd.Series(np.zeros_like(t))).values

    # reward
    rew = df.get("reward", pd.Series(np.zeros_like(t))).values

    # yaw (rad) → 상대 yaw(deg) + unwrap
    yaw_rad_raw = df.get("base_rpy_yaw", pd.Series(np.zeros_like(t))).values
    yaw0 = yaw_rad_raw[0]
    yaw_rad_centered = yaw_rad_raw - yaw0
    yaw_rad_unwrapped = np.unwrap(yaw_rad_centered)
    yaw_deg_unwrapped = np.rad2deg(yaw_rad_unwrapped)

    # commands (없으면 0으로 채움)
    cmd0 = df.get("command_0", pd.Series(np.zeros_like(t))).values  # vx command (m/s)
    cmd1 = df.get("command_1", pd.Series(np.zeros_like(t))).values  # vy command (m/s)
    cmd2 = df.get("command_2", pd.Series(np.zeros_like(t))).values  # yaw / yaw_rate command (rad or rad/s)

    # yaw command는 단순히 degree로 변환해서 같이 그린다
    cmd2_deg = np.rad2deg(cmd2)

    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=False)

    ax_xy   = axes[0, 0]
    ax_z    = axes[0, 1]
    ax_rew  = axes[0, 2]
    ax_vx   = axes[1, 0]
    ax_vy   = axes[1, 1]
    ax_yaw  = axes[1, 2]

    # 1) x-y trajectory
    ax_xy.plot(x, y)
    ax_xy.set_xlabel("x [m]", fontsize=AX_LABEL_FONTSIZE)
    ax_xy.set_ylabel("y [m]", fontsize=AX_LABEL_FONTSIZE)
    ax_xy.set_title("Base trajectory (x-y)", fontsize=AX_TITLE_FONTSIZE)
    ax_xy.grid(True)
    ax_xy.set_ylim(-0.1, 0.1)
    # ax_xy.axis("equal")

    # 2) z vs time
    ax_z.plot(t, z)
    ax_z.set_xlabel("time [s]", fontsize=AX_LABEL_FONTSIZE)
    ax_z.set_ylabel("z [m]", fontsize=AX_LABEL_FONTSIZE)
    ax_z.set_title("Base height vs time", fontsize=AX_TITLE_FONTSIZE)
    ax_z.grid(True)

    # 3) reward vs time
    ax_rew.plot(t, rew)
    ax_rew.set_xlabel("time [s]", fontsize=AX_LABEL_FONTSIZE)
    ax_rew.set_ylabel("reward", fontsize=AX_LABEL_FONTSIZE)
    ax_rew.set_title("Reward vs time", fontsize=AX_TITLE_FONTSIZE)
    ax_rew.grid(True)

    # 4) vx & command_0 vs time
    ax_vx.plot(t, vx, label="vx actual", color="b")
    ax_vx.plot(t, cmd0, label="vx command", color="r", linestyle="--")
    ax_vx.set_xlabel("time [s]", fontsize=AX_LABEL_FONTSIZE)
    ax_vx.set_ylabel("vx [m/s]", fontsize=AX_LABEL_FONTSIZE)
    ax_vx.set_title("Forward velocity vs command", fontsize=AX_TITLE_FONTSIZE)
    ax_vx.grid(True)
    ax_vx.legend(fontsize=LEGEND_FONTSIZE)

    # 5) vy & command_1 vs time
    ax_vy.plot(t, vy, label="vy actual", color="b")
    ax_vy.plot(t, cmd1, label="vy command", color="r", linestyle="--")
    ax_vy.set_xlabel("time [s]", fontsize=AX_LABEL_FONTSIZE)
    ax_vy.set_ylabel("vy [m/s]", fontsize=AX_LABEL_FONTSIZE)
    ax_vy.set_title("Lateral velocity vs command", fontsize=AX_TITLE_FONTSIZE)
    ax_vy.grid(True)
    ax_vy.legend(fontsize=LEGEND_FONTSIZE)

    # 6) yaw & command_2 vs time
    ax_yaw.plot(t, yaw_deg_unwrapped, label="yaw actual", color="b")
    ax_yaw.plot(t, cmd2_deg, label="yaw cmd (deg)", color="r", linestyle="--")
    ax_yaw.set_xlabel("time [s]", fontsize=AX_LABEL_FONTSIZE)
    ax_yaw.set_ylabel("yaw [deg]", fontsize=AX_LABEL_FONTSIZE)
    ax_yaw.set_title("Yaw vs yaw command", fontsize=AX_TITLE_FONTSIZE)
    ax_yaw.grid(True)
    ax_yaw.legend(fontsize=LEGEND_FONTSIZE)

    fig.suptitle(
        "Task Summary: Trajectory, Height, Reward, Velocities, Yaw & Commands",
        fontsize=FIG_TITLE_FONTSIZE,
    )

    fig.tight_layout(rect=[0, 0.02, 0.98, 0.95])
    fig.savefig(os.path.join(out_dir, "summary_group.png"), dpi=300)
    plt.close(fig)

def analyze_single_csv(csv_path: str, out_root: str):
    """Load one CSV and save plots under out_root/<csv_stem>/."""
    df = load_csv(csv_path)

    csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.join(out_root, csv_stem)
    os.makedirs(out_dir, exist_ok=True)

    plot_joint_pos_action_group(df, out_dir)
    plot_torque_vs_vel_group(df, out_dir)
    plot_summary_group(df, out_dir)

    print(f"[Analysis] Saved plots for {csv_stem} to: {out_dir}")

# ------------------ main ------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to single timeseries CSV (e.g., logs/eval/g1/walk_xxx.csv)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing multiple CSVs to analyze (all *.csv will be processed)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="analysis_plots",
        help="Root directory to store plots (each CSV gets its own subfolder)",
    )
    args = parser.parse_args()

    if args.input_dir is not None:
        in_dir = os.path.abspath(args.input_dir)
        if not os.path.isdir(in_dir):
            raise NotADirectoryError(f"input_dir is not a directory: {in_dir}")

        csv_paths = []
        for root, _, files in os.walk(in_dir):
            for f in files:
                if f.lower().endswith(".csv"):
                    csv_paths.append(os.path.join(root, f))

        if not csv_paths:
            print(f"[Analysis] No CSV files found under: {in_dir}")
            return

        print(f"[Analysis] Found {len(csv_paths)} CSV files under {in_dir}:")
        for path in csv_paths:
            print(f"  - {path}")

        for path in csv_paths:
            analyze_single_csv(path, args.out_dir)

    elif args.csv is not None:
        analyze_single_csv(args.csv, args.out_dir)
    else:
        parser.error("Either --csv or --input_dir must be provided.")

if __name__ == "__main__":
    main()
