#!/usr/bin/env python3
"""
Simple trajectory replay → MP4.

- Input can be a single episode .npz file OR a folder of .npz files.
- We visualize observations over time as a heatmap (agents × features).
- Robust to different key names and shapes:
    obs_mat OR obs OR observations with shape (T, A, D) or (T, D).

Usage examples:
    python tools/replay.py artifacts/run_20250101_1200/traj --out artifacts/replay.mp4 --fps 12
    python tools/replay.py artifacts/run_20250101_1200/traj/ep_3.npz --out replay_ep3.mp4
"""

from __future__ import annotations
import argparse
import os
import glob
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from typing import List, Tuple


def _find_npz_items(path: str) -> List[str]:
    """Return a list of .npz files. If `path` is a file, return [path]."""
    if os.path.isfile(path):
        return [path]
    # assume directory
    files = sorted(glob.glob(os.path.join(path, "*.npz")))
    return files


def _load_obs(npz_path: str) -> np.ndarray:
    """
    Load observations from a .npz file. Try common keys and normalize shape.

    Returns:
        obs: np.ndarray of shape (T, A, D)
    Raises:
        ValueError if no obs-like key is found.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        for k in ("obs_mat", "obs", "observations"):
            if k in data:
                arr = data[k]
                break
        else:
            raise ValueError(
                f"No observation key found in {npz_path}. "
                "Expected one of: obs_mat, obs, observations."
            )

    arr = np.asarray(arr)
    if arr.ndim == 2:
        # (T, D) -> (T, 1, D)
        arr = arr[:, None, :]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported obs shape {arr.shape} in {npz_path}; expected (T,A,D) or (T,D).")
    return arr  # (T, A, D)


def _normalize_for_visual(obs_t: np.ndarray) -> np.ndarray:
    """
    Normalize a single time-step obs (A, D) to [0,1] per-feature-window for heatmap.
    We keep it simple/robust: min/max across the whole frame with epsilon guard.
    """
    x = obs_t.astype(np.float32)
    mn = np.min(x)
    mx = np.max(x)
    if mx - mn < 1e-6:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def _render_episode_heatmap(
    obs: np.ndarray,
    writer: imageio.FFMPEGWriter,
    dpi: int = 120,
    title_prefix: str = ""
) -> None:
    """
    Render one episode as a sequence of heatmaps (agents × features).

    Args:
        obs: (T, A, D)
        writer: open imageio writer
    """
    T, A, D = obs.shape

    # Pre-create the figure/axes to avoid slow matplotlib re-creation
    fig, ax = plt.subplots(figsize=(max(4, D * 0.18), max(2.5, A * 0.45)), dpi=dpi)
    im = None

    for t in range(T):
        frame = _normalize_for_visual(obs[t])  # (A, D) in [0,1]

        ax.clear()
        ax.set_title(f"{title_prefix}t={t}  (A={A}, D={D})", fontsize=10)
        im = ax.imshow(frame, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
        ax.set_xlabel("feature")
        ax.set_ylabel("agent")
        # Minimal ticks to keep it readable on many shapes
        ax.set_xticks([0, D - 1] if D > 1 else [0])
        ax.set_yticks([0, A - 1] if A > 1 else [0])

        # Draw a small colorbar only on first few frames for speed
        if t == 0:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("normalized value", rotation=270, labelpad=12)

        fig.tight_layout()

        # Convert fig to numpy frame for video
        fig.canvas.draw()
        frame_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame_img = frame_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.append_data(frame_img)

    plt.close(fig)


def make_video(trajectory_path: str, out_mp4: str, fps: int) -> Tuple[int, List[str]]:
    """
    Create an MP4 from a trajectory folder or single .npz.

    Returns:
        (num_episodes_rendered, list_of_sources)
    """
    npz_files = _find_npz_items(trajectory_path)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found at: {trajectory_path}")

    # Ensure output directory
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)

    # A slightly larger bitrate helps readability for heatmaps
    writer = imageio.get_writer(out_mp4, fps=fps, codec="libx264", bitrate="8000k", quality=8)
    used = []

    try:
        for idx, f in enumerate(npz_files, start=1):
            obs = _load_obs(f)  # (T, A, D)
            title = f"Episode {idx}/{len(npz_files)} — {os.path.basename(f)} "
            _render_episode_heatmap(obs, writer, title_prefix=title)
            used.append(f)
    finally:
        writer.close()

    return len(used), used


def main():
    parser = argparse.ArgumentParser(description="Replay recorded trajectories as MP4.")
    parser.add_argument("trajectory_path", help="Path to trajectory .npz or folder containing .npz files")
    parser.add_argument("--out", type=str, default="replay.mp4", help="Output MP4 file path")
    parser.add_argument("--fps", type=int, default=12, help="Frames per second")
    parser.add_argument("--env_id", type=str, default="simple_tag_v3",
                        help="(Reserved) PettingZoo env id if you later add env-based rendering")
    args = parser.parse_args()

    print(f"[replay] input={args.trajectory_path}")
    print(f"[replay] out={args.out}  fps={args.fps}")

    try:
        n, used = make_video(args.trajectory_path, args.out, args.fps)
    except Exception as e:
        print(f"[replay] ERROR: {e}")
        raise

    print(f"[replay] Wrote {args.out} using {n} episode(s).")
    if used:
        print("[replay] sources:")
        for u in used[-5:]:
            print("  -", u)


if __name__ == "__main__":
    main()
