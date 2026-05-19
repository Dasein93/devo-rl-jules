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
import os, glob, json
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


def _episode_index(path: str) -> int:
    """Extract the integer N from a basename like 'ep_<N>.npz'. Falls back to 0."""
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    if stem.startswith("ep_"):
        try:
            return int(stem[3:])
        except ValueError:
            pass
    return 0


def _find_trajectory_files(path: str) -> List[str]:
    """Return a list of .npz files in chronological (episode-index) order.
    If `path` is a file, return [path]."""
    if os.path.isfile(path):
        return [path]
    files = glob.glob(os.path.join(path, "*.npz"))
    return sorted(files, key=lambda p: (_episode_index(p), p))


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


def _load_positions_and_names(npz_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load positions and agent_names from a .npz file.
    Returns:
        (pos, agent_names)
        pos: np.ndarray of shape (T, A, 2)
        agent_names: list of strings
    """
    with np.load(npz_path, allow_pickle=True) as data:
        if "pos" not in data:
            raise ValueError(f"No position key 'pos' found in {npz_path}.")
        if "agent_names" not in data:
            raise ValueError(f"No agent_names key found in {npz_path}.")

        pos = np.asarray(data["pos"])
        names = list(data["agent_names"])

        if pos.ndim != 3 or pos.shape[2] != 2:
            raise ValueError(f"Expected pos shape (T,A,2), got {pos.shape}")
        if len(names) != pos.shape[1]:
            raise ValueError(f"Agent name/position mismatch: {len(names)} vs {pos.shape[1]}")

        return pos, names


def _load_pos_alive_names(npz_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Like _load_positions_and_names but also returns the (T, A) alive mask
    for the ecosystem replay. If no 'alive' key is present, defaults to all-True."""
    with np.load(npz_path, allow_pickle=True) as data:
        pos, names = _load_positions_and_names(npz_path)
        if "alive" in data:
            alive = np.asarray(data["alive"], dtype=bool)
        else:
            alive = np.ones(pos.shape[:2], dtype=bool)
        return pos, alive, names


def _load_genome(npz_path: str) -> Optional[np.ndarray]:
    """Load (T, A, G) genome array if present, else None."""
    with np.load(npz_path, allow_pickle=True) as data:
        if "genome" not in data:
            return None
        return np.asarray(data["genome"], dtype=np.float32)


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
        rgba_buf = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        frame_img = np.frombuffer(rgba_buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        if w % 16 != 0 or h % 16 != 0:
             w = (w // 16) * 16
             h = (h // 16) * 16
             frame_img = frame_img[:h, :w, :]
        writer.append_data(frame_img)

    plt.close(fig)


def _render_episode_positions(
    positions: np.ndarray,
    agent_names: List[str],
    writer: imageio.FFMPEGWriter,
    title_prefix: str = "",
    dpi: int = 120,
    frameskip: int = 1,
    trail: int = 0,
) -> None:
    """Render one episode as a 2D scatter plot of agent positions.

    `trail`: if >0, draw the last `trail` positions per agent as a fading line."""
    T, A, _ = positions.shape
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    world_bounds = [np.min(positions) - 0.1, np.max(positions) + 0.1]

    predator_indices = [i for i, name in enumerate(agent_names) if "adversary" in name]
    prey_indices = [i for i, name in enumerate(agent_names) if "adversary" not in name]

    for t in range(0, T, frameskip):
        ax.clear()
        ax.set_xlim(world_bounds)
        ax.set_ylim(world_bounds)
        ax.set_aspect("equal")
        ax.set_title(f"{title_prefix}t={t}", fontsize=10)

        if trail > 0 and t > 0:
            start = max(0, t - trail)
            past = positions[start:t + 1]  # (k, A, 2)
            k = past.shape[0]
            alphas = np.linspace(0.1, 0.7, max(1, k - 1))
            for i in range(k - 1):
                seg = past[i:i + 2]  # (2, A, 2)
                for j in predator_indices:
                    ax.plot(seg[:, j, 0], seg[:, j, 1], color="red", alpha=alphas[i], linewidth=1.2)
                for j in prey_indices:
                    ax.plot(seg[:, j, 0], seg[:, j, 1], color="green", alpha=alphas[i], linewidth=1.2)

        pos_t = positions[t]
        if predator_indices:
            ax.scatter(pos_t[predator_indices, 0], pos_t[predator_indices, 1],
                       c="red", label="Predators", s=120, edgecolors="black", linewidths=0.8)
        if prey_indices:
            ax.scatter(pos_t[prey_indices, 0], pos_t[prey_indices, 1],
                       c="green", label="Prey", s=120, edgecolors="black", linewidths=0.8)

        if t == 0:
            ax.legend(loc="upper right", fontsize=8)

        fig.canvas.draw()
        rgba_buf = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        frame_img = np.frombuffer(rgba_buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        if w % 16 != 0 or h % 16 != 0:
            w = (w // 16) * 16
            h = (h // 16) * 16
            frame_img = frame_img[:h, :w, :]
        writer.append_data(frame_img)

    plt.close(fig)


def _render_episode_ecosystem(
    positions: np.ndarray,
    alive: np.ndarray,
    agent_names: List[str],
    writer: imageio.FFMPEGWriter,
    pop_pred_global: np.ndarray,
    pop_prey_global: np.ndarray,
    cumulative_offset: int,
    title_prefix: str = "",
    dpi: int = 120,
    frameskip: int = 1,
    trail: int = 0,
    genome: Optional[np.ndarray] = None,
) -> None:
    """Render one ecosystem episode as a split panel.

    Left: 2D scatter of currently-alive predators (red) and prey (green),
    with a fading trail for each alive agent.
    Right: population-over-time line plot covering ALL episodes so far,
    with a marker at the current global step.

    `pop_pred_global` / `pop_prey_global` are the full multi-episode
    population time series (concatenated across episodes). `cumulative_offset`
    is the global step index at the start of this episode.
    """
    T, A, _ = positions.shape
    predator_idx = [i for i, n in enumerate(agent_names) if "adversary" in str(n)]
    prey_idx = [i for i, n in enumerate(agent_names) if "adversary" not in str(n)]

    # World bounds: use the bounding box over all *alive* positions, plus a margin.
    alive_pos = positions[alive]
    if alive_pos.size > 0:
        lo = float(np.nanmin(alive_pos)) - 0.1
        hi = float(np.nanmax(alive_pos)) + 0.1
    else:
        lo, hi = -1.0, 1.0
    world_bounds = (lo, hi)

    fig, (ax_world, ax_pop) = plt.subplots(1, 2, figsize=(11, 5), dpi=dpi,
                                           gridspec_kw={"width_ratios": [1.1, 1.4]})

    pop_max = max(int(pop_pred_global.max()), int(pop_prey_global.max()), 1)
    n_global = len(pop_pred_global)

    for t in range(0, T, frameskip):
        # --- Left panel: world ---
        ax_world.clear()
        ax_world.set_xlim(world_bounds)
        ax_world.set_ylim(world_bounds)
        ax_world.set_aspect("equal")
        ax_world.set_title(f"{title_prefix}t={t}", fontsize=10)

        pred_alive_now = [i for i in predator_idx if alive[t, i]]
        prey_alive_now = [i for i in prey_idx if alive[t, i]]
        n_pred_now = len(pred_alive_now)
        n_prey_now = len(prey_alive_now)

        if trail > 0 and t > 0:
            start = max(0, t - trail)
            past = positions[start:t + 1]            # (k, A, 2)
            past_alive = alive[start:t + 1]          # (k, A) — only draw segments where agent is alive
            k = past.shape[0]
            alphas = np.linspace(0.1, 0.55, max(1, k - 1))
            for i in range(k - 1):
                # Draw a line segment for each agent alive at both endpoints.
                both_alive = past_alive[i] & past_alive[i + 1]
                seg = past[i:i + 2]                  # (2, A, 2)
                for j in predator_idx:
                    if both_alive[j]:
                        ax_world.plot(seg[:, j, 0], seg[:, j, 1],
                                       color="red", alpha=alphas[i], linewidth=1.0)
                for j in prey_idx:
                    if both_alive[j]:
                        ax_world.plot(seg[:, j, 0], seg[:, j, 1],
                                       color="green", alpha=alphas[i], linewidth=1.0)

        pos_t = positions[t]

        def _sizes_and_alphas(idxs: List[int]):
            """For each agent index, return (marker size, marker alpha) based on
            its genome at this timestep. Defaults if no genome data."""
            if genome is None:
                return [100] * len(idxs), [0.9] * len(idxs)
            g_now = genome[t, idxs]   # (n, G), G >= 2
            # marker size scales with hp_g (gene 1): roughly [60, 200]
            sizes = (60.0 + 90.0 * np.clip(g_now[:, 1], 0.5, 2.0)).tolist()
            # marker alpha by speed_g (gene 0): faster = more saturated
            alphas = (0.45 + 0.45 * np.clip((g_now[:, 0] - 0.5) / 1.5, 0.0, 1.0)).tolist()
            return sizes, alphas

        if pred_alive_now:
            xs = pos_t[pred_alive_now, 0]; ys = pos_t[pred_alive_now, 1]
            sizes, alphas = _sizes_and_alphas(pred_alive_now)
            for x, y, s, a in zip(xs, ys, sizes, alphas):
                ax_world.scatter([x], [y], c="red", s=s, alpha=a,
                                  edgecolors="black", linewidths=0.6)
            ax_world.scatter([], [], c="red", s=100, edgecolors="black", linewidths=0.6,
                              label=f"Predators ({n_pred_now})")
        if prey_alive_now:
            xs = pos_t[prey_alive_now, 0]; ys = pos_t[prey_alive_now, 1]
            sizes, alphas = _sizes_and_alphas(prey_alive_now)
            for x, y, s, a in zip(xs, ys, sizes, alphas):
                ax_world.scatter([x], [y], c="green", s=s, alpha=a,
                                  edgecolors="black", linewidths=0.6)
            ax_world.scatter([], [], c="green", s=100, edgecolors="black", linewidths=0.6,
                              label=f"Prey ({n_prey_now})")
        ax_world.legend(loc="upper right", fontsize=8)
        ax_world.set_xticks([]); ax_world.set_yticks([])

        # --- Right panel: population curves (global, with current marker) ---
        ax_pop.clear()
        x = np.arange(n_global)
        ax_pop.plot(x, pop_pred_global, color="red", linewidth=1.0, label="Predators")
        ax_pop.plot(x, pop_prey_global, color="green", linewidth=1.0, label="Prey")
        cur = cumulative_offset + t
        ax_pop.axvline(cur, color="black", linewidth=0.8, alpha=0.5)
        ax_pop.set_xlim(0, max(1, n_global - 1))
        ax_pop.set_ylim(0, pop_max + 2)
        ax_pop.set_xlabel("global step")
        ax_pop.set_ylabel("population")
        ax_pop.set_title("Population over training", fontsize=10)
        ax_pop.legend(loc="upper right", fontsize=8)
        ax_pop.grid(alpha=0.3)

        fig.tight_layout()
        fig.canvas.draw()
        rgba_buf = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        frame_img = np.frombuffer(rgba_buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        if w % 16 != 0 or h % 16 != 0:
            w = (w // 16) * 16
            h = (h // 16) * 16
            frame_img = frame_img[:h, :w, :]
        writer.append_data(frame_img)

    plt.close(fig)


def make_video(
    trajectory_path: str,
    out_mp4: str,
    fps: int,
    mode: str = "heatmap",
    dpi: int = 120,
    frameskip: int = 1,
    trail: int = 0,
) -> Tuple[int, List[str]]:
    """
    Create an MP4 from a trajectory folder or single .npz.

    Returns:
        (num_episodes_rendered, list_of_sources)
    """
    npz_files = _find_trajectory_files(trajectory_path)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found at: {trajectory_path}")

    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    writer = imageio.get_writer(out_mp4, fps=fps, codec="libx264", bitrate="8000k", quality=8)
    used = []

    # For ecosystem mode, build the global population time series up-front so
    # the right panel can show "where we are in training" alongside each frame.
    pop_pred_global = None
    pop_prey_global = None
    ep_offsets = None
    if mode == "ecosystem":
        pop_pred_runs = []
        pop_prey_runs = []
        ep_offsets = []
        running = 0
        for f in npz_files:
            try:
                _pos, _alive, names = _load_pos_alive_names(f)
            except ValueError as e:
                print(f"Skipping {f} for ecosystem replay: {e}")
                continue
            pred_idx_local = [i for i, n in enumerate(names) if "adversary" in str(n)]
            prey_idx_local = [i for i, n in enumerate(names) if "adversary" not in str(n)]
            pp = _alive[:, pred_idx_local].sum(axis=1)
            qq = _alive[:, prey_idx_local].sum(axis=1)
            pop_pred_runs.append(pp)
            pop_prey_runs.append(qq)
            ep_offsets.append(running)
            running += len(pp)
        if not pop_pred_runs:
            raise FileNotFoundError(f"No ecosystem-compatible npz files in {trajectory_path}")
        pop_pred_global = np.concatenate(pop_pred_runs)
        pop_prey_global = np.concatenate(pop_prey_runs)

    try:
        for idx, f in enumerate(npz_files, start=1):
            title = f"Episode {idx}/{len(npz_files)} — {os.path.basename(f)} "
            if mode == "positions":
                try:
                    positions, agent_names = _load_positions_and_names(f)
                    _render_episode_positions(positions, agent_names, writer, title_prefix=title,
                                              dpi=dpi, frameskip=frameskip, trail=trail)
                except ValueError as e:
                    print(f"Skipping {f} for position replay: {e}")
                    continue
            elif mode == "ecosystem":
                try:
                    positions, alive, agent_names = _load_pos_alive_names(f)
                    genome = _load_genome(f)
                    _render_episode_ecosystem(
                        positions, alive, agent_names, writer,
                        pop_pred_global=pop_pred_global,
                        pop_prey_global=pop_prey_global,
                        cumulative_offset=ep_offsets[idx - 1] if ep_offsets else 0,
                        title_prefix=title, dpi=dpi, frameskip=frameskip, trail=trail,
                        genome=genome,
                    )
                except ValueError as e:
                    print(f"Skipping {f} for ecosystem replay: {e}")
                    continue
            else:  # heatmap
                obs = _load_obs(f)
                _render_episode_heatmap(obs, writer, dpi=dpi, title_prefix=title)
            used.append(f)
    finally:
        writer.close()

    return len(used), used


def main():
    parser = argparse.ArgumentParser(description="Replay recorded trajectories as MP4.")
    parser.add_argument("in", help="Path to trajectory .npz or folder")
    parser.add_argument("--out", type=str, default="replay.mp4", help="Output MP4 file path")
    parser.add_argument("--fps", type=int, default=12, help="Frames per second")
    parser.add_argument("--mode", type=str,
                        choices=["heatmap", "positions", "ecosystem"], default="heatmap",
                        help="Replay mode: 'heatmap' for obs, 'positions' for 2D scatter, "
                             "'ecosystem' for split-panel (2D scene + population curve)")
    parser.add_argument("--dpi", type=int, default=120, help="DPI for rendering frames")
    parser.add_argument("--frameskip", type=int, default=1, help="Render 1 of N frames")
    parser.add_argument("--trail", type=int, default=0,
                        help="positions mode only: draw last N positions per agent as a fading trail")
    args = parser.parse_args()

    # Rename 'in' to 'trajectory_path' for clarity
    args.trajectory_path = getattr(args, "in")

    print(f"[replay] input={args.trajectory_path}")
    print(f"[replay] out={args.out} fps={args.fps} mode={args.mode}")

    try:
        n, used = make_video(
            args.trajectory_path,
            args.out,
            args.fps,
            mode=args.mode,
            dpi=args.dpi,
            frameskip=args.frameskip,
            trail=args.trail,
        )
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
