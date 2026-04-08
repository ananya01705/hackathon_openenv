"""
================================================================
  Integrated 2D + 3D Space Traffic Control System
  Satellite Collision Avoidance via Reinforcement Learning
================================================================
  Analogy:
    Satellite  → Vehicle
    Debris     → Pothole / Traffic congestion
    Space      → Road network
    Collision  → Accident
    Avoidance  → Lane change / Rerouting

  HOW TO RUN:
    pip install numpy matplotlib
    python space_traffic_control.py

  Switch mode at the bottom of this file:
    mode = "2D"   ← top-down road-traffic view
    mode = "3D"   ← realistic 3D space navigation
================================================================
"""

import numpy as np
import matplotlib
# Auto-select a GUI backend; falls back to Agg (no window) if none available
_BACKENDS = ["TkAgg", "Qt5Agg", "WXAgg", "MacOSX", "Agg"]
for _be in _BACKENDS:
    try:
        matplotlib.use(_be)
        import matplotlib.pyplot as _test_plt  # noqa: F401
        break
    except Exception:
        continue
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (side-effect import)
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import time
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
SPACE_SIZE       = 100.0   # boundary of the simulation universe
DANGER_RADIUS    = 8.0     # minimum safe distance from debris
COLLISION_RADIUS = 3.0     # hard collision threshold
NUM_DEBRIS       = 12      # total debris objects
MAX_STEPS        = 500     # episode length
N_NEAREST        = 3       # how many nearest debris to include in state
SPEED_BASE       = 1.5     # nominal satellite speed per timestep
SPEED_MAX        = 3.0
SPEED_MIN        = 0.5
DEBRIS_SPEED     = 0.3     # debris drift speed


# ─────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────
@dataclass
class Satellite:
    pos:   np.ndarray          # [x, y] or [x, y, z]
    vel:   np.ndarray          # velocity vector
    angle: float = 0.0         # heading (radians) – 2D
    pitch: float = 0.0         # pitch angle       – 3D
    yaw:   float = 0.0         # yaw angle         – 3D
    speed: float = SPEED_BASE

    def copy(self):
        return Satellite(
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            angle=self.angle,
            pitch=self.pitch,
            yaw=self.yaw,
            speed=self.speed,
        )


@dataclass
class Debris:
    pos: np.ndarray
    vel: np.ndarray

    def copy(self):
        return Debris(pos=self.pos.copy(), vel=self.vel.copy())


# ─────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ─────────────────────────────────────────────
def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def wrap_pos(pos, size=SPACE_SIZE):
    """Toroidal wrap-around so objects stay in-bounds."""
    return pos % size


def distances(satellite_pos: np.ndarray, debris_list: List[Debris]) -> np.ndarray:
    """Return array of Euclidean distances from satellite to each debris."""
    return np.array([np.linalg.norm(satellite_pos - d.pos) for d in debris_list])


def nearest_n(satellite_pos: np.ndarray, debris_list: List[Debris], n: int):
    """Return indices of the n nearest debris objects."""
    dists = distances(satellite_pos, debris_list)
    return np.argsort(dists)[:n]


def predict_future_pos(obj_pos, obj_vel, steps=5):
    """Simple linear extrapolation of position."""
    return obj_pos + obj_vel * steps


def risk_level(satellite_pos, debris_list):
    """Normalised risk score: 0 = safe, 1 = imminent collision."""
    dists = distances(satellite_pos, debris_list)
    min_d = dists.min()
    if min_d <= COLLISION_RADIUS:
        return 1.0
    if min_d >= DANGER_RADIUS * 2:
        return 0.0
    return 1.0 - (min_d - COLLISION_RADIUS) / (DANGER_RADIUS * 2 - COLLISION_RADIUS)


# ─────────────────────────────────────────────
#  GYM-LIKE ENVIRONMENT
# ─────────────────────────────────────────────
class SpaceTrafficEnv:
    """
    Integrated 2D / 3D Space Traffic Control Environment.

    mode = "2D"  →  5 actions, (x,y) state
    mode = "3D"  →  7 actions, (x,y,z) state
    """

    # ------ action spaces ------
    ACTIONS_2D = {
        0: "maintain",
        1: "turn_left",
        2: "turn_right",
        3: "accelerate",
        4: "decelerate",
    }
    ACTIONS_3D = {
        0: "maintain",
        1: "yaw_left",
        2: "yaw_right",
        3: "pitch_up",
        4: "pitch_down",
        5: "accelerate",
        6: "decelerate",
    }

    def __init__(self, mode: str = "2D", render_mode: str = "human"):
        assert mode in ("2D", "3D"), "mode must be '2D' or '3D'"
        self.mode        = mode
        self.render_mode = render_mode
        self.dim         = 2 if mode == "2D" else 3
        self.n_actions   = 5 if mode == "2D" else 7
        self.step_count  = 0
        self.done        = False
        self.total_reward = 0.0

        # matplotlib handles
        self.fig = None
        self.ax  = None

        # trajectory history for rendering
        self.traj: List[np.ndarray] = []

        self.satellite: Optional[Satellite] = None
        self.debris_list: List[Debris]      = []

        self.reset()

    # ── reset ──────────────────────────────────
    def reset(self):
        self.step_count   = 0
        self.done         = False
        self.total_reward = 0.0
        self.traj         = []

        center = np.array([SPACE_SIZE * 0.1] * self.dim, dtype=float)
        vel    = self._initial_vel()

        self.satellite = Satellite(pos=center, vel=vel,
                                   angle=0.0, pitch=0.0, yaw=0.0,
                                   speed=SPEED_BASE)

        # scatter debris randomly, keeping them away from the start
        self.debris_list = []
        rng = np.random.default_rng()
        for _ in range(NUM_DEBRIS):
            while True:
                p = rng.uniform(5, SPACE_SIZE - 5, self.dim)
                if np.linalg.norm(p - center) > DANGER_RADIUS * 3:
                    break
            v = rng.uniform(-DEBRIS_SPEED, DEBRIS_SPEED, self.dim)
            self.debris_list.append(Debris(pos=p, vel=v))

        return self._get_obs()

    # ── step ───────────────────────────────────
    def step(self, action: int):
        assert not self.done, "Episode over; call reset()."
        self.step_count += 1

        prev_pos = self.satellite.pos.copy()

        # apply action
        reward = self._apply_action(action)

        # move satellite
        self._move_satellite()

        # drift debris
        self._drift_debris()

        # collision detection
        dists = distances(self.satellite.pos, self.debris_list)
        min_dist = dists.min()

        if min_dist <= COLLISION_RADIUS:
            reward -= 10.0
            self.done = True

        elif min_dist <= DANGER_RADIUS:
            reward -= 3.0   # entered danger zone
            reward -= 0.5   # too close to debris

        else:
            reward += 1.0   # safe timestep

            # near-future collision prediction bonus
            future_sat = predict_future_pos(self.satellite.pos, self.satellite.vel)
            future_dists = np.array([
                np.linalg.norm(future_sat - predict_future_pos(d.pos, d.vel))
                for d in self.debris_list
            ])
            if future_dists.min() <= DANGER_RADIUS and min_dist > DANGER_RADIUS:
                reward += 5.0   # proactive avoidance

        # smooth trajectory bonus
        delta = np.linalg.norm(self.satellite.pos - prev_pos)
        if SPEED_MIN * 0.8 < delta < SPEED_MAX * 1.2:
            reward += 2.0

        self.total_reward += reward
        self.traj.append(self.satellite.pos.copy())

        if self.step_count >= MAX_STEPS:
            self.done = True

        obs  = self._get_obs()
        info = {"step": self.step_count, "min_dist": min_dist,
                "risk": risk_level(self.satellite.pos, self.debris_list)}
        return obs, reward, self.done, info

    # ── render ─────────────────────────────────
    def render(self):
        if self.render_mode != "human":
            return

        if self.fig is None:
            plt.ion()
            if self.mode == "2D":
                self.fig, self.ax = plt.subplots(figsize=(9, 9))
                self.fig.patch.set_facecolor("#0a0a1a")
                self.ax.set_facecolor("#0a0a1a")
            else:
                self.fig = plt.figure(figsize=(11, 9))
                self.fig.patch.set_facecolor("#0a0a1a")
                self.ax = self.fig.add_subplot(111, projection="3d")
                self.ax.set_facecolor("#0a0a1a")
            self.fig.canvas.manager.set_window_title(
                f"Space Traffic Control — {self.mode} Mode"
            )

        self.ax.cla()
        self._style_axes()

        if self.mode == "2D":
            self._render_2d()
        else:
            self._render_3d()

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax  = None

    # ── private helpers ────────────────────────

    def _initial_vel(self):
        """Starting velocity vector pointing roughly toward centre."""
        angle = random.uniform(0, 2 * np.pi)
        if self.dim == 2:
            return np.array([np.cos(angle), np.sin(angle)]) * SPEED_BASE
        return np.array([np.cos(angle), np.sin(angle),
                         random.uniform(-0.3, 0.3)]) * SPEED_BASE

    def _apply_action(self, action: int) -> float:
        """Mutate satellite heading/speed; return fuel penalty if moving."""
        fuel_penalty = 0.0
        sat = self.satellite

        if self.mode == "2D":
            turn = np.radians(15)
            if   action == 1:   sat.angle += turn;  fuel_penalty = -1.0
            elif action == 2:   sat.angle -= turn;  fuel_penalty = -1.0
            elif action == 3:   sat.speed  = clamp(sat.speed + 0.3, SPEED_MIN, SPEED_MAX); fuel_penalty = -1.0
            elif action == 4:   sat.speed  = clamp(sat.speed - 0.3, SPEED_MIN, SPEED_MAX); fuel_penalty = -1.0
            # update velocity from angle
            sat.vel = np.array([np.cos(sat.angle), np.sin(sat.angle)]) * sat.speed

        else:  # 3D
            delta_ang = np.radians(12)
            if   action == 1:   sat.yaw   += delta_ang; fuel_penalty = -1.0
            elif action == 2:   sat.yaw   -= delta_ang; fuel_penalty = -1.0
            elif action == 3:   sat.pitch += delta_ang; fuel_penalty = -1.0
            elif action == 4:   sat.pitch -= delta_ang; fuel_penalty = -1.0
            elif action == 5:   sat.speed  = clamp(sat.speed + 0.3, SPEED_MIN, SPEED_MAX); fuel_penalty = -1.0
            elif action == 6:   sat.speed  = clamp(sat.speed - 0.3, SPEED_MIN, SPEED_MAX); fuel_penalty = -1.0
            # rebuild 3D velocity from yaw + pitch
            sat.vel = np.array([
                np.cos(sat.pitch) * np.cos(sat.yaw),
                np.cos(sat.pitch) * np.sin(sat.yaw),
                np.sin(sat.pitch),
            ]) * sat.speed

        return fuel_penalty

    def _move_satellite(self):
        self.satellite.pos = wrap_pos(self.satellite.pos + self.satellite.vel)

    def _drift_debris(self):
        for d in self.debris_list:
            d.pos = wrap_pos(d.pos + d.vel)

    def _get_obs(self) -> np.ndarray:
        """
        State vector:
          satellite pos (dim) + velocity (dim) +
          N nearest debris positions (N*dim) +
          distance to closest (1) + risk level (1)
        """
        sat   = self.satellite
        idx   = nearest_n(sat.pos, self.debris_list, N_NEAREST)
        dists = distances(sat.pos, self.debris_list)

        obs = list(sat.pos) + list(sat.vel)
        for i in idx:
            obs += list(self.debris_list[i].pos)
        obs.append(dists.min())
        obs.append(risk_level(sat.pos, self.debris_list))
        return np.array(obs, dtype=np.float32)

    # ── styling ────────────────────────────────
    def _style_axes(self):
        ax = self.ax
        if self.mode == "2D":
            ax.set_xlim(0, SPACE_SIZE)
            ax.set_ylim(0, SPACE_SIZE)
            ax.set_aspect("equal")
            ax.tick_params(colors="#555577")
            ax.spines[["top","right","left","bottom"]].set_color("#222244")
            ax.set_title(
                f"2D Space Traffic  |  Step {self.step_count}  |  "
                f"Reward {self.total_reward:.1f}",
                color="#aaaacc", fontsize=11
            )
            # faint grid
            ax.grid(True, color="#1a1a3a", linewidth=0.5)
        else:
            ax.set_xlim(0, SPACE_SIZE)
            ax.set_ylim(0, SPACE_SIZE)
            ax.set_zlim(0, SPACE_SIZE)
            ax.tick_params(colors="#555577")
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("#1a1a3a")
            ax.yaxis.pane.set_edgecolor("#1a1a3a")
            ax.zaxis.pane.set_edgecolor("#1a1a3a")
            ax.set_title(
                f"3D Space Traffic  |  Step {self.step_count}  |  "
                f"Reward {self.total_reward:.1f}",
                color="#aaaacc", fontsize=11
            )

    def _render_2d(self):
        ax  = self.ax
        sat = self.satellite

        # ── trajectory ──
        if len(self.traj) > 1:
            traj = np.array(self.traj)
            ax.plot(traj[:, 0], traj[:, 1], color="#2244aa", linewidth=0.8,
                    alpha=0.6, linestyle="--", label="trajectory")

        # ── debris ──
        for d in self.debris_list:
            dist = np.linalg.norm(sat.pos - d.pos)
            color = "#ff4444" if dist < DANGER_RADIUS * 1.5 else "#cc2222"
            # danger circle
            circle = plt.Circle(d.pos, DANGER_RADIUS,
                                 color=color, alpha=0.12, linewidth=0)
            ax.add_patch(circle)
            border = plt.Circle(d.pos, DANGER_RADIUS,
                                 color=color, alpha=0.5, fill=False,
                                 linewidth=0.8, linestyle=":")
            ax.add_patch(border)
            ax.scatter(*d.pos, color=color, s=60, zorder=5,
                       edgecolors="#ff8888", linewidths=0.5)

            # velocity arrow
            ax.annotate("", xy=d.pos + d.vel * 6, xytext=d.pos,
                        arrowprops=dict(arrowstyle="->",
                                        color="#ff6644", lw=0.8))

        # ── satellite ──
        ax.scatter(*sat.pos, color="#33ff77", s=120, zorder=10,
                   edgecolors="#aaffcc", linewidths=1.5, label="satellite")
        # heading arrow
        ax.annotate("", xy=sat.pos + sat.vel * 3, xytext=sat.pos,
                    arrowprops=dict(arrowstyle="->",
                                    color="#55ffaa", lw=1.5))
        # safety bubble
        bubble = plt.Circle(sat.pos, COLLISION_RADIUS,
                             color="#33ff77", alpha=0.1, linewidth=0)
        ax.add_patch(bubble)

        ax.legend(loc="upper right", facecolor="#0d0d2a",
                  labelcolor="#aaaacc", fontsize=8, framealpha=0.7)

    def _render_3d(self):
        ax  = self.ax
        sat = self.satellite

        # ── trajectory ──
        if len(self.traj) > 1:
            traj = np.array(self.traj)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color="#2244aa", linewidth=0.8, alpha=0.6,
                    linestyle="--", label="trajectory")

        # ── debris ──
        for d in self.debris_list:
            dist = np.linalg.norm(sat.pos - d.pos)
            col  = "#ff4444" if dist < DANGER_RADIUS * 1.5 else "#cc2222"
            ax.scatter(*d.pos, color=col, s=60, depthshade=True,
                       edgecolors="#ff8888", linewidths=0.4)
            # velocity arrow
            ax.quiver(*d.pos, *(d.vel * 5),
                      color="#ff6644", linewidth=0.8, alpha=0.7,
                      arrow_length_ratio=0.4)
            # safety sphere (latitude-longitude lines)
            self._draw_sphere(ax, d.pos, DANGER_RADIUS, col, alpha=0.07)

        # ── satellite ──
        ax.scatter(*sat.pos, color="#33ff77", s=140, depthshade=False,
                   edgecolors="#aaffcc", linewidths=1.5,
                   zorder=10, label="satellite")
        # velocity arrow
        ax.quiver(*sat.pos, *(sat.vel * 3),
                  color="#55ffaa", linewidth=1.8, alpha=0.9,
                  arrow_length_ratio=0.4)

        ax.legend(loc="upper left", facecolor="#0d0d2a",
                  labelcolor="#aaaacc", fontsize=8, framealpha=0.7)

    @staticmethod
    def _draw_sphere(ax, center, radius, color, alpha=0.1):
        """Draw a wireframe sphere to represent the safety zone."""
        u = np.linspace(0, 2 * np.pi, 18)
        v = np.linspace(0, np.pi, 9)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(x, y, z, color=color, alpha=alpha,
                          linewidth=0.4, rstride=2, cstride=2)


# ─────────────────────────────────────────────
#  AGENTS
# ─────────────────────────────────────────────
class RandomAgent:
    """Baseline: uniformly random action selection."""

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def act(self, obs: np.ndarray) -> int:
        return random.randint(0, self.n_actions - 1)


class RuleBasedAgent:
    """
    Rule-based avoidance agent.

    Logic:
      1. If imminent collision → hard turn / pitch away
      2. If inside danger zone → gentle turn
      3. If future collision predicted → early turn
      4. Otherwise → maintain (small speed tweaks)
    """

    def __init__(self, env: SpaceTrafficEnv):
        self.env = env

    def act(self, obs: np.ndarray) -> int:
        sat   = self.env.satellite
        dlist = self.env.debris_list

        dists        = distances(sat.pos, dlist)
        min_dist     = dists.min()
        closest_idx  = int(np.argmin(dists))
        closest_pos  = dlist[closest_idx].pos

        # direction from satellite to closest debris
        direction = closest_pos - sat.pos
        dist_norm = np.linalg.norm(direction)
        if dist_norm < 1e-6:
            dist_norm = 1.0

        # ── 2D rule-based ──
        if self.env.mode == "2D":
            future_sat = predict_future_pos(sat.pos, sat.vel)
            future_dists = np.array([
                np.linalg.norm(future_sat - predict_future_pos(d.pos, d.vel))
                for d in dlist
            ])

            if min_dist <= COLLISION_RADIUS * 1.5:
                # emergency: turn away from threat
                cross = direction[0] * sat.vel[1] - direction[1] * sat.vel[0]
                return 1 if cross > 0 else 2

            elif min_dist <= DANGER_RADIUS:
                # danger zone: gentle turn + decelerate
                cross = direction[0] * sat.vel[1] - direction[1] * sat.vel[0]
                return 1 if cross > 0 else 2

            elif future_dists.min() <= DANGER_RADIUS:
                # predicted danger: proactive turn
                cross = direction[0] * sat.vel[1] - direction[1] * sat.vel[0]
                return 2 if cross > 0 else 1

            else:
                # safe: maintain or slight speed adjustment
                if sat.speed < SPEED_BASE:
                    return 3
                return 0

        # ── 3D rule-based ──
        else:
            future_sat = predict_future_pos(sat.pos, sat.vel)
            future_dists = np.array([
                np.linalg.norm(future_sat - predict_future_pos(d.pos, d.vel))
                for d in dlist
            ])

            if min_dist <= COLLISION_RADIUS * 1.5:
                # choose dominant avoidance axis
                dx, dy, dz = direction / dist_norm
                if abs(dz) > max(abs(dx), abs(dy)):
                    return 3 if dz > 0 else 4   # pitch away
                else:
                    return 1 if dy > 0 else 2   # yaw away

            elif min_dist <= DANGER_RADIUS:
                dx, dy, dz = direction / dist_norm
                if abs(dz) > max(abs(dx), abs(dy)):
                    return 3 if dz < 0 else 4
                else:
                    return 1 if dy < 0 else 2

            elif future_dists.min() <= DANGER_RADIUS:
                return 3   # proactive pitch up

            else:
                if sat.speed < SPEED_BASE:
                    return 5
                return 0


# ─────────────────────────────────────────────
#  STATS TRACKER
# ─────────────────────────────────────────────
class EpisodeStats:
    def __init__(self):
        self.rewards:    List[float] = []
        self.collisions: int         = 0
        self.survival:   List[int]   = []

    def record(self, total_reward: float, collided: bool, steps: int):
        self.rewards.append(total_reward)
        self.survival.append(steps)
        if collided:
            self.collisions += 1

    def summary(self, n_episodes: int):
        print("\n" + "=" * 52)
        print("  EPISODE SUMMARY")
        print("=" * 52)
        print(f"  Episodes run     : {n_episodes}")
        print(f"  Collisions       : {self.collisions}")
        print(f"  Avg total reward : {np.mean(self.rewards):.2f}")
        print(f"  Max total reward : {np.max(self.rewards):.2f}")
        print(f"  Avg survival     : {np.mean(self.survival):.1f} / {MAX_STEPS} steps")
        print("=" * 52 + "\n")

    def plot(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                                 facecolor="#0a0a1a")
        for ax in axes:
            ax.set_facecolor("#0a0a1a")
            ax.tick_params(colors="#aaaacc")
            ax.spines[["top","right","left","bottom"]].set_color("#222244")

        axes[0].plot(self.rewards, color="#33ff77", linewidth=1.5)
        axes[0].axhline(np.mean(self.rewards), color="#ffaa33",
                        linestyle="--", linewidth=1, label="mean")
        axes[0].set_title("Total Reward per Episode",
                           color="#aaaacc", fontsize=11)
        axes[0].set_xlabel("Episode", color="#aaaacc")
        axes[0].set_ylabel("Reward",  color="#aaaacc")
        axes[0].legend(facecolor="#0d0d2a", labelcolor="#aaaacc")

        axes[1].plot(self.survival, color="#4499ff", linewidth=1.5)
        axes[1].axhline(MAX_STEPS, color="#ff4444",
                        linestyle="--", linewidth=1, label=f"max={MAX_STEPS}")
        axes[1].set_title("Survival Steps per Episode",
                           color="#aaaacc", fontsize=11)
        axes[1].set_xlabel("Episode", color="#aaaacc")
        axes[1].set_ylabel("Steps",   color="#aaaacc")
        axes[1].legend(facecolor="#0d0d2a", labelcolor="#aaaacc")

        plt.suptitle("Space Traffic Control — Training Stats",
                     color="#ccccee", fontsize=13)
        plt.tight_layout()
        plt.show(block=True)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    # ╔══════════════════════════════════════╗
    # ║  CHANGE THIS to "2D" or "3D"        ║
    MODE = "2D"
    # ╚══════════════════════════════════════╝

    AGENT_TYPE  = "rule"    # "rule"   or  "random"
    N_EPISODES  = 5         # how many full episodes to run
    RENDER      = True      # set False for headless / faster training
    PAUSE_MS    = 0.03      # seconds between rendered frames

    print("=" * 52)
    print(f"  Space Traffic Control  |  mode={MODE}  |  agent={AGENT_TYPE}")
    print("=" * 52)

    stats = EpisodeStats()

    for ep in range(N_EPISODES):
        env   = SpaceTrafficEnv(mode=MODE,
                                render_mode="human" if RENDER else "none")
        agent = (RuleBasedAgent(env) if AGENT_TYPE == "rule"
                 else RandomAgent(env.n_actions))

        obs       = env.reset()
        collided  = False
        step      = 0

        print(f"\n  Episode {ep + 1}/{N_EPISODES}  — starting …")

        while True:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)

            if RENDER:
                env.render()
                time.sleep(PAUSE_MS)

            step += 1

            # console heartbeat every 50 steps
            if step % 50 == 0 or done:
                print(f"    step={step:4d}  reward={reward:+.2f}  "
                      f"min_dist={info['min_dist']:.2f}  "
                      f"risk={info['risk']:.2f}  "
                      f"total={env.total_reward:.1f}")

            if done:
                collided = info["min_dist"] <= COLLISION_RADIUS
                status   = "💥 COLLISION" if collided else "✅ survived"
                print(f"  → Episode {ep + 1} ended at step {step}  {status}")
                break

        stats.record(env.total_reward, collided, step)
        env.close()

    stats.summary(N_EPISODES)
    stats.plot()


if __name__ == "__main__":
    main()