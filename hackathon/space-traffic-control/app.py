"""
Space Traffic Control — Hugging Face Spaces Web Interface
Connects the RL simulation to a Gradio UI with HF Inference Router
"""

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for server
import matplotlib.pyplot as plt
import io
from PIL import Image

# import our simulation
from space_traffic_control import SpaceTrafficEnv, RuleBasedAgent, RandomAgent

def run_simulation(mode, agent_type, n_steps):
    """Run the simulation and return a plot + stats."""

    env   = SpaceTrafficEnv(mode=mode, render_mode="none")
    agent = (RuleBasedAgent(env) if agent_type == "Rule-Based"
             else RandomAgent(env.n_actions))

    obs      = env.reset()
    rewards  = []
    min_dists = []

    for _ in range(int(n_steps)):
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        min_dists.append(info["min_dist"])
        if done:
            break

    # ── build plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             facecolor="#0a0a1a")
    for ax in axes:
        ax.set_facecolor("#0a0a1a")
        ax.tick_params(colors="#aaaacc")
        for spine in ax.spines.values():
            spine.set_color("#222244")

    # trajectory
    if len(env.traj) > 1:
        traj = np.array(env.traj)
        if mode == "2D":
            axes[0].plot(traj[:, 0], traj[:, 1],
                        color="#33ff77", linewidth=1.5, label="trajectory")
            for d in env.debris_list:
                circle = plt.Circle(d.pos, 8, color="#ff4444",
                                   alpha=0.2, linewidth=0)
                axes[0].add_patch(circle)
                axes[0].scatter(*d.pos, color="#ff4444", s=50)
            axes[0].scatter(*env.satellite.pos,
                           color="#33ff77", s=120, zorder=10,
                           label="satellite")
            axes[0].set_xlim(0, 100)
            axes[0].set_ylim(0, 100)
            axes[0].set_title("2D Trajectory", color="#aaaacc")
            axes[0].legend(facecolor="#0d0d2a", labelcolor="#aaaacc")
        else:
            axes[0].plot(traj[:, 0], traj[:, 1],
                        color="#33ff77", linewidth=1.5)
            axes[0].set_title("3D Trajectory (X-Y plane)",
                             color="#aaaacc")

    # reward chart
    axes[1].plot(rewards, color="#4499ff", linewidth=1.5,
                label="step reward")
    axes[1].plot(np.cumsum(rewards) / (np.arange(len(rewards)) + 1),
                color="#ffaa33", linewidth=1.5, linestyle="--",
                label="avg reward")
    axes[1].axhline(0, color="#555577", linewidth=0.8)
    axes[1].set_title("Rewards over Time", color="#aaaacc")
    axes[1].set_xlabel("Step", color="#aaaacc")
    axes[1].legend(facecolor="#0d0d2a", labelcolor="#aaaacc")

    plt.tight_layout()

    # convert to image for Gradio
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor="#0a0a1a",
                bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    # stats summary
    collided  = min_dists[-1] <= 3.0
    status    = "💥 COLLISION" if collided else "✅ Survived"
    stats_txt = (
        f"**Status:** {status}\n\n"
        f"**Steps completed:** {env.step_count}\n\n"
        f"**Total reward:** {env.total_reward:.2f}\n\n"
        f"**Minimum distance to debris:** {min(min_dists):.2f}\n\n"
        f"**Final risk level:** {info['risk']:.2f}"
    )

    return img, stats_txt


# ── Gradio UI ──
with gr.Blocks(
    theme=gr.themes.Base(),
    css="""
        body { background: #0a0a1a; }
        .gradio-container { background: #0a0a1a; color: #aaaacc; }
        h1 { color: #33ff77 !important; }
        h3 { color: #4499ff !important; }
    """
) as demo:

    gr.Markdown("# 🛸 Space Traffic Control — Satellite Collision Avoidance RL")
    gr.Markdown("### Integrated 2D + 3D simulation using Reinforcement Learning")

    with gr.Row():
        with gr.Column(scale=1):
            mode_input    = gr.Radio(["2D", "3D"],
                                     label="Simulation Mode",
                                     value="2D")
            agent_input   = gr.Radio(["Rule-Based", "Random"],
                                     label="Agent Type",
                                     value="Rule-Based")
            steps_input   = gr.Slider(50, 500, value=200, step=50,
                                      label="Number of Steps")
            run_btn       = gr.Button("🚀 Run Simulation",
                                      variant="primary")

        with gr.Column(scale=2):
            plot_output   = gr.Image(label="Simulation Output")
            stats_output  = gr.Markdown(label="Stats")

    run_btn.click(
        fn=run_simulation,
        inputs=[mode_input, agent_input, steps_input],
        outputs=[plot_output, stats_output]
    )

    gr.Markdown("""
    ---
    **How it works:**
    - 🛸 **Satellite** = vehicle navigating through space
    - ☄️ **Debris** = potholes / traffic obstacles
    - 🟢 **Safe timestep** = +1 reward
    - 💥 **Collision** = -10 reward
    - 🔄 **Rule-Based agent** uses predictive avoidance
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)