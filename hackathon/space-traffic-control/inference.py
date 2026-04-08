# -*- coding: utf-8 -*-
"""
inference.py — Space Traffic Control
=====================================
OpenENV-compatible inference script for the Meta AI Hackathon.

Wiring:
  - SpaceTrafficEnv  = the openenv environment
  - HF Inference Router (Qwen2.5-72B) = the LLM agent brain
  - RuleBasedAgent   = fallback + action executor
  - LLM reads state → decides direction → RL executes

STDOUT FORMAT (mandatory):
  [START] task=<task> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<null|msg>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

ENVIRONMENT VARIABLES:
  API_BASE_URL   — HF Inference Router endpoint
  MODEL_NAME     — LLM model identifier
  HF_TOKEN       — Hugging Face token
  SPACE_MODE     — "2D" or "3D" (default: "2D")
"""

import asyncio
import os
import textwrap
from typing import List, Optional

import numpy as np
from openai import OpenAI

# ── import our simulation environment ──
from space_traffic_control import (
    SpaceTrafficEnv,
    RuleBasedAgent,
    distances,
    risk_level,
    DANGER_RADIUS,
    COLLISION_RADIUS,
    MAX_STEPS,
)

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
SPACE_MODE   = os.getenv("SPACE_MODE",   "2D")   # "2D" or "3D"
TASK_NAME    = "satellite-collision-avoidance"
BENCHMARK    = "space-traffic-control"

EPISODE_STEPS         = 50     # steps per inference episode
TEMPERATURE           = 0.3    # lower = more focused decisions
MAX_TOKENS            = 120
SUCCESS_SCORE_THRESHOLD = 0.4  # score >= 0.4 = success

# ─────────────────────────────────────────────
#  ACTION MAPS
# ─────────────────────────────────────────────
ACTION_MAP_2D = {
    "maintain":    0,
    "turn left":   1,
    "turn right":  2,
    "accelerate":  3,
    "decelerate":  4,
}

ACTION_MAP_3D = {
    "maintain":    0,
    "yaw left":    1,
    "yaw right":   2,
    "pitch up":    3,
    "pitch down":  4,
    "accelerate":  5,
    "decelerate":  6,
}

# ─────────────────────────────────────────────
#  LOGGING (mandatory stdout format)
# ─────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int,
            score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────
#  STATE → PROMPT BUILDER
# ─────────────────────────────────────────────
def build_system_prompt(mode: str) -> str:
    if mode == "2D":
        actions = (
            "maintain, turn left, turn right, accelerate, decelerate"
        )
    else:
        actions = (
            "maintain, yaw left, yaw right, "
            "pitch up, pitch down, accelerate, decelerate"
        )

    return textwrap.dedent(f"""
        You are the navigation AI for a satellite in {mode} space.
        Your job is to avoid space debris at all costs.

        Think of it like driving a car:
          - Satellite = your vehicle
          - Debris    = obstacles on the road
          - Collision = game over (-10 reward)
          - Safe step = +1 reward

        STRICT RULES — follow these exactly:

        1. EMERGENCY (distance < 5, risk > 0.8):
           → Immediately turn left or turn right
           → NEVER accelerate when debris is close
           → This is your top priority

        2. DANGER ZONE (distance 5-8, risk 0.5-0.8):
           → Decelerate first, then turn away
           → Do NOT maintain — you must act

        3. CAUTION (distance 8-15, risk 0.2-0.5):
           → Gentle turn left or turn right
           → You can maintain speed here

        4. SAFE (distance > 15, risk < 0.2):
           → maintain or accelerate carefully
           → Never accelerate above safe speed

        IMPORTANT:
          - Collision = instant failure, avoid at ALL costs
          - Decelerating when close to debris saves lives
          - Accelerating into debris = collision
          - Reply with ONLY the action name, nothing else
          - One word or two words maximum

        Available actions: {actions}
    """).strip()
# ─────────────────────────────────────────────
#  STATE → USER PROMPT BUILDER
# ─────────────────────────────────────────────
def build_user_prompt(env: SpaceTrafficEnv, step: int,
                      last_reward: float, history: List[str]) -> str:
    sat    = env.satellite
    dlist  = env.debris_list
    dists  = distances(sat.pos, dlist)
    risk   = risk_level(sat.pos, dlist)
    min_d  = float(dists.min())
    top3   = np.argsort(dists)[:3]

    # describe nearest debris
    debris_lines = []
    for i, idx in enumerate(top3):
        d = dlist[idx]
        if env.mode == "2D":
            debris_lines.append(
                f"  Debris {i+1}: pos=({d.pos[0]:.1f},{d.pos[1]:.1f}) "
                f"dist={dists[idx]:.1f}"
            )
        else:
            debris_lines.append(
                f"  Debris {i+1}: pos=({d.pos[0]:.1f},{d.pos[1]:.1f},"
                f"{d.pos[2]:.1f}) dist={dists[idx]:.1f}"
            )
    debris_str = "\n".join(debris_lines)

    # danger assessment — updated with clearer instructions
    if min_d <= COLLISION_RADIUS:
       danger = "[CRITICAL] COLLISION IMMINENT - turn left or right NOW, do NOT accelerate"
    elif min_d <= DANGER_RADIUS:
        danger = "[WARNING]  DANGER ZONE - decelerate then turn away immediately"
    elif min_d <= DANGER_RADIUS * 1.5:
        danger = "[CAUTION]  debris nearby, turn gently, do not accelerate"
    else:
       danger = "[CLEAR]    path is safe, maintain or gentle accelerate"
    # recent history
    history_block = "\n".join(history[-3:]) if history else "None"

    if env.mode == "2D":
        pos_str = f"({sat.pos[0]:.1f}, {sat.pos[1]:.1f})"
        vel_str = f"({sat.vel[0]:.2f}, {sat.vel[1]:.2f})"
    else:
        pos_str = f"({sat.pos[0]:.1f}, {sat.pos[1]:.1f}, {sat.pos[2]:.1f})"
        vel_str = f"({sat.vel[0]:.2f}, {sat.vel[1]:.2f}, {sat.vel[2]:.2f})"

    return textwrap.dedent(f"""
        Step: {step}
        Satellite position : {pos_str}
        Satellite velocity : {vel_str}
        Speed              : {sat.speed:.2f}
        Closest debris dist: {min_d:.2f}
        Risk level         : {risk:.2f}  (0=safe, 1=collision)
        Status             : {danger}

        3 Nearest debris:
        {debris_str}

        Last reward : {last_reward:.2f}
        Recent actions:
        {history_block}

        What is your next action?
    """).strip()
# ─────────────────────────────────────────────
#  LLM → ACTION PARSER
# ─────────────────────────────────────────────
def parse_llm_action(text: str, mode: str) -> int:
    """
    Convert LLM text response to numeric action.
    Falls back to 0 (maintain) if unrecognised.
    """
    text = text.lower().strip()
    action_map = ACTION_MAP_2D if mode == "2D" else ACTION_MAP_3D

    # exact match first
    if text in action_map:
        return action_map[text]

    # partial match
    for key, val in action_map.items():
        if key in text:
            return val

    # no match → maintain
    return 0


# ─────────────────────────────────────────────
#  LLM CALL
# ─────────────────────────────────────────────
def get_llm_action(client: OpenAI, env: SpaceTrafficEnv,
                   step: int, last_reward: float,
                   history: List[str]) -> tuple[str, int]:
    """
    Ask the LLM what action to take.
    Returns (action_text, action_int).
    Falls back to rule-based agent on failure.
    """
    system = build_system_prompt(env.mode)
    user   = build_user_prompt(env, step, last_reward, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("Empty response from LLM")
        action_int = parse_llm_action(text, env.mode)
        return text, action_int

    except Exception as exc:
        # fallback to rule-based agent
        print(f"[DEBUG] LLM call failed: {exc} — using rule-based fallback",
              flush=True)
        fallback   = RuleBasedAgent(env)
        obs        = env._get_obs()
        action_int = fallback.act(obs)
        action_map = ACTION_MAP_2D if env.mode == "2D" else ACTION_MAP_3D
        text       = {v: k for k, v in action_map.items()}.get(action_int, "maintain")
        return text, action_int


# ─────────────────────────────────────────────
#  SCORING
# ─────────────────────────────────────────────
def compute_score(rewards: List[float], survived: bool,
                  steps: int) -> float:
    """
    Normalised score in [0, 1]:
      - Base: total positive reward / max possible reward
      - Bonus: +0.2 if survived without collision
      - Penalty: -0.1 per collision
    """
    if not rewards:
        return 0.0
    max_possible = EPISODE_STEPS * (1.0 + 2.0 + 5.0)  # safe+smooth+avoidance
    base  = sum(r for r in rewards if r > 0) / max_possible
    bonus = 0.2 if survived else -0.1
    score = base + bonus
    return float(min(max(score, 0.0), 1.0))


# ─────────────────────────────────────────────
#  MAIN ASYNC LOOP
# ─────────────────────────────────────────────
async def main() -> None:
    # ── init ──
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env    = SpaceTrafficEnv(mode=SPACE_MODE, render_mode="none")

    history:     List[str]   = []
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    survived:    bool        = True
    last_error:  Optional[str] = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── reset environment ──
        env.reset()
        last_reward = 0.0

        for step in range(1, EPISODE_STEPS + 1):

            # LLM decides → RL executes
            action_text, action_int = get_llm_action(
                client, env, step, last_reward, history
            )

            # step the environment
            try:
                obs, reward, done, info = env.step(action_int)
                last_error = None
            except Exception as e:
                last_error = str(e)
                reward     = 0.0
                done       = True

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            # check collision
            if info.get("min_dist", 99) <= COLLISION_RADIUS:
                survived   = False
                last_error = "collision"

            log_step(
                step=step,
                action=action_text,
                reward=reward,
                done=done,
                error=last_error,
            )

            history.append(
                f"Step {step}: {action_text!r} → reward {reward:+.2f} "
                f"dist={info.get('min_dist', 0):.1f} "
                f"risk={info.get('risk', 0):.2f}"
            )

            if done:
                break

        # ── compute final score ──
        score   = compute_score(rewards, survived, steps_taken)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        last_error = str(e)
        print(f"[DEBUG] Episode error: {e}", flush=True)

    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())