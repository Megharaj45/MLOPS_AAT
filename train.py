"""
train.py — Main training script for Energy Grid RL Scheduler
Usage:
    python train.py                          # uses configs/qlearning_v1.yaml
    python train.py --config configs/qlearning_v2.yaml
"""

import argparse
import csv
import os
import sys
import time

import mlflow
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim.energy_grid_env import EnergyGridEnv
from sim.q_agent import QLearningAgent

# ── Argument parsing ─────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train Energy Grid RL Agent")
parser.add_argument("--config", default="configs/qlearning_v1.yaml",
                    help="Path to YAML config file")
args = parser.parse_args()

# ── Load config ──────────────────────────────────────────────────────
with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

RUN_ID     = cfg["run_id"]
N_EPISODES = cfg["n_episodes"]
MAX_STEPS  = cfg["max_steps"]
ALPHA      = cfg["agent"]["alpha"]
GAMMA      = cfg["agent"]["gamma"]
EPSILON    = cfg["agent"]["epsilon"]
EPS_MIN    = cfg["agent"]["epsilon_min"]
EPS_DECAY  = cfg["agent"]["epsilon_decay"]

print("=" * 60)
print(f"  Energy Grid RL Scheduler — {RUN_ID}")
print(f"  Config : {args.config}")
print(f"  alpha={ALPHA}  gamma={GAMMA}  epsilon_decay={EPS_DECAY}")
print("=" * 60)

# ── Setup ────────────────────────────────────────────────────────────
os.makedirs("results",     exist_ok=True)
os.makedirs("policies",    exist_ok=True)
os.makedirs("experiments", exist_ok=True)

env   = EnergyGridEnv()
agent = QLearningAgent(
    alpha=ALPHA, gamma=GAMMA,
    epsilon=EPSILON, epsilon_min=EPS_MIN, epsilon_decay=EPS_DECAY
)

rewards_log = []
carbon_log  = []
epsilon_log = []
start_time  = time.time()

# ── MLflow Experiment Setup ──────────────────────────────────────────
mlflow.set_experiment("EnergyGrid_RL_Scheduler")

with mlflow.start_run(run_name=RUN_ID):

    # Log all hyperparameters
    mlflow.log_param("run_id",        RUN_ID)
    mlflow.log_param("config",        args.config)
    mlflow.log_param("n_episodes",    N_EPISODES)
    mlflow.log_param("alpha",         ALPHA)
    mlflow.log_param("gamma",         GAMMA)
    mlflow.log_param("epsilon",       EPSILON)
    mlflow.log_param("epsilon_min",   EPS_MIN)
    mlflow.log_param("epsilon_decay", EPS_DECAY)
    mlflow.log_param("sdg",           "SDG7, SDG9, SDG13")

    # ── Training loop ────────────────────────────────────────────────
    print("\nTraining started...\n")

    for episode in range(N_EPISODES):
        obs, _ = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            action                       = agent.choose_action(obs)
            next_obs, reward, done, _, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, done)
            obs          = next_obs
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        rewards_log.append(total_reward)
        carbon_log.append(env.total_carbon)
        epsilon_log.append(agent.epsilon)

        # Log metrics to MLflow every episode
        mlflow.log_metric("reward",  total_reward,     step=episode)
        mlflow.log_metric("carbon",  env.total_carbon, step=episode)
        mlflow.log_metric("epsilon", agent.epsilon,    step=episode)

        # Save policy_v1 at episode 500
        if (episode + 1) == 500:
            agent.save_pkl(f"policies/policy_v1_{RUN_ID}.pkl")
            np.save(f"results/rewards_log_mid_{RUN_ID}.npy", np.array(rewards_log))
            mlflow.log_artifact(f"policies/policy_v1_{RUN_ID}.pkl")
            print(f"  [Snapshot] policy_v1 saved at episode 500")

        if (episode + 1) % 100 == 0:
            avg_r = np.mean(rewards_log[-100:])
            avg_c = np.mean(carbon_log[-100:])
            print(f"  Episode {episode+1:4d} | Avg Reward: {avg_r:6.1f} | "
                  f"Avg Carbon: {avg_c:4.1f} | Epsilon: {agent.epsilon:.3f}")

    elapsed = round(time.time() - start_time, 2)

    # ── Save policy_v2 (fully trained) ───────────────────────────────
    agent.save_pkl(f"policies/policy_v2_explored_{RUN_ID}.pkl")
    agent.save_npy(f"results/q_table_{RUN_ID}.npy")
    np.save(f"results/rewards_log_{RUN_ID}.npy",  np.array(rewards_log))
    np.save(f"results/carbon_log_{RUN_ID}.npy",   np.array(carbon_log))
    np.save(f"results/epsilon_log_{RUN_ID}.npy",  np.array(epsilon_log))

    print(f"\nTraining complete in {elapsed}s")

    # ── Test RL agent (greedy) ────────────────────────────────────────
    print("\n--- Testing Trained RL Agent (1 day) ---")
    agent.epsilon = 0.0
    obs, _        = env.reset()
    rl_reward     = 0
    for _ in range(MAX_STEPS):
        action = agent.choose_action(obs)
        obs, reward, done, _, _ = env.step(action)
        rl_reward += reward
        env.render()
        if done:
            break

    print(f"\nRL Agent   → Total Reward : {rl_reward} | Carbon: {env.total_carbon}")
    print(f"RL Sources → {env.energy_log}")

    # ── Random baseline ───────────────────────────────────────────────
    print("\n--- Testing Random Baseline (1 day) ---")
    obs, _        = env.reset()
    random_reward = 0
    for _ in range(MAX_STEPS):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        random_reward += reward
        if done:
            break

    random_carbon = env.total_carbon
    print(f"Random Agent → Total Reward : {random_reward} | Carbon: {random_carbon}")

    # ── Log final metrics to MLflow ───────────────────────────────────
    mlflow.log_metric("avg_reward_last100", round(float(np.mean(rewards_log[-100:])), 2))
    mlflow.log_metric("avg_carbon_last100", round(float(np.mean(carbon_log[-100:])),  2))
    mlflow.log_metric("rl_test_reward",     rl_reward)
    mlflow.log_metric("rl_test_carbon",     env.total_carbon)
    mlflow.log_metric("random_test_reward", random_reward)
    mlflow.log_metric("random_test_carbon", random_carbon)
    mlflow.log_metric("training_time_s",    elapsed)

    # Log artifacts to MLflow
    mlflow.log_artifact(f"policies/policy_v2_explored_{RUN_ID}.pkl")
    mlflow.log_artifact(args.config)

    # ── Experiment tracking CSV ───────────────────────────────────────
    csv_path = f"experiments/results_{RUN_ID}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_id", "config", "episodes", "avg_reward_last100",
            "avg_carbon_last100", "rl_test_reward", "rl_test_carbon",
            "random_test_reward", "random_test_carbon",
            "epsilon_final", "learning_rate", "gamma",
            "epsilon_decay", "training_time_s"
        ])
        writer.writeheader()
        writer.writerow({
            "run_id":              RUN_ID,
            "config":              args.config,
            "episodes":            N_EPISODES,
            "avg_reward_last100":  round(float(np.mean(rewards_log[-100:])), 2),
            "avg_carbon_last100":  round(float(np.mean(carbon_log[-100:])),  2),
            "rl_test_reward":      rl_reward,
            "rl_test_carbon":      env.total_carbon,
            "random_test_reward":  random_reward,
            "random_test_carbon":  random_carbon,
            "epsilon_final":       round(agent.epsilon, 4),
            "learning_rate":       ALPHA,
            "gamma":               GAMMA,
            "epsilon_decay":       EPS_DECAY,
            "training_time_s":     elapsed
        })

    mlflow.log_artifact(csv_path)
    print(f"\nExperiment log saved → {csv_path}")

    # ── Comparison table ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  BASELINE vs RL COMPARISON")
    print("=" * 50)
    print(f"  {'Metric':<28} {'Random':>8} {'RL':>8}")
    print(f"  {'-'*44}")
    print(f"  {'Test Day Total Reward':<28} {random_reward:>8} {rl_reward:>8}")
    print(f"  {'Test Day Carbon Units':<28} {random_carbon:>8} {env.total_carbon:>8}")
    print(f"  {'Avg Reward (last 100 eps)':<28} {'N/A':>8} "
          f"{round(float(np.mean(rewards_log[-100:])),1):>8}")
    print("=" * 50)
    print(f"\nSDGs Addressed: SDG 7, SDG 9, SDG 13")
    print(f"\nMLflow UI → run: mlflow ui")
    print(f"Open browser → http://127.0.0.1:5000")