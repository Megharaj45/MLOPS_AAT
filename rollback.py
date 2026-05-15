"""
rollback.py — Rollback to best policy version
Usage:
    python rollback.py
Compares policy_v1 vs policy_v2 and rolls back to v1 if v2 is worse
"""

import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim.energy_grid_env import EnergyGridEnv
from sim.q_agent import QLearningAgent

RUN = "qlearning_v1"

def evaluate_policy(policy_path, n_episodes=10):
    """Evaluate a policy over n episodes and return average reward."""
    agent = QLearningAgent()
    agent.load_pkl(policy_path)
    agent.epsilon = 0.0

    env = EnergyGridEnv()
    rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(24):
            action = agent.choose_action(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)

    return np.mean(rewards)

# ── Evaluate both policies ───────────────────────────────────────────
v1_path = f"policies/policy_v1_{RUN}.pkl"
v2_path = f"policies/policy_v2_explored_{RUN}.pkl"

print("=" * 50)
print("  ROLLBACK EVALUATION")
print("=" * 50)

v1_score = evaluate_policy(v1_path)
v2_score = evaluate_policy(v2_path)

print(f"  policy_v1 avg reward : {v1_score:.1f}")
print(f"  policy_v2 avg reward : {v2_score:.1f}")

# ── Rollback decision ────────────────────────────────────────────────
if v2_score >= v1_score:
    print(f"\n  policy_v2 is BETTER or EQUAL → keeping v2")
    print(f"  Active policy: policy_v2_explored_{RUN}.pkl")
else:
    print(f"\n  policy_v2 is WORSE → rolling back to v1!")
    # Copy v1 as active policy
    with open(v1_path, "rb") as f:
        best_policy = pickle.load(f)
    with open(f"policies/active_policy_{RUN}.pkl", "wb") as f:
        pickle.dump(best_policy, f)
    print(f"  Rolled back to: policy_v1_{RUN}.pkl")
    print(f"  Active policy saved: policies/active_policy_{RUN}.pkl")

print("=" * 50)