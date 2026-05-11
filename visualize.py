"""
visualize.py — Generate all required performance charts
Usage:
    python visualize.py                        # uses qlearning_v1 results
    python visualize.py --run qlearning_v2
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim.energy_grid_env import EnergyGridEnv
from sim.q_agent import QLearningAgent

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="qlearning_v1")
args = parser.parse_args()
RUN  = args.run

rewards = np.load(f"results/rewards_log_{RUN}.npy")
carbon  = np.load(f"results/carbon_log_{RUN}.npy")
epsilon = np.load(f"results/epsilon_log_{RUN}.npy")

def moving_avg(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# ── Load trained agent ───────────────────────────────────────────────
agent = QLearningAgent()
agent.load_pkl(f"policies/policy_v2_explored_{RUN}.pkl")
agent.epsilon = 0.0
env = EnergyGridEnv()
obs, _ = env.reset()

# Run one full day — collect battery levels and energy sources
battery_levels = []
rl_rewards_day = []
for _ in range(24):
    battery_levels.append(env.battery_level)
    action = agent.choose_action(obs)
    obs, reward, done, _, _ = env.step(action)
    rl_rewards_day.append(reward)
    if done:
        break

rl_energy_log  = env.energy_log
rl_total_reward = sum(rl_rewards_day)
rl_carbon       = env.total_carbon

# ── Random baseline for comparison ──────────────────────────────────
obs, _ = env.reset()
random_battery = []
random_rewards_day = []
for _ in range(24):
    random_battery.append(env.battery_level)
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    random_rewards_day.append(reward)
    if done:
        break
random_total_reward = sum(random_rewards_day)
random_carbon       = env.total_carbon

# ════════════════════════════════════════════════════════════════════
# FIGURE 1 — Training Results (4 panels)
# ════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(14, 10))
fig1.suptitle(f"RL Energy Grid Scheduler — Training Results ({RUN})",
              fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

# Plot 1: Reward curve
ax1 = fig1.add_subplot(gs[0, 0])
ax1.plot(rewards, alpha=0.3, color='steelblue', linewidth=0.8, label='Per-episode')
ax1.plot(range(49, len(rewards)), moving_avg(rewards),
         color='steelblue', linewidth=2, label='50-ep avg')
ax1.set_title("Reward per Episode")
ax1.set_xlabel("Episode"); ax1.set_ylabel("Total Reward")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# Plot 2: Carbon reduction
ax2 = fig1.add_subplot(gs[0, 1])
ax2.plot(carbon, alpha=0.3, color='tomato', linewidth=0.8)
ax2.plot(range(49, len(carbon)), moving_avg(carbon), color='tomato', linewidth=2)
ax2.set_title("Carbon Units per Episode")
ax2.set_xlabel("Episode"); ax2.set_ylabel("Carbon units")
ax2.grid(True, alpha=0.3)

# Plot 3: Energy source pie chart
ax3 = fig1.add_subplot(gs[1, 0])
source_counts = Counter(rl_energy_log)
labels = list(source_counts.keys())
sizes  = list(source_counts.values())
colors = ['#4CAF50' if l == 'renewable'
          else '#2196F3' if l == 'battery'
          else '#F44336' for l in labels]
ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
        startangle=90, textprops={'fontsize': 10})
ax3.set_title("Energy Source Mix\n(Trained Agent — 1 Day)")

# Plot 4: Epsilon decay
ax4 = fig1.add_subplot(gs[1, 1])
ax4.plot(epsilon, color='darkorange', linewidth=2)
ax4.set_title("Exploration Rate (Epsilon) Decay")
ax4.set_xlabel("Episode"); ax4.set_ylabel("Epsilon")
ax4.set_ylim(0, 1.05); ax4.grid(True, alpha=0.3)

plt.savefig(f"results/training_results_{RUN}.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot 1 saved → results/training_results_{RUN}.png")

# ════════════════════════════════════════════════════════════════════
# FIGURE 2 — Battery Level Over Time (RL vs Random)
# ════════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Battery Level Over Time — RL vs Random Baseline",
              fontsize=13, fontweight='bold')

hours = list(range(1, len(battery_levels) + 1))

axes[0].plot(hours, battery_levels, color='#4CAF50', linewidth=2.5,
             marker='o', markersize=5, label='RL Agent')
axes[0].set_title("RL Agent — Battery Level per Hour")
axes[0].set_xlabel("Hour of Day"); axes[0].set_ylabel("Battery Level (units)")
axes[0].set_ylim(0, 11); axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(hours[:len(random_battery)], random_battery,
             color='#F44336', linewidth=2.5,
             marker='o', markersize=5, label='Random Agent')
axes[1].set_title("Random Agent — Battery Level per Hour")
axes[1].set_xlabel("Hour of Day"); axes[1].set_ylabel("Battery Level (units)")
axes[1].set_ylim(0, 11); axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig(f"results/battery_over_time_{RUN}.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot 2 saved → results/battery_over_time_{RUN}.png")

# ════════════════════════════════════════════════════════════════════
# FIGURE 3 — RL vs Random Comparison (Bar Chart)
# ════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle("RL Agent vs Random Baseline — Test Day Comparison",
              fontsize=13, fontweight='bold')

# Bar 1: Total Reward
categories = ['Random Policy', 'RL Policy']
reward_vals = [random_total_reward, rl_total_reward]
bar_colors  = ['#F44336', '#4CAF50']
bars1 = axes3[0].bar(categories, reward_vals, color=bar_colors, width=0.4, edgecolor='white')
axes3[0].set_title("Test Day Total Reward")
axes3[0].set_ylabel("Total Reward")
axes3[0].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars1, reward_vals):
    axes3[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                  str(round(val)), ha='center', fontweight='bold')

# Bar 2: Carbon Units
carbon_vals = [random_carbon, rl_carbon]
bars2 = axes3[1].bar(categories, carbon_vals, color=bar_colors, width=0.4, edgecolor='white')
axes3[1].set_title("Test Day Carbon Units")
axes3[1].set_ylabel("Carbon Units")
axes3[1].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars2, carbon_vals):
    axes3[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                  str(round(val)), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f"results/rl_vs_random_{RUN}.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot 3 saved → results/rl_vs_random_{RUN}.png")

# ════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  ALL PLOTS GENERATED SUCCESSFULLY")
print("=" * 55)
print(f"  results/training_results_{RUN}.png")
print(f"  results/battery_over_time_{RUN}.png")
print(f"  results/rl_vs_random_{RUN}.png")
print("=" * 55)
print(f"\n  RL  Total Reward : {rl_total_reward}  | Carbon: {rl_carbon}")
print(f"  RND Total Reward : {random_total_reward} | Carbon: {random_carbon}")
carbon_reduction = round((random_carbon - rl_carbon) / max(random_carbon, 1) * 100)
print(f"\n  Carbon Reduction : {carbon_reduction}% — supports SDG 7, 9, 13")
print("=" * 55)