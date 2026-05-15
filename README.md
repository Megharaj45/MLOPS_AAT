# Energy Grid RL Scheduler
**Reinforcement Learning + MLOps** | 24AM6PCREL | B.M.S College of Engineering

---

## Problem Statement
Train a Q-Learning agent to schedule energy sources — renewable (solar/wind), 
battery storage, or grid power — for each hour of a 24-hour day, minimising 
carbon emissions and maximising clean energy usage.

---

## SDGs Addressed
| SDG | Title | How |
|-----|-------|-----|
| SDG 7  | Affordable and Clean Energy | 75%+ renewable usage achieved |
| SDG 9  | Industry, Innovation & Infrastructure | AI-driven smart grid scheduling |
| SDG 13 | Climate Action | Carbon reduced by 92% over 1000 episodes |

**SDG Impact:**
> "Reducing carbon units by 92% supports SDG 13. Achieving 75% renewable 
> usage supports SDG 7. Building AI-driven smart grid supports SDG 9."

---

## Algorithm Choice
**Q-Learning** was selected because:
- State space (battery: 0-10, hour: 0-23) is discrete and small
- No prior model of environment needed (model-free)
- Q-table shape [11 x 24 x 3] is sufficient
- Bellman updates converge reliably for episodic 24-step tasks

---

## State, Action, Reward
| Component | Definition |
|-----------|-----------|
| **State** | (battery_level, hour_of_day) |
| **Action** | 0=Renewable, 1=Battery, 2=Grid |
| **Reward** | +10 renewable, +3 battery, -5 grid |

---

## Project Structure

- .github/workflows/train.yml  → CI/CD pipeline
- sim/energy_grid_env.py       → Custom Gymnasium environment
- sim/q_agent.py               → Q-Learning agent
- configs/qlearning_v1.yaml    → Experiment 1 config
- configs/qlearning_v2.yaml    → Experiment 2 config
- experiments/                 → Auto-generated CSV logs
- results/                     → Plots and training logs
- policies/                    → Saved policy snapshots
- docs/problem_analysis.md     → Full problem analysis
- train.py                     → Main training + MLflow
- visualize.py                 → Plot generation
- rollback.py                  → Rollback to best policy
- requirements.txt             → Dependencies
- README.md                    → Documentation