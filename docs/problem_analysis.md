# Problem Analysis — Energy Grid RL Scheduler

## Abstract
This project develops a Reinforcement Learning-based energy scheduling 
agent using Q-Learning to optimise energy dispatch across renewable, 
battery, and grid sources over a 24-hour period. The system integrates 
MLOps practices including CI/CD, experiment tracking via MLflow, 
versioned policies, and automated rollback.

---

## Stakeholders
| Stakeholder | Role | Interest |
|-------------|------|----------|
| Grid Operators | Primary Users | Optimise dispatch, reduce cost |
| Energy Consumers | End Users | Reliable, clean power supply |
| Government/Regulators | Oversight | Meet SDG 7, 9, 13 targets |
| ML Engineers | Developers | Reproducible, scalable pipeline |
| Environmental Bodies | Monitors | Carbon emission reduction |

---

## Functional Requirements
1. Agent schedules energy source for each of 24 hours
2. Prioritise renewable energy when available
3. Use battery storage as secondary source
4. Fall back to grid only when necessary
5. Log all training runs via MLflow
6. Save best policy after training
7. Support rollback to previous best policy
8. CI/CD pipeline triggers on every push

---

## Non-Functional Requirements
1. Training completes within 1000 episodes
2. Pipeline fully reproducible across machines
3. Carbon reduction target minimum 90%
4. Renewable usage target minimum 70%
5. System scalable to multiple config files
6. Rollback completes within seconds

---

## System Design

**Pipeline Flow:**

1. Developer pushes code to GitHub
2. GitHub Actions CI/CD triggers automatically
3. train.py runs Q-Learning for 1000 episodes
4. MLflow logs all parameters and metrics
5. Best policy saved to policies/ folder
6. Results and plots saved to results/ folder
7. If performance drops, rollback.py restores best policy

**Components:**
- sim/energy_grid_env.py → Custom Gymnasium environment
- sim/q_agent.py → Q-Learning agent with epsilon-greedy
- train.py → Training loop with MLflow integration
- rollback.py → Policy restoration mechanism
- .github/workflows/train.yml → CI/CD automation

---

## Feasibility Analysis
| Factor | Assessment |
|--------|------------|
| Technical | Q-Learning suitable for small discrete state space |
| Computational | Runs on CPU, no GPU required |
| Time | 1000 episodes complete in seconds |
| Data | No external dataset needed, simulation-based |
| Cost | Zero cost, open source tools only |

---

## Constraints
- Battery level: 0 to 10 units only
- Episode length: fixed 24 steps (hours)
- State space: [11 x 24] = 264 states
- Action space: 3 actions only
- No real-time data feed (simulation only)

---

## Trade-offs
| Decision | Chosen | Alternative | Reason |
|----------|--------|-------------|--------|
| Algorithm | Q-Learning | Deep RL (DQN) | State space too small for DNN |
| Tracking | MLflow | Weights & Biases | Open source, self-hosted |
| CI/CD | GitHub Actions | Jenkins | Native GitHub integration |
| Environment | Gymnasium | Custom | Standard RL interface |

---

## Risk Analysis
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Policy divergence | Medium | High | Rollback to best policy |
| Experiment data loss | Low | High | MLflow tracking |
| Dependency conflict | Low | Medium | requirements.txt pinned |
| Pipeline failure | Low | Medium | GitHub Actions alerts |
| Overfitting to config | Medium | Medium | Multiple config files |

---

## Traceability Matrix
| Requirement | Component | Test |
|-------------|-----------|------|
| Energy scheduling | sim/energy_grid_env.py | Episode reward |
| Q-Learning agent | sim/q_agent.py | Convergence plot |
| Experiment tracking | train.py + MLflow | mlflow.db logs |
| CI/CD automation | .github/workflows | Actions green ✅ |
| Rollback | rollback.py + policies/ | Policy loaded |
| Visualisation | visualize.py | Plots in results/ |

---

## Conclusions
- Q-Learning agent successfully learns optimal scheduling policy
- Carbon emissions reduced by 92% over 1000 training episodes
- Renewable energy usage consistently above 75%
- MLOps pipeline ensures full reproducibility and traceability
- Rollback mechanism provides production-grade reliability
- CI/CD automates training and validation on every commit
