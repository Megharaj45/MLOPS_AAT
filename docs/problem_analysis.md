# Problem Analysis & Design Document
## Energy Grid RL Scheduler

---

## 1. Stakeholders
| Stakeholder | Role | Interest |
|---|---|---|
| Energy Grid Operators | Primary User | Minimize carbon, maximize renewable usage |
| Government/Policy Makers | Regulator | SDG compliance, carbon reduction targets |
| End Consumers | Beneficiary | Reliable, affordable clean energy |
| ML Engineers | Developer | Model performance, reproducibility |
| Environmental Organizations | Monitor | Carbon emission reduction |

---

## 2. Use Cases
| Use Case | Description |
|---|---|
| UC1 | Schedule hourly energy source for 24-hour period |
| UC2 | Monitor battery charge/discharge cycles |
| UC3 | Track and minimize carbon emissions per episode |
| UC4 | Compare RL policy vs random baseline |
| UC5 | Reproduce experiments using config files |

---

## 3. Functional Requirements
| ID | Requirement |
|---|---|
| FR1 | System shall select energy source every hour |
| FR2 | System shall track battery level (0-10 units) |
| FR3 | System shall simulate solar/wind availability |
| FR4 | System shall assign rewards: +10 renewable, +3 battery, -5 grid |
| FR5 | System shall log carbon units per episode |
| FR6 | System shall save policy snapshots at episode 500 and 1000 |
| FR7 | System shall track experiments using MLflow |
| FR8 | System shall generate performance visualizations |

---

## 4. Non-Functional Requirements
| ID | Requirement |
|---|---|
| NFR1 | Training must complete within 60 seconds for 1000 episodes |
| NFR2 | System must be reproducible using YAML config files |
| NFR3 | Code must be modular and well-documented |
| NFR4 | MLflow must log all parameters and metrics |
| NFR5 | System must work on Python 3.8+ |
| NFR6 | Results must be version controlled using Git |

---

## 5. Feasibility & Constraints
### Technical Feasibility
- Q-Learning is proven for discrete state-action spaces
- State space [11 x 24 x 3] is small enough for tabular Q-table
- Python + Gymnasium + MLflow are industry-standard tools

### Constraints
- Battery limited to 0-10 units
- Episode length fixed at 24 hours (one day)
- Renewable availability depends on time of day (solar/wind simulation)
- No real-time weather data integration in current version

---

## 6. Trade-offs & Risks
| Trade-off | Decision | Reason |
|---|---|---|
| Tabular Q-Learning vs DQN | Q-Learning chosen | State space is small and discrete |
| 1000 episodes vs more | 1000 chosen | Sufficient convergence, fast training |
| CSV + MLflow vs only MLflow | Both used | CSV for backup, MLflow for visualization |

| Risk | Mitigation |
|---|---|
| Overfitting to simulation | Randomized battery start (3-8 units) |
| Slow convergence | epsilon_decay tuned to 0.995 |
| Reproducibility failure | YAML configs + MLflow tracking |

---

## 7. System Design
Input: hour_of_day, battery_level
         ↓
EnergyGridEnv (Gymnasium)
         ↓
QLearningAgent (epsilon-greedy)
         ↓
Q-table update (Bellman equation)
         ↓
Output: energy_source, reward, carbon_count
         ↓
MLflow tracking + CSV logging

## 8. SDG Traceability
| SDG | Requirement | Implementation |
|---|---|---|
| SDG 7 | Clean energy usage | +10 reward for renewable, 75% usage achieved |
| SDG 9 | Smart infrastructure | AI-driven grid scheduling with MLflow |
| SDG 13 | Carbon reduction | Carbon reduced from 12 to ~0 by episode 800 |