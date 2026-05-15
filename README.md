## Stakeholders
| Stakeholder | Interest |
|-------------|----------|
| Grid Operators | Optimise energy dispatch |
| Energy Consumers | Reliable, clean power |
| Government/Regulators | Meet SDG targets |
| ML Engineers | Reproducible MLOps pipeline |

---

## Functional Requirements
- Agent must schedule energy for 24 hours/day
- Must prefer renewable sources when available
- Must log all experiments via MLflow
- Must support rollback to best saved policy
- CI/CD must auto-train on every push

## Non-Functional Requirements
- Training must complete within 1000 episodes
- Pipeline must be reproducible across environments
- System must be scalable to multiple configs
- Carbon reduction target: >90%

---

## Feasibility & Constraints
- **Technical:** Q-Learning suitable for small discrete state space
- **Constraint:** Battery level capped at 10 units
- **Constraint:** 24-hour fixed episode length
- **Trade-off:** Simplicity of Q-table vs Deep RL accuracy
- **Feasibility:** Fully runnable on CPU, no GPU needed

---

## Risks & Traceability
| Risk | Mitigation |
|------|------------|
| Policy divergence | Rollback to best saved policy |
| Experiment loss | MLflow tracking all runs |
| Dependency issues | requirements.txt pinned |
| Pipeline failure | GitHub Actions CI/CD alerts |

---

## System Design
---

## Tools Used
| Tool | Purpose |
|------|---------|
| Python + Gymnasium | Custom RL environment |
| MLflow | Experiment tracking & model registry |
| GitHub Actions | CI/CD automation |
| Q-Learning | Scheduling agent |
| Matplotlib | Visualisation |
| PyYAML | Config management |

---

## Results & Conclusions
- Carbon emissions reduced by **92%** over 1000 episodes
- Renewable usage achieved: **75%+**
- Best policy saved and rollback tested successfully
- CI/CD pipeline fully automated and reproducible
