# field-dynamics-sim

**Fleet spectral health simulation — multi-agent field dynamics with cooperative, adversarial, emergent, and phase transition scenarios.**

Python simulation of 20 agents (explorers, builders, guardians) in a 2D field. Uses the Conservation Spectral SDK to build tension graphs from agent interactions, compute Laplacians, and track spectral health metrics (conservation ratio, Fiedler value, spectral entropy) in real-time across four scenarios.

## What This Gives You

- **Cooperative scenario** — all agents share goals, high conservation (CR ≈ 0.95+)
- **Adversarial scenario** — injected rogue agents, conservation drops
- **Emergent scenario** — no cooperation bonus, watch if conservation arises naturally
- **Phase transition** — gradually increase rogue count, detect critical point where fleet coherence collapses
- **Spectral fingerprinting** — effective dimension, spectral entropy, Cheeger constant
- **Publication-quality plots** — agent positions, conservation over time, Fiedler partition, phase diagrams

## Quick Start

```bash
pip install numpy scipy matplotlib scikit-learn
python run_all.py
```

Outputs go to `output/` with PNG plots and console reports for each scenario.

## Scenarios

| Scenario | What Happens | Expected CR |
|----------|-------------|-------------|
| Cooperative | All agents cooperate | ~0.95+ |
| Adversarial | 7 rogue agents injected | ~0.3–0.5 |
| Emergent | No cooperation bonus | Variable — tests emergence |
| Phase Transition | Gradual rogue injection | CR drops at critical point |

## Architecture

```
simulation.py     # MultiAgentField: agents, physics, interaction graph
scenarios.py      # 4 predefined scenarios
visualization.py  # Publication-quality matplotlib plots
run_all.py        # Entry point: runs all scenarios
```

## Testing

```bash
python run_all.py  # generates output/ directory with plots
```

## How It Fits

Part of the SuperInstance ecosystem:

- **[field-dynamics](https://github.com/SuperInstance/field-dynamics)** — Browser-based interactive version
- **[conservation-spectral-python](https://github.com/SuperInstance/conservation-spectral-python)** — The Conservation Spectral SDK used here
- **field-dynamics-sim** — Python simulation with full analysis pipeline (this repo)

## License

MIT
