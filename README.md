# Field Dynamics Simulation

Multi-agent field dynamics simulation with **Conservation Spectral Analysis** — demonstrating fleet spectral health using the Conservation Spectral SDK.

## What It Does

Simulates 20 agents in a 2D field with three roles:
- **Explorers** — seek unexplored regions, spread out
- **Builders** — cluster together, construct near center
- **Guardians** — patrol perimeter, protect low-energy agents

Tracks the **spectral health** of the fleet in real-time by building a tension graph from agent interactions and computing conservation ratios, spectral gap, Cheeger constant, and Fiedler vector partitions.

## Scenarios

| Scenario | Description | Expected |
|----------|-------------|----------|
| **Cooperative** | All agents work together | High conservation, coherent structure |
| **Adversarial** | 7 agents go rogue from start | Lower conservation, energy drain |
| **Emergent** | No cooperation bonus | See if conservation emerges naturally |
| **Phase Transition** | Gradually increase adversaries | Conservation drops monotonically |
| **Key Experiment** | Inject adversaries at step 500 | Conservation drops, Fiedler separates |

## Key Results

- **Phase transition curve**: Conservation drops from ~899 (0% adversaries) to ~313 (35% adversaries) — a clear monotonic decline
- **Fiedler separation**: The Fiedler vector naturally separates cooperative from adversarial agents
- **Spectral gap**: Tracks fleet cohesion — drops as adversarial fraction increases
- **Cheeger constant**: Measures how easily the fleet can be partitioned

## Setup

```bash
pip install numpy scipy matplotlib
```

## Run

```bash
python run_all.py
```

Output plots saved to `output/`.

## Architecture

```
simulation.py      — Agent, MultiAgentField (core simulation engine)
scenarios.py       — 5 experimental scenarios
visualization.py   — All matplotlib plots
run_all.py         — Execute everything
```

## Fleet Spectral Health

This simulation demonstrates the core concept from the CUDA Fleet ecosystem:

> The Laplacian of the fleet interaction graph captures the **conservation** of the multi-agent system. When agents cooperate, the system has high conservation (coherent structure). When adversaries disrupt, conservation drops. The Fiedler vector naturally identifies the partition between healthy and unhealthy parts of the fleet.

This is spectral graph theory applied to multi-agent systems — the same mathematics underlying the Conservation Spectral SDK.

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct) ecosystem.
