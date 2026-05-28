"""Multi-Agent Field Dynamics Simulation with Conservation Spectral Analysis.

Simulates 20 agents in a 2D field with roles (explorer, builder, guardian).
Tracks spectral health of the fleet in real-time using the Conservation Spectral SDK.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Add SDK to path
SDK_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'conservation-spectral-python', 'src'))
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)

from conservation_spectral import (
    TensionGraph,
    build_laplacian,
    eigendecompose,
    analyze,
    conservation_ratios,
    spectral_gap,
    ConservationReport,
)


@dataclass
class Agent:
    id: int
    role: str  # 'explorer', 'builder', 'guardian'
    pos: np.ndarray  # 2D position
    vel: np.ndarray  # 2D velocity
    energy: float
    messages: list = field(default_factory=list)
    adversarial: bool = False
    health: float = 1.0

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.vel = np.asarray(self.vel, dtype=np.float64)


class MultiAgentField:
    """Multi-agent simulation in a 2D field."""

    def __init__(self, n_agents=20, field_size=100.0, seed=42):
        self.rng = np.random.RandomState(seed)
        self.field_size = field_size
        self.time = 0.0
        self.dt = 0.1

        roles = ['explorer', 'builder', 'guardian']
        self.agents: list[Agent] = []
        for i in range(n_agents):
            role = roles[i % 3]
            pos = self.rng.uniform(10, field_size - 10, size=2)
            vel = self.rng.uniform(-1, 1, size=2)
            energy = self.rng.uniform(50, 100)
            self.agents.append(Agent(id=i, role=role, pos=pos, vel=vel, energy=energy))

        # History tracking
        self.conservation_history: list[float] = []
        self.spectral_gap_history: list[float] = []
        self.cheeger_history: list[float] = []
        self.fiedler_history: list[np.ndarray] = []
        self.position_history: list[np.ndarray] = []
        self.energy_history: list[np.ndarray] = []
        self.mean_energy_history: list[float] = []

    def step(self, dt: float = 0.1):
        """One simulation step."""
        self.dt = dt
        self.time += dt

        # Communication phase
        self._communicate()

        # Movement phase (role-dependent)
        for agent in self.agents:
            if agent.adversarial:
                self._move_adversarial(agent)
            else:
                self._move_cooperative(agent)

            # Boundary reflection
            for d in range(2):
                if agent.pos[d] < 0:
                    agent.pos[d] = -agent.pos[d]
                    agent.vel[d] = abs(agent.vel[d])
                elif agent.pos[d] > self.field_size:
                    agent.pos[d] = 2 * self.field_size - agent.pos[d]
                    agent.vel[d] = -abs(agent.vel[d])

            # Energy dynamics
            if not agent.adversarial:
                agent.energy -= 0.05 * np.linalg.norm(agent.vel)  # movement cost
                agent.energy = max(agent.energy, 1.0)
            else:
                agent.energy -= 0.02  # slow drain for adversaries

        # Adversarial energy drain (happens to nearby agents)
        self._adversarial_drain()

        # Cooperation bonus: agents near same-role agents gain energy
        self._cooperation_bonus()

        # Energy normalization to prevent runaway growth
        total_energy = sum(a.energy for a in self.agents)
        if total_energy > 3000:  # Cap total system energy
            scale = 3000.0 / total_energy
            for a in self.agents:
                a.energy *= scale

        # Record state
        positions = np.array([a.pos.copy() for a in self.agents])
        energies = np.array([a.energy for a in self.agents])
        self.position_history.append(positions)
        self.energy_history.append(energies)
        self.mean_energy_history.append(float(np.mean(energies)))

    def _communicate(self):
        """Agents within communication range share information."""
        comm_range = 30.0
        for i, a1 in enumerate(self.agents):
            a1.messages = []
            for j, a2 in enumerate(self.agents):
                if i == j:
                    continue
                dist = np.linalg.norm(a1.pos - a2.pos)
                if dist < comm_range:
                    a1.messages.append({
                        'from': a2.id,
                        'role': a2.role,
                        'pos': a2.pos.copy(),
                        'energy': a2.energy,
                        'adversarial': a2.adversarial,
                    })

    def _move_cooperative(self, agent: Agent):
        """Role-based cooperative movement."""
        force = np.zeros(2)

        if agent.role == 'explorer':
            # Move away from other agents (seek unexplored regions)
            for msg in agent.messages:
                away = agent.pos - msg['pos']
                dist = np.linalg.norm(away)
                if dist > 0.1:
                    force += away / (dist ** 2) * 0.5
            # Random exploration drive
            force += self.rng.normal(0, 0.3, size=2)

        elif agent.role == 'builder':
            # Move toward other builders
            for msg in agent.messages:
                if msg['role'] == 'builder':
                    toward = msg['pos'] - agent.pos
                    dist = np.linalg.norm(toward)
                    if dist > 1.0:
                        force += toward / dist * 0.3
            # Stay near center
            center = np.array([self.field_size / 2, self.field_size / 2])
            to_center = center - agent.pos
            force += to_center / np.linalg.norm(to_center + 1e-8) * 0.1

        elif agent.role == 'guardian':
            # Patrol: orbit around center + move toward low-energy agents
            center = np.array([self.field_size / 2, self.field_size / 2])
            to_center = center - agent.pos
            dist_c = np.linalg.norm(to_center)
            # Tangential force for orbiting
            tangent = np.array([-to_center[1], to_center[0]])
            if dist_c > 1.0:
                tangent = tangent / dist_c
            force += tangent * 0.5
            # Seek low-energy allies
            for msg in agent.messages:
                if msg['energy'] < 30 and not msg['adversarial']:
                    toward = msg['pos'] - agent.pos
                    dist = np.linalg.norm(toward)
                    if dist > 1.0:
                        force += toward / dist * 0.4

        # Damping
        agent.vel = agent.vel * 0.95 + force * self.dt
        # Speed limit
        speed = np.linalg.norm(agent.vel)
        if speed > 3.0:
            agent.vel = agent.vel / speed * 3.0
        agent.pos += agent.vel * self.dt

    def _move_adversarial(self, agent: Agent):
        """Adversarial agents disrupt the system."""
        force = np.zeros(2)
        # Move toward nearest cooperative agent to drain their energy
        closest_dist = float('inf')
        closest_pos = None
        for msg in agent.messages:
            if not msg['adversarial']:
                dist = np.linalg.norm(agent.pos - msg['pos'])
                if dist < closest_dist:
                    closest_dist = dist
                    closest_pos = msg['pos']
        if closest_pos is not None:
            toward = closest_pos - agent.pos
            dist = np.linalg.norm(toward)
            if dist > 1.0:
                force += toward / dist * 0.6
        # Random jitter
        force += self.rng.normal(0, 0.5, size=2)

        agent.vel = agent.vel * 0.95 + force * self.dt
        speed = np.linalg.norm(agent.vel)
        if speed > 3.5:
            agent.vel = agent.vel / speed * 3.5
        agent.pos += agent.vel * self.dt

        # Drain energy of nearby cooperative agents (moved to _adversarial_drain)

    def _adversarial_drain(self):
        """Adversarial agents drain energy from nearby cooperative agents."""
        for i, a1 in enumerate(self.agents):
            if not a1.adversarial:
                continue
            for j, a2 in enumerate(self.agents):
                if a2.adversarial:
                    continue
                dist = np.linalg.norm(a1.pos - a2.pos)
                if dist < 20.0:
                    drain = 0.5 * (1.0 - dist / 20.0)
                    a2.energy -= drain
                    a2.health = max(0.0, a2.health - drain * 0.02)
                    a2.energy = max(a2.energy, 1.0)

    def _cooperation_bonus(self):
        """Cooperative agents near same-role allies gain energy."""
        for i, a1 in enumerate(self.agents):
            if a1.adversarial:
                continue
            for j, a2 in enumerate(self.agents):
                if i >= j:
                    continue
                if a2.adversarial:
                    continue
                dist = np.linalg.norm(a1.pos - a2.pos)
                if dist < 15.0:
                    bonus = 0.1 * (1.0 - dist / 15.0)
                    if a1.role == a2.role:
                        bonus *= 2.0  # Same-role synergy
                    a1.energy += bonus
                    a2.energy += bonus

    def build_tension_graph(self) -> TensionGraph:
        """Build a TensionGraph from current agent interactions.

        Nodes: agents
        Edges: communication frequency × role similarity
        """
        graph = TensionGraph(directed=False)
        n = len(self.agents)

        # Add vertices
        for agent in self.agents:
            graph.add_vertex(agent.id)

        # Add edges based on proximity and role interaction
        for i in range(n):
            for j in range(i + 1, n):
                a1, a2 = self.agents[i], self.agents[j]
                dist = np.linalg.norm(a1.pos - a2.pos)

                # Edge weight: inversely proportional to distance, boosted by role synergy
                if dist < 40.0:
                    base_weight = (1.0 - dist / 40.0)

                    # Role similarity boost
                    role_similarity = 1.0
                    if a1.role == a2.role:
                        role_similarity = 2.0
                    elif {a1.role, a2.role} == {'guardian', 'builder'}:
                        role_similarity = 1.5
                    elif {a1.role, a2.role} == {'explorer', 'builder'}:
                        role_similarity = 1.3

                    # Adversarial penalty
                    if a1.adversarial or a2.adversarial:
                        # Adversarial connections are weaker but still present
                        role_similarity *= 0.3

                    weight = base_weight * role_similarity
                    if weight > 0.01:
                        graph.add_edge(a1.id, a2.id, weight=weight)

        # Set energy attribute
        energies = np.array([a.energy for a in self.agents])
        graph.set_attribute('energy', energies)

        # Set health attribute
        healths = np.array([a.health for a in self.agents])
        graph.set_attribute('health', healths)

        return graph

    def conservation_report(self) -> ConservationReport:
        """Full spectral analysis of current agent state."""
        graph = self.build_tension_graph()
        return analyze(graph, attribute_name='energy')

    def inject_adversaries(self, n_adversaries: int = 5):
        """Convert random cooperative agents to adversarial."""
        cooperative = [a for a in self.agents if not a.adversarial]
        if len(cooperative) < n_adversaries:
            n_adversaries = len(cooperative)
        targets = self.rng.choice(len(cooperative), size=n_adversaries, replace=False)
        for idx in targets:
            agent = cooperative[idx]
            agent.adversarial = True
            agent.role = 'rogue'
            agent.energy = 80.0  # Reset energy

    def run_steps(self, n_steps: int, record_conservation_every: int = 5):
        """Run simulation for n_steps, recording conservation periodically."""
        for step_i in range(n_steps):
            self.step()

            if step_i % record_conservation_every == 0:
                try:
                    report = self.conservation_report()
                    self.conservation_history.append(
                        np.mean([r.ratio for r in report.ratios])
                    )
                    self.spectral_gap_history.append(report.spectral_gap)
                    self.cheeger_history.append(report.cheeger_constant)

                    # Fiedler vector (2nd eigenvector)
                    graph = self.build_tension_graph()
                    lap = build_laplacian(graph, normalized=True)
                    eigen = eigendecompose(lap, num_vectors=min(3, len(self.agents) - 1))
                    if eigen.num_vectors >= 2:
                        self.fiedler_history.append(eigen.eigenvectors[:, 1].copy())
                except Exception as e:
                    print(f"  [Warning] Conservation analysis failed at step {step_i}: {e}")

            if step_i % 100 == 0:
                n_adv = sum(1 for a in self.agents if a.adversarial)
                print(f"  Step {step_i}/{n_steps} | t={self.time:.1f} | "
                      f"adversaries={n_adv} | mean_energy={np.mean([a.energy for a in self.agents]):.1f}")
