"""Tests for Field Dynamics Simulation."""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from simulation import Agent, MultiAgentField


class TestAgent:
    def test_creation(self):
        agent = Agent(id=0, role="explorer", pos=[10, 20], vel=[1, 0], energy=80)
        assert agent.id == 0
        assert agent.role == "explorer"
        assert agent.energy == 80
        assert not agent.adversarial
        assert agent.health == 1.0

    def test_pos_is_numpy(self):
        agent = Agent(id=0, role="builder", pos=[10, 20], vel=[0, 0], energy=50)
        assert isinstance(agent.pos, np.ndarray)
        assert agent.pos.shape == (2,)

    def test_vel_is_numpy(self):
        agent = Agent(id=0, role="guardian", pos=[0, 0], vel=[1, -1], energy=60)
        assert isinstance(agent.vel, np.ndarray)
        assert agent.vel.shape == (2,)


class TestMultiAgentField:
    def test_init(self):
        sim = MultiAgentField(n_agents=6, seed=42)
        assert len(sim.agents) == 6
        assert sim.time == 0.0
        assert sim.field_size == 100.0

    def test_agent_roles(self):
        sim = MultiAgentField(n_agents=9, seed=42)
        roles = {a.role for a in sim.agents}
        assert "explorer" in roles
        assert "builder" in roles
        assert "guardian" in roles

    def test_step(self):
        sim = MultiAgentField(n_agents=6, seed=42)
        pos_before = sim.agents[0].pos.copy()
        sim.step(dt=0.1)
        assert sim.time == pytest.approx(0.1)

    def test_multiple_steps(self):
        sim = MultiAgentField(n_agents=6, seed=42)
        sim.run_steps(10, record_conservation_every=5)
        assert sim.time == pytest.approx(1.0)
        assert len(sim.position_history) > 0

    def test_inject_adversaries(self):
        sim = MultiAgentField(n_agents=10, seed=42)
        sim.inject_adversaries(3)
        adversarial_count = sum(1 for a in sim.agents if a.adversarial)
        assert adversarial_count == 3

    def test_build_tension_graph(self):
        sim = MultiAgentField(n_agents=6, seed=42)
        sim.run_steps(5)
        tg = sim.build_tension_graph()
        assert tg is not None

    def test_conservation_report(self):
        sim = MultiAgentField(n_agents=6, seed=42)
        sim.run_steps(10, record_conservation_every=2)
        report = sim.conservation_report()
        assert report is not None

    def test_deterministic_with_seed(self):
        sim1 = MultiAgentField(n_agents=6, seed=42)
        sim2 = MultiAgentField(n_agents=6, seed=42)
        np.testing.assert_array_equal(sim1.agents[0].pos, sim2.agents[0].pos)

    def test_conservation_history_grows(self):
        sim = MultiAgentField(n_agents=6, seed=42)
        sim.run_steps(20, record_conservation_every=5)
        assert len(sim.conservation_history) > 0

    def test_energy_history_grows(self):
        sim = MultiAgentField(n_agents=6, seed=42)
        sim.run_steps(20, record_conservation_every=5)
        assert len(sim.energy_history) > 0
