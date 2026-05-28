"""Visualization: Agent positions, conservation over time, Fiedler partition, phase transition."""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from simulation import MultiAgentField


ROLE_COLORS = {
    'explorer': '#3498db',
    'builder': '#2ecc71',
    'guardian': '#e74c3c',
    'rogue': '#8e44ad',
}

ROLE_MARKERS = {
    'explorer': 'D',
    'builder': 's',
    'guardian': '^',
    'rogue': 'X',
}


def plot_agent_positions(sim: MultiAgentField, title: str, filename: str):
    """2D scatter plot of agent positions, colored by role."""
    fig, ax = plt.subplots(figsize=(10, 10))

    for agent in sim.agents:
        color = '#8e44ad' if agent.adversarial else ROLE_COLORS.get(agent.role, '#95a5a6')
        marker = 'X' if agent.adversarial else ROLE_MARKERS.get(agent.role, 'o')
        size = 120 if agent.adversarial else 80
        ax.scatter(agent.pos[0], agent.pos[1], c=color, marker=marker, s=size,
                   edgecolors='white', linewidths=0.5, zorder=5,
                   label=agent.role if not any(a.role == agent.role and a.id < agent.id
                                                for a in sim.agents) else None)

    # Draw communication edges
    graph = sim.build_tension_graph()
    W = graph.adjacency_matrix()
    for i in range(len(sim.agents)):
        for j in range(i + 1, len(sim.agents)):
            if W[i, j] > 0.1:
                alpha = min(W[i, j] / 2.0, 0.5)
                ax.plot([sim.agents[i].pos[0], sim.agents[j].pos[0]],
                        [sim.agents[i].pos[1], sim.agents[j].pos[1]],
                        'k-', alpha=alpha, linewidth=0.5)

    ax.set_xlim(-5, sim.field_size + 5)
    ax.set_ylim(-5, sim.field_size + 5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.2)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#3498db', markersize=10, label='Explorer'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ecc71', markersize=10, label='Builder'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#e74c3c', markersize=10, label='Guardian'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='#8e44ad', markersize=10, label='Rogue'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_conservation_over_time(sim: MultiAgentField, title: str, filename: str,
                                 inject_step: int = None):
    """Conservation ratio over time."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    steps = np.arange(len(sim.conservation_history)) * 2 * sim.dt

    # Conservation ratio
    axes[0].plot(steps, sim.conservation_history, color='#2c3e50', linewidth=1)
    axes[0].set_ylabel('Mean Conservation Ratio')
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Spectral gap
    axes[1].plot(steps, sim.spectral_gap_history, color='#e74c3c', linewidth=1)
    axes[1].set_ylabel('Spectral Gap')
    axes[1].grid(True, alpha=0.3)

    # Cheeger constant
    axes[2].plot(steps, sim.cheeger_history, color='#27ae60', linewidth=1)
    axes[2].set_ylabel('Cheeger Constant')
    axes[2].set_xlabel('Time')
    axes[2].grid(True, alpha=0.3)

    # Mark injection point
    if inject_step is not None:
        inject_time = inject_step * sim.dt
        for ax in axes:
            ax.axvline(x=inject_time, color='red', linestyle='--', linewidth=2, alpha=0.7,
                       label=f'Adversary injection (step {inject_step})')
            ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_fiedler_partition(sim: MultiAgentField, title: str, filename: str):
    """Show Fiedler vector partition: cooperative vs adversarial agents."""
    if not sim.fiedler_history:
        print("  [Warning] No Fiedler history available")
        return

    fiedler = sim.fiedler_history[-1]
    cooperative_mask = np.array([not a.adversarial for a in sim.agents])
    adversarial_mask = ~cooperative_mask

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart of Fiedler values
    agent_ids = np.arange(len(sim.agents))
    colors = ['#2ecc71' if not a.adversarial else '#8e44ad' for a in sim.agents]
    axes[0].bar(agent_ids, fiedler, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_xlabel('Agent ID')
    axes[0].set_ylabel('Fiedler Vector Component')
    axes[0].set_title('Fiedler Vector by Agent', fontweight='bold')
    axes[0].axhline(y=0, color='black', linewidth=0.5, linestyle='-')

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Cooperative'),
        Patch(facecolor='#8e44ad', label='Adversarial'),
    ]
    axes[0].legend(handles=legend_elements)

    # 2D position plot colored by Fiedler value
    sc = axes[1].scatter(
        [a.pos[0] for a in sim.agents],
        [a.pos[1] for a in sim.agents],
        c=fiedler, cmap='RdYlGn', s=100, edgecolors='black', linewidths=0.5
    )
    axes[1].set_xlim(-5, sim.field_size + 5)
    axes[1].set_ylim(-5, sim.field_size + 5)
    axes[1].set_aspect('equal')
    axes[1].set_title('Agent Positions colored by Fiedler Value', fontweight='bold')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(sc, ax=axes[1], label='Fiedler Value')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_phase_transition(results: dict, filename: str):
    """Phase transition curve: adversarial fraction vs conservation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fracs = results['adversary_fractions']

    # Conservation vs fraction
    axes[0, 0].plot(fracs, results['conservation_means'], 'o-', color='#2c3e50', linewidth=2)
    axes[0, 0].set_xlabel('Adversarial Fraction')
    axes[0, 0].set_ylabel('Mean Conservation Ratio')
    axes[0, 0].set_title('Conservation vs Adversarial Fraction', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Spectral gap vs fraction
    axes[0, 1].plot(fracs, results['spectral_gaps'], 'o-', color='#e74c3c', linewidth=2)
    axes[0, 1].set_xlabel('Adversarial Fraction')
    axes[0, 1].set_ylabel('Spectral Gap')
    axes[0, 1].set_title('Spectral Gap vs Adversarial Fraction', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Cheeger constant vs fraction
    axes[1, 0].plot(fracs, results['cheeger_constants'], 'o-', color='#27ae60', linewidth=2)
    axes[1, 0].set_xlabel('Adversarial Fraction')
    axes[1, 0].set_ylabel('Cheeger Constant')
    axes[1, 0].set_title('Cheeger Constant vs Adversarial Fraction', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Mean energy vs fraction
    axes[1, 1].plot(fracs, results['mean_energies'], 'o-', color='#3498db', linewidth=2)
    axes[1, 1].set_xlabel('Adversarial Fraction')
    axes[1, 1].set_ylabel('Mean Agent Energy')
    axes[1, 1].set_title('Fleet Energy vs Adversarial Fraction', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('Phase Transition: Fleet Spectral Health under Adversarial Pressure',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_energy_over_time(sim: MultiAgentField, title: str, filename: str):
    """Mean energy over time for the fleet."""
    if not sim.mean_energy_history:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    steps = np.arange(len(sim.mean_energy_history))
    ax.plot(steps, sim.mean_energy_history, color='#f39c12', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Fleet Energy')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")
