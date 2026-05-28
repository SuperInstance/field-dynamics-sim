"""Scenarios: Cooperative, Adversarial, Emergent, Phase Transition."""

from __future__ import annotations

import numpy as np
from simulation import MultiAgentField


def scenario_cooperative(n_steps=500, n_agents=20) -> MultiAgentField:
    """All agents cooperate. Expect high conservation."""
    print("\n" + "="*70)
    print("SCENARIO A: Fully Cooperative Fleet")
    print("="*70)
    sim = MultiAgentField(n_agents=n_agents, seed=42)
    sim.run_steps(n_steps, record_conservation_every=2)

    report = sim.conservation_report()
    print(f"\n  Results after {n_steps} steps:")
    print(f"    Spectral gap:     {report.spectral_gap:.4f}")
    print(f"    Cheeger constant: {report.cheeger_constant:.4f}")
    print(f"    Spectral entropy: {report.fingerprint.spectral_entropy:.4f}")
    print(f"    Effective dim:    {report.fingerprint.effective_dimension:.2f}")
    print(f"    Mean energy:      {np.mean([a.energy for a in sim.agents]):.1f}")
    print(f"    Mean conservation: {np.mean(sim.conservation_history[-10:]):.6f}")
    return sim


def scenario_adversarial(n_steps=500, n_agents=20, n_adversaries=7) -> MultiAgentField:
    """Some agents go rogue from the start. Expect lower conservation."""
    print("\n" + "="*70)
    print(f"SCENARIO B: Adversarial Fleet ({n_adversaries} rogue agents)")
    print("="*70)
    sim = MultiAgentField(n_agents=n_agents, seed=42)
    sim.inject_adversaries(n_adversaries)
    sim.run_steps(n_steps, record_conservation_every=2)

    report = sim.conservation_report()
    print(f"\n  Results after {n_steps} steps:")
    print(f"    Spectral gap:     {report.spectral_gap:.4f}")
    print(f"    Cheeger constant: {report.cheeger_constant:.4f}")
    print(f"    Mean energy:      {np.mean([a.energy for a in sim.agents]):.1f}")
    print(f"    Mean conservation: {np.mean(sim.conservation_history[-10:]):.6f}")
    return sim


def scenario_emergent(n_steps=500, n_agents=20) -> MultiAgentField:
    """No explicit cooperation rules. See if conservation emerges naturally.
    We disable the cooperation bonus to see raw dynamics."""
    print("\n" + "="*70)
    print("SCENARIO C: Emergent (no cooperation bonus)")
    print("="*70)

    # Monkey-patch to disable cooperation bonus
    original_bonus = MultiAgentField._cooperation_bonus

    def no_bonus(self):
        pass

    MultiAgentField._cooperation_bonus = no_bonus

    sim = MultiAgentField(n_agents=n_agents, seed=42)
    sim.run_steps(n_steps, record_conservation_every=2)

    # Restore
    MultiAgentField._cooperation_bonus = original_bonus

    report = sim.conservation_report()
    print(f"\n  Results after {n_steps} steps:")
    print(f"    Spectral gap:     {report.spectral_gap:.4f}")
    print(f"    Cheeger constant: {report.cheeger_constant:.4f}")
    print(f"    Mean conservation: {np.mean(sim.conservation_history[-10:]):.6f}")
    return sim


def scenario_phase_transition(n_agents=20, max_adversaries=15) -> dict:
    """Gradually increase adversarial fraction, measure conservation drop."""
    print("\n" + "="*70)
    print("SCENARIO D: Phase Transition Curve")
    print("="*70)

    results = {
        'adversary_fractions': [],
        'conservation_means': [],
        'spectral_gaps': [],
        'cheeger_constants': [],
        'mean_energies': [],
    }

    steps_per_point = 200
    adversary_counts = list(range(0, min(max_adversaries + 1, n_agents)))

    for n_adv in adversary_counts:
        sim = MultiAgentField(n_agents=n_agents, seed=42)
        if n_adv > 0:
            sim.inject_adversaries(n_adv)
        sim.run_steps(steps_per_point, record_conservation_every=5)

        report = sim.conservation_report()
        frac = n_adv / n_agents

        mean_cons = np.mean(sim.conservation_history[-5:]) if sim.conservation_history else 0
        mean_energy = np.mean([a.energy for a in sim.agents])

        results['adversary_fractions'].append(frac)
        results['conservation_means'].append(mean_cons)
        results['spectral_gaps'].append(report.spectral_gap)
        results['cheeger_constants'].append(report.cheeger_constant)
        results['mean_energies'].append(mean_energy)

        print(f"  Adversaries: {n_adv:2d}/{n_agents} ({frac:.0%}) | "
              f"Conservation: {mean_cons:.6f} | "
              f"Gap: {report.spectral_gap:.4f} | "
              f"Energy: {mean_energy:.1f}")

    return results


def scenario_key_experiment(n_steps=1000, inject_at=500, n_adversaries=5) -> MultiAgentField:
    """THE KEY EXPERIMENT:
    - Run 1000 steps cooperative
    - At step 500, inject adversarial agents
    - Show conservation DROPS
    - Show Fiedler vector SEPARATES cooperative from adversarial
    """
    print("\n" + "="*70)
    print("KEY EXPERIMENT: Cooperative → Adversary Injection at Step 500")
    print("="*70)

    sim = MultiAgentField(n_agents=20, seed=42)

    # Phase 1: Cooperative
    print(f"\n  Phase 1: Running cooperative steps 0-{inject_at}...")
    sim.run_steps(inject_at, record_conservation_every=2)
    pre_injection_conservation = np.mean(sim.conservation_history[-10:]) if sim.conservation_history else 0

    # Inject adversaries
    print(f"\n  *** INJECTING {n_adversaries} ADVERSARIAL AGENTS AT STEP {inject_at} ***")
    sim.inject_adversaries(n_adversaries)

    # Phase 2: Post-injection
    remaining = n_steps - inject_at
    print(f"\n  Phase 2: Running with adversaries for {remaining} more steps...")
    sim.run_steps(remaining, record_conservation_every=2)
    post_injection_conservation = np.mean(sim.conservation_history[-10:]) if sim.conservation_history else 0

    # Analysis
    report = sim.conservation_report()
    print(f"\n  Results:")
    print(f"    Pre-injection conservation:  {pre_injection_conservation:.6f}")
    print(f"    Post-injection conservation: {post_injection_conservation:.6f}")
    print(f"    Change: {post_injection_conservation - pre_injection_conservation:.6f}")
    print(f"    Spectral gap:     {report.spectral_gap:.4f}")
    print(f"    Cheeger constant: {report.cheeger_constant:.4f}")

    # Fiedler analysis: separate cooperative from adversarial
    if sim.fiedler_history:
        fiedler = sim.fiedler_history[-1]
        cooperative_vals = [fiedler[a.id] for a in sim.agents if not a.adversarial]
        adversarial_vals = [fiedler[a.id] for a in sim.agents if a.adversarial]

        print(f"\n    Fiedler Vector Separation:")
        if cooperative_vals:
            print(f"      Cooperative agents: mean={np.mean(cooperative_vals):.4f}, "
                  f"std={np.std(cooperative_vals):.4f}")
        if adversarial_vals:
            print(f"      Adversarial agents: mean={np.mean(adversarial_vals):.4f}, "
                  f"std={np.std(adversarial_vals):.4f}")
        if cooperative_vals and adversarial_vals:
            separation = abs(np.mean(cooperative_vals) - np.mean(adversarial_vals))
            print(f"      Separation distance: {separation:.4f}")

    return sim
