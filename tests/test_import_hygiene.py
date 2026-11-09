"""The simulator must not depend on RL, environment, or rendering libraries.

Keeping swarm.sim free of torch/pettingzoo/pygame is what lets the rules be
tested, profiled, and batched without the RL stack installed -- and what stops
game logic from drifting into the environment wrapper.
"""

import subprocess
import sys
import textwrap

FORBIDDEN = ("torch", "pettingzoo", "gymnasium", "pygame")


def run_probe(source: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        cwd=".",
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_sim_imports_no_rl_or_render_libraries():
    leaked = run_probe(
        f"""
        import sys
        sys.path.insert(0, "src")
        import swarm.sim.state, swarm.sim.moves, swarm.sim.connectivity, swarm.sim.shapes
        print(",".join(m for m in {FORBIDDEN!r} if m in sys.modules))
        """
    )
    assert leaked == "", f"swarm.sim pulled in forbidden modules: {leaked}"


def test_config_imports_no_rl_or_render_libraries():
    leaked = run_probe(
        f"""
        import sys
        sys.path.insert(0, "src")
        import swarm.config
        print(",".join(m for m in {FORBIDDEN!r} if m in sys.modules))
        """
    )
    assert leaked == "", f"swarm.config pulled in forbidden modules: {leaked}"


def test_ascii_renderer_needs_no_pygame():
    leaked = run_probe(
        """
        import sys
        sys.path.insert(0, "src")
        import swarm.render.ascii
        print("pygame" if "pygame" in sys.modules else "")
        """
    )
    assert leaked == ""


def test_render_package_does_not_import_pygame():
    """pygame must stay behind an explicit import of the viewer module.

    Training imports the simulator and the observation builder; if touching
    swarm.render pulled in SDL, every training run would pay for a display
    stack it never uses.
    """
    leaked = run_probe(
        """
        import sys
        sys.path.insert(0, "src")
        import swarm.render
        print("pygame" if "pygame" in sys.modules else "")
        """
    )
    assert leaked == ""


def test_observations_and_rewards_are_torch_free():
    leaked = run_probe(
        f"""
        import sys
        sys.path.insert(0, "src")
        import swarm.obs.local_window, swarm.reward
        print(",".join(m for m in {FORBIDDEN!r} if m in sys.modules))
        """
    )
    assert leaked == "", f"obs/reward pulled in forbidden modules: {leaked}"
