"""
The shared policy network (a bitmap encoder plus an MLP over neighbors and position,
with separate actor and critic heads) and the masked IPPO loop:
collect rollouts from all robots, compute advantages, update, and log metrics.
"""