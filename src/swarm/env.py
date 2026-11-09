"""
A thin PettingZoo ParallelEnv wrapper around sim.py.
On reset it picks the start and target shapes.
It builds each robot's observation (6 neighbor bits, normalized (x,y,z), and the target bitmap),
passes along the action masks, and gives the completion reward when the episode ends.
"""