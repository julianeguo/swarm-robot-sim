# simulation

"""
The simulator itself, with no RL code. It holds the state (the set of occupied cells and each robot's position) and has:

reset(start_shape) to place the robots
legal_action_mask(), which gives each robot's 7 actions with the pivot check
apply_actions(), which applies the moves and resolves conflicts
a completion score comparing the current configuration against the target cells
"""