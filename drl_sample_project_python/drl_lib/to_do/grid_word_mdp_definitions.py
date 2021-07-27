import numpy as np

# Environnement grid world
############
width = 4
height = 4
num_states = width * height
S_grid = np.arange(num_states)
A_grid = np.arange(4)  # 0: left, 1: Right, 2: Up, 3: Down
# A_grid = [
#     0,  # droite
#     1,  # gauche
#     2,  # haut
#     3   # bas
# ]

#T = np.array([width - 1, num_states - 1])
R_grid = [-1, 0, 1]
P_grid = np.zeros((len(S_grid), len(A_grid), len(S_grid), len(R_grid)))

for s in S_grid:
    if (s % width) == 0:
        P_grid[s, 0, s, 0] = 1.0
    else:
        P_grid[s, 0, s - 1, 0] = 1.0
    if (s + 1) % width == 0:
        P_grid[s, 1, s, 0] = 1.0
    else:
        P_grid[s, 1, s + 1, 0] = 1.0
        
        
    if s < width:
        P_grid[s, 2, s, 0] = 1.0
    else:
        P_grid[s, 2, s - width, 0] = 1.0
    if s >= (num_states - width):
        P_grid[s, 3, s, 0] = 1.0
    else:
        P_grid[s, 3, s + width, 0] = 1.0

P_grid[width - 1, :, :, 0] = 0.0
P_grid[num_states - 1, :, :, 0] = 0.0

P_grid[:, :, width - 1, 1] = -1.0
P_grid[:, :, num_states - 1, 1] = 1.0

