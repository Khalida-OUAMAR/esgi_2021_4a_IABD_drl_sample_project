import numpy as np

MAX_CELLS = 51

S_line = np.arange(MAX_CELLS)
A_line = np.array([0, 1])  # 0 : Left, 1 : Right
R_line = np.array([-1, 0, 1])

P_line = np.zeros((len(S_line), len(A_line), len(S_line), len(R_line)))  # st, at, st+1, rt+1

for s in range(1, MAX_CELLS - 2):
    P_line[s, 1, s + 1, 1] = 1.0

for s in range(2, MAX_CELLS - 1):
    P_line[s, 0, s - 1, 1] = 1.0

P_line[MAX_CELLS - 2, 1, MAX_CELLS - 1, 2] = 1.0
P_line[1, 0, 0, 0] = 1.0
