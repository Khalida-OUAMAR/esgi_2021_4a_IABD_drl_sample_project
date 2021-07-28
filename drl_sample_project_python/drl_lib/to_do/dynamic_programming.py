from ..do_not_touch.mdp_env_wrapper import Env1
from ..do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction
import time
from .line_word_mdp_definitions import *
from .grid_word_mdp_definitions import *
from .dynamic_programming_utils import *

def policy_evaluation_on_line_world() -> ValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """   
    
    
    pi = np.zeros((len(S_line), len(A_line)))  # policy

    pi[:, 1] = 0.5  # Stratégie
    pi[:, 0] = 0.5  # Stratégie

    theta = 0.00001
    gamma = 1.0
    start_time = time.time()
    V = policy_evaluation(P_line, pi, S_line, A_line, R_line, theta, gamma)
    print("--- %s seconds ---" % (time.time() - start_time))
    return V



def policy_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    

    pi = np.ones((len(S_line), len(A_line)))
    pi /= len(A_line)


    theta = 0.00001
    gamma = 0.99999
    
    start_time = time.time()
    (pi, V) = policy_iteration(P_line, pi, S_line, A_line, R_line, theta, gamma)
    print("--- %s seconds ---" % (time.time() - start_time))
    return PolicyAndValueFunction(pi=pi, v=V)


def value_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """

    pi = np.ones((len(S_line), len(A_line)))
    pi /= len(A_line)
    
    #V[T] = 0.0
    
    theta = 0.00001
    gamma = 0.99999
    start_time = time.time()
    (pi, V) = value_iteration(P_line, pi, S_line, A_line, R_line, theta, gamma)
    print("--- %s seconds ---" % (time.time() - start_time))
    return PolicyAndValueFunction(pi=pi, v=V)


def policy_evaluation_on_grid_world() -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    pi = np.ones((len(S_grid), len(A_grid))) 
    pi /= len(A_grid)
    
    theta = 0.00001
    gamma = 1.0
    start_time = time.time()
    V = policy_evaluation(P_grid, pi, S_grid, A_grid, R_grid, theta, gamma)
    print("--- %s seconds ---" % (time.time() - start_time))
    return V


def policy_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    pi = np.ones((len(S_grid), len(A_grid))) 
    pi /= len(A_grid)
    
    theta = 0.00001
    gamma = 1.0
    start_time = time.time()
    (pi, V) = policy_iteration(P_grid, pi, S_grid, A_grid, R_grid, theta, gamma)
    print("--- %s seconds ---" % (time.time() - start_time))
    return PolicyAndValueFunction(pi=pi, v=V)


def value_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # TODO
    pi = np.ones((len(S_grid), len(A_grid))) 
    pi /= len(A_grid)
    
    theta = 0.00001
    gamma = 1.0
    start_time = time.time()
    (pi, V) = value_iteration(P_grid, pi, S_grid, A_grid, R_grid, theta, gamma)
    print("--- %s seconds ---" % (time.time() - start_time))
    return PolicyAndValueFunction(pi=pi, v=V)


def policy_evaluation_on_secret_env1() -> ValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    env = Env1()
    pass


def policy_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    # TODO
    pass


def value_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Prints the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    # TODO
    pass


def demo():
    print(policy_evaluation_on_line_world())
    print(policy_iteration_on_line_world())
    print(value_iteration_on_line_world())

    print(policy_evaluation_on_grid_world())
    print(policy_iteration_on_grid_world())
    print(value_iteration_on_grid_world())

    print(policy_evaluation_on_secret_env1())
    print(policy_iteration_on_secret_env1())
    print(value_iteration_on_secret_env1())
