import numpy as np 
import time

def policy_evaluation(p, pi, S, A, R, theta, gamma):
    V = np.zeros((len(S),))
    t = time.time()
    while True:
        delta = 0
        for s in S:
            old_v = V[s]
            V[s] = 0.0
            for a in A:
                for s_next in S:
                    for r_idx, r in enumerate(R):
                        V[s] += pi[s, a] * p[s, a, s_next, r_idx] * (r + gamma * V[s_next])
            delta = max(delta, abs(V[s] - old_v))

        if delta < theta:
            break
    print(time.time() - t)
    return V



def policy_iteration(p, pi, S, A, R, theta, gamma):
    t = time.time()
    
    # 1 - Init
    V = np.zeros((len(S),))
    while True:

        # 2 - Policy Evaluation
        while True:
            delta = 0
            for s in S:
                old_v = V[s]
                V[s] = 0.0
                for a in A:
                    for s_next in S:
                        for r_idx, r in enumerate(R):
                            V[s] += pi[s, a] * p[s, a, s_next, r_idx] * (r + gamma * V[s_next])
                delta = max(delta, abs(V[s] - old_v))

            if delta < theta:
                break

        # 3 - Policy improvement
        policy_stable = True
        for s in S:
            old_policy = pi[s, :]

            best_a = None
            best_a_value = None
            for a in A:
                a_value = 0
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        a_value += p[s, a, s_p, r_idx] * (r + gamma * V[s_p])
                if best_a_value is None or best_a_value < a_value:
                    best_a_value = a_value
                    best_a = a

            pi[s, :] = 0.0
            pi[s, best_a] = 1.0
            if not np.array_equal(pi[s], old_policy):
                policy_stable = False

        if policy_stable:
            break
    print(time.time() - t)
    return (pi, V)


def value_iteration(p, pi, S, A, R, theta, gamma):
  
    V = np.random.random((S.shape[0],))
    while True:
        delta = 0
        for s in S:
            old_v = V[s]
            V[s] = 0.0
            for a in A:
                for s_next in S:
                    for r_idx, r in enumerate(R):
                        V[s] += pi[s, a] * p[s, a, s_next, r_idx] * (r + gamma * V[s_next])
            delta = max(delta, abs(V[s] - old_v))

        if delta < theta:
            break

    Pi = np.zeros((S.shape[0], A.shape[0]))
    for s in S:
        best_action = 0
        best_action_score = -9999999999999
        for a in A:
            tmp_sum = 0
            for s_p in S:
                tmp_sum += p[s, a, s_p, 0] * (
                        p[s, a, s_p, 1] + gamma * V[s_p]
                )
            if tmp_sum > best_action_score:
                best_action = a
                best_action_score = tmp_sum
        Pi[s] = 0.0
        Pi[s, best_action] = 1.0
    return Pi,V

