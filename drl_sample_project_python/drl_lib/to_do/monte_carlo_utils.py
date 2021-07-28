from ..do_not_touch.contracts import SingleAgentEnv
from ..do_not_touch.result_structures import PolicyAndActionValueFunction
import numpy as np


def monte_carlo_with_exploring_starts_control( env: SingleAgentEnv,
        max_episodes_count: int,
        max_steps: int,
        gamma: float):
    pi = {}
    q = {}
    action_dim = len(env.available_actions_ids())
    returns = {} 
    returns_count = {} 

    for ep in range(max_episodes_count):
        env.reset_random()
        if not env.is_game_over():
            s0 = env.state_id()
            score_before = env.score()
            actions = env.available_actions_ids()
            a0 = np.random.choice(actions)
            
            env.act_with_action_id(a0)
            s1 = env.state_id()
            score_after = env.score()
            r1 = score_before - score_after
            
            s_list = []
            a_list = []
            s_p_list = []
            r_list = []
            st = s1
            actions = env.available_actions_ids()
            steps_count = 0
            while not env.is_game_over() and steps_count < max_steps:
                at = np.random.choice(actions)
                score_before = env.score()
                env.act_with_action_id(at)
                st_p = env.state_id()
                rt_p = score_before - env.score()
                
                s_list.append(st)
                a_list.append(at)
                s_p_list.append(st_p)
                r_list.append(rt_p)
                st = st_p
                steps_count += 1
                actions = env.available_actions_ids()
                
            s_list = [s0] + s_list
            a_list = [a0] + a_list
            r_list = [r1] + r_list

            G = 0
            for t in reversed(range(len(s_list))):
                G = gamma * G + r_list[t]
                st = s_list[t]
                at = a_list[t]

                if (st, at) in zip(s_list[0:t], a_list[0:t]):
                    continue
                
                possible_actions = env.available_actions_ids()
                if st not in returns.keys():
                    returns[st] = np.zeros(action_dim)
                    returns_count[st] = np.zeros(action_dim)
                    q[st] = np.zeros(action_dim)
                    pi[st]= np.zeros(action_dim)

                    for a in range(action_dim):
                        if a not in possible_actions:
                            q[st][a] = -1

                returns[st][at] += G
                returns_count[st][at] += 1
                q[st][at] = returns[st][at] / returns_count[st][at]
                pi[st]= np.zeros(action_dim)
                
                pi[st][np.argmax(q[st])] = 1.0
    return q, pi

 
 
def on_policy_first_visit_monte_carlo_control(
        env: SingleAgentEnv,
        epsilon: float,
        max_episodes_count: int,
        gamma: float
):
    pi = {}
    q = {}
    returns = {}

    for ep in range(max_episodes_count):
        env.reset()

        S = []
        A = []
        R = []

        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions = env.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                returns[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    q[s][a] = 0.0
                    returns[s][a] = []

            chosen_action = np.random.choice(
                list(pi[s].keys()),
                1,
                False,
                p=list(pi[s].values())
            )[0]
            A.append(chosen_action)

            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)

        G = 0
        for t in reversed(range(len(S))):
            G = gamma * G + R[t]

            found = False
            for prev_s, prev_a in zip(S[:t], A[:t]):
                if prev_s == S[t] and prev_a == A[t]:
                    found = True
                    break
            if found:
                continue

            returns[S[t]][A[t]].append(G)
            q[S[t]][A[t]] = np.mean(returns[S[t]][A[t]])

            best_action = list(q[S[t]].keys())[np.argmax(
                list(q[S[t]].values())
            )]

            for a_key in pi[S[t]].keys():
                if a_key == best_action:
                    pi[S[t]][a_key] = 1 - epsilon + epsilon / len(pi[S[t]])
                else:
                    pi[S[t]][a_key] = epsilon / len(pi[S[t]])

    return q, pi

def off_policy_monte_carlo_control(
        env: SingleAgentEnv,
        max_episodes_count: int,
        gamma: float
):
    pi = {}
    q = {}
    returns = {}
    C = {}
    b = {}
    
    action_dim = len(env.available_actions_ids())

    for ep in range(max_episodes_count):
        env.reset()

        S = []
        A = []
        R = []

        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions = env.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                returns[s] = {}
                b[s] = {}
                for a in available_actions:
                    
                    pi[s][a] = 1.0 / len(available_actions)
                    q[s][a] = 0.0
                    returns[s][a] = []
                    b[s][a] = 1.0 

            
            try:
                chosen_action = np.random.choice(
                    list(pi[s].keys()),
                    1,
                    False,
                    p=list(pi[s].values())
                )[0]
            except:
                chosen_action = pi[s][a]
            A.append(chosen_action)

            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)

         
        G = 0
        W = 1
        for t in reversed(range(len(S))):
            G = gamma * G + R[t]
            st = S[t]
            at = int(A[t])
            possible_actions = env.available_actions_ids()
            if st not in C.keys():
            	C[st] = np.zeros(action_dim)
            	q[st] = np.zeros(action_dim)
            	pi[st] = np.zeros(action_dim)
            	b[st]=np.ones(action_dim)*1.0/len(possible_actions)
            	for a in range(action_dim):
	            	if a not in possible_actions:
	            		q[st][a] = -9999999
	            		b[st][a] = 0
            C[st][at] += W

            q[st][at] += W / C[st][at] * (G - q[st][at])
            pi[st] = np.zeros(action_dim)
            pi[st][np.argmax(q[st])] = 1.0

            if at != np.argmax(q[st]):
                break

            W = W / b[st][at]

    return q, pi