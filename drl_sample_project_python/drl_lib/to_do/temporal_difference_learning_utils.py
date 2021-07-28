from ..do_not_touch.contracts import SingleAgentEnv
from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..do_not_touch.single_agent_env_wrapper import Env3
import numpy as np


def sarsa_algorithm(env: SingleAgentEnv,
                    episodes=10000,
                    max_steps=10,
                    epsilon=0.2,
                    alpha=0.1,
                    gamma=0.99) -> PolicyAndActionValueFunction:
    # init q(S,A)
    q = {}
    pi = {}
    nb_actions = len(env.available_actions_ids())

    # loop foreach episode
    for ep in range(episodes):

        # init s
        env.reset()
        state = env.state_id()
        env, q, action = get_random_action(env, q, state, nb_actions, epsilon)

        # loop foreach step of episode
        step = 0
        while not env.is_game_over() and step < max_steps:
            # Observe S' and R
            state, state_p, reward = observe_action_result(env, action)

            # Get A'
            env, q, action_p = get_random_action(env, q, state_p, nb_actions, epsilon)

            # Update Q(S,A)
            q[state][action] += alpha * (reward + gamma * q[state][action_p] - q[state][action])
            state = state_p
            action = action_p
            step += 1

    for s in q.keys():
        available_actions = env.available_actions_ids()
        if not len(available_actions) == 0 and not env.is_game_over():
            pi[s] = np.ones(nb_actions) * (epsilon / len(available_actions))
            for action in range(nb_actions):
                if action not in available_actions:
                    pi[s][action] = 0
            pi[s][np.argmax(q[s])] = 1.0 - epsilon + epsilon / len(available_actions)

    result = PolicyAndActionValueFunction(pi=pi, q=q)
    
    evaluate(env,pi)
    return result


def q_learning(
        env: SingleAgentEnv,
        alpha: float,
        epsilon: float,
        gamma: float,
        max_episodes: int
):
    pi = {}  # learned greedy policy
    b = {}  # behaviour epsilon-greedy policy
    q = {}  # action-value function of pi

    for episode_id in range(max_episodes):
        env.reset()

        while not env.is_game_over():
            s = env.state_id()
            available_actions = env.available_actions_ids()

            available_actions_count = len(available_actions)

            if s not in pi:
                pi[s] = {}
                b[s] = {}
                q[s] = {}

                for a in available_actions:
                    pi[s][a] = 1.0 / available_actions_count
                    b[s][a] = 1.0 / available_actions_count
                    q[s][a] = 0.0

            optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]

            for a_key, q_s_a in q[s].items():
                if a_key == optimal_a:
                    b[s][a_key] = 1.0 - epsilon + epsilon / available_actions_count
                else:
                    b[s][a_key] = epsilon / available_actions_count

            chosen_action = np.random.choice(list(pi[s].keys()), 1, False, p=list(pi[s].values()))[0]

            previous_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - previous_score

            s_p = env.state_id()
            next_available_actions = env.available_actions_ids()
            next_available_actions_count = len(next_available_actions)

            if env.is_game_over():
                q[s][chosen_action] += alpha * (r + 0.0 - q[s][chosen_action])
            else:
                if s_p not in pi:
                    pi[s_p] = {}
                    b[s_p] = {}
                    q[s_p] = {}

                    for a in next_available_actions:
                        pi[s_p][a] = 1.0 / next_available_actions_count
                        b[s_p][a] = 1.0 / next_available_actions_count
                        q[s_p][a] = 0.0

                q[s][chosen_action] += alpha * (r + gamma * np.max(list(q[s_p].values())) - q[s][chosen_action])

    for s in q.keys():
        optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]
        for a_key, q_s_a in q[s].items():
            if a_key == optimal_a:
                pi[s][a_key] = 1.0
            else:
                pi[s][a_key] = 0.0

    result = PolicyAndActionValueFunction(pi=pi, q=q)
    evaluate(env,pi)
    return result



def expected_sarsa(
        env: SingleAgentEnv,
        episodes=50000,
        max_steps=10,
        epsilon=0.2,
        alpha=0.1,
        gamma=0.99

):
    actions = env.available_actions_ids()
    
    pi = {}
    nb_actions = len(env.available_actions_ids())
    q = np.random.random((nb_actions, nb_actions))
    

    for ep in range(episodes):
        env.reset()
        s = env.state_id()
        rdm = np.random.random()
        #a = np.random.choice(actions) if rdm < epsilon else np.argmax(q[s, :])
        step = 0
        while not env.is_game_over() and step < max_steps:
            
            
            s = env.state_id()
            q[s, :] = 0.0
            a = np.random.choice(actions) if rdm < epsilon else np.argmax(q[s, :])
            score_before = env.score()
            actions = env.available_actions_ids()
            env.act_with_action_id(a)       
            s_p = env.state_id()
            score_after = env.score()

            r = score_before - score_after
            
          
            rdm = np.random.random()
            try:
                a_p = np.random.choice(actions) if rdm < epsilon else np.argmax(q[s_p, :])
            except:
                a_p = np.random.choice(actions)
            expected_value = np.mean(q[s_p,:])
            q[s, a] += alpha * (r + gamma * expected_value - q[s, a])
            s = s_p
            a = a_p
            step += 1


            pi[s, np.argmax(q[s, :])] = 1.0 - epsilon + epsilon / nb_actions
    evaluate(env,pi)
    result = PolicyAndActionValueFunction(pi=pi, q=q)
    return result


def get_random_action(env: SingleAgentEnv, q, state, nb_actions, epsilon):
    rdm = np.random.random()
    if state not in q.keys():
        q[state] = np.random.random(nb_actions)

    available_actions = env.available_actions_ids()

    for act in range(nb_actions):
        if act not in available_actions:
            q[state][act] = -1

    # choose A from S
    if len(available_actions) > 0:
        action = np.random.choice(available_actions) if rdm < epsilon else np.argmax(q[state])
    else:
        action = np.argmax(q[state])
    return env, q, action


def observe_action_result(env: SingleAgentEnv, action):
    state = env.state_id()
    score_before = env.score()
    env.act_with_action_id(action)
    state_p = env.state_id()
    score_after = env.score()

    if score_after > score_before:
        reward = 1
    else:
        reward = 0
    return state, state_p, reward



def evaluate(env: SingleAgentEnv, pi):
    done = False
    state = env.reset()
    nb_episodes_test = 1000
    successes = 0
    fails = 0
    action_dim = len(env.available_actions_ids())
    for i in range(nb_episodes_test):
        env.reset()
        done = False
        while not done:
            if state in pi.keys():
                action = np.random.choice(np.arange(action_dim))
            else:
                action = np.random.choice(env.available_actions_ids())

            score_before = env.score()
            env.act_with_action_id(action)
            state = env.state_id()
            reward = score_before - env.score()
            done = env.is_game_over()
            if reward == 1:
                successes += 1
            elif reward == -1:
                fails += 1
                
    print("Success rate: ", successes*1.0/nb_episodes_test, "Failure rate: ", fails*1.0/nb_episodes_test)

