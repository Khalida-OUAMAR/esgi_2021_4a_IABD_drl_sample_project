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
    return result


def q_learning(env: SingleAgentEnv,
               episodes=50000,
               max_steps=10,
               epsilon=0.2,
               alpha=0.1,
               gamma=0.99):
    q = {}  # np.random.random((states_count, actions_count))
    pi = {}
    nb_actions = len(env.available_actions_ids())

    for episode_id in range(episodes):
        # init s
        env.reset()
        state = env.state_id()
        step = 0

        while not env.is_game_over() and step < max_steps:

            if state not in q.keys():
                q[state] = np.random.random(nb_actions)

            available_actions = env.available_actions_ids()

            for act in range(nb_actions):
                if act not in available_actions:
                    q[state][act] = -1

            rdm = np.random.random()
            action = np.random.choice(available_actions) if rdm < epsilon else np.argmax(q[state])
            (state, state_p, reward) = observe_action_result(env, action)
            if state_p not in q.keys():
                q[state_p] = np.random.random(nb_actions)
            available_actions = env.available_actions_ids()

            for act in range(nb_actions):
                if act not in available_actions:
                    q[state_p][act] = -1
            q[state][action] += alpha * (reward + gamma * np.max(q[state_p]) - q[state][action])
            state = state_p
            step += 1

    for s in q.keys():
        pi[s] = np.zeros(nb_actions)
        pi[s][np.argmax(q[s])] = 1.0

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


def evaluate(env, pi):
    done = False
    state = env.reset()
    nb_episodes_test = 1000
    successes = 0
    fails = 0
    action_dim = 9
    for i in range(nb_episodes_test):
        state = env.reset()
        done = False
        while not done:
            if state in pi.keys():
                action = np.random.choice(np.arange(action_dim), p=Pi[state])
            else:
                action = np.random.choice(get_possible_actions(state))

            state, reward, done = step(action)
            if reward == 1:
                successes += 1
            elif reward == -1:
                fails += 1
